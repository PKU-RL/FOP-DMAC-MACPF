import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmac import FOPMixer,state_Mixer,state_obs_Mixer
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop
import numpy as np
from torch.distributions import Categorical
from modules.critics.dmac import FOPCritic,OffPGCritic
from utils.rl_utils import build_td_lambda_targets
from controllers import REGISTRY as mac_REGISTRY
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
class DMAC_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac

        if not self.args.check:
            if self.args.diff_target_init:
                groups = {
                    "agents": args.n_agents
                }
                self.target_mac = mac_REGISTRY[args.mac](scheme, groups, args=args)
                self.target_mac.agent_output_type = 'pi_logits'
            else:
                self.target_mac = copy.deepcopy(mac)


        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.last_target_update_step = 0
        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        if self.args.use_DOP_critic:
            critic_module = OffPGCritic
        else:
            critic_module = FOPCritic

        if self.args.state_mixer:
            if self.args.state_obs_mixer:
                mixer_module = state_obs_Mixer
            else:
                mixer_module = state_Mixer
        else:
            mixer_module = FOPMixer


        self.critic1 = critic_module(scheme, args)
        if self.args.double_min:
            self.critic2 = critic_module(scheme, args)
        self.mixer1 = mixer_module(args)
        if self.args.double_min:
            self.mixer2 = mixer_module(args)
        
        self.target_mixer1 = copy.deepcopy(self.mixer1)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.agent_params = list(mac.parameters())
        self.critic_params1 = list(self.critic1.parameters()) + list(self.mixer1.parameters())
        self.p_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.c_optimiser1 = RMSprop(params=self.critic_params1, lr=args.c_lr, alpha=args.optim_alpha,
                                    eps=args.optim_eps)

        if self.args.double_min:
            self.target_mixer2 = copy.deepcopy(self.mixer2)
            self.target_critic2 = copy.deepcopy(self.critic2)
            self.critic_params2 = list(self.critic2.parameters()) + list(self.mixer2.parameters())
            self.c_optimiser2 = RMSprop(params=self.critic_params2, lr=args.c_lr, alpha=args.optim_alpha, eps=args.optim_eps)
    def train_actor_DOP(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        actions = batch["actions"][:, :-1]
        # print('batch_action = {}'.format(actions.shape))
        terminated = batch["terminated"][:, :-1].float()
        avail_actions = batch["avail_actions"][:, :-1]
        mask_init = batch["filled"][:, :-1].float()
        mask_init[:, 1:] = mask_init[:, 1:] * (1 - terminated[:, :-1])
        mask = mask_init.repeat(1, 1, self.n_agents).view(-1)
        states = batch["state"][:, :-1]
        obs = batch["obs"][:, :-1]

        #build q
        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals = self.critic1.forward(inputs).detach()[:, :-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        sample_prob = mac_out.clone().detach()
        sample_prob[avail_actions == 0] = 0

        sample_prob_sum = sample_prob.clone().detach()
        # print('prob_sum_shape_1 = {}'.format(prob_sum.shape))
        sample_prob_sum = sample_prob_sum.sum(dim=-1)
        # print('prob_sum_shape_2 = {}'.format(prob_sum.shape))
        sample_prob_sum = sample_prob_sum.reshape([-1])
        # print('prob_sum_shape_3 = {}'.format(prob_sum.shape))
        for i in range(len(sample_prob_sum)):
            if sample_prob_sum[i] == 0:
                sample_prob_sum[i] = 1
        sample_prob_sum = sample_prob_sum.reshape(*sample_prob.shape[:-1]).unsqueeze(-1)

        sample_prob = sample_prob / sample_prob_sum
        sample_prob[avail_actions == 0] = 0
        mask_for_action = mask_init.unsqueeze(3).repeat(1, 1, self.n_agents, avail_actions.shape[-1])
        sample_prob[mask_for_action == 0] = 1

        actions = Categorical(sample_prob).sample().long().unsqueeze(3)

        pi = mac_out.view(-1, self.n_actions)
        log_pi = th.log(pi)
        entropies = - (pi * log_pi).sum(dim=-1)


        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0


        # print('sample_action = {}'.format(actions.shape))
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        # print('target_mac !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(agent_outs)
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        target_mac_out[avail_actions == 0] = 0
        target_mac_out = target_mac_out / target_mac_out.sum(dim=-1, keepdim=True)
        target_mac_out[avail_actions == 0] = 0

        # Calculated baseline
        q_taken = th.gather(q_vals, dim=3, index=actions).squeeze(3)
        pi = mac_out.view(-1, self.n_actions)

        baseline = th.sum(mac_out * q_vals, dim=-1).view(-1).detach()

        target_pi = target_mac_out.view(-1, self.n_actions)

        # Calculate policy grad with mask
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        target_pi_taken = th.gather(target_pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        target_pi_taken[mask == 0] = 1.0
        log_target_pi_taken = th.log(target_pi_taken)


        curr_bias = log_pi_taken - log_target_pi_taken

        curr_action_prob_for_log = mac_out.clone().detach()
        curr_action_prob_for_log[avail_actions == 0] = 1.0
        curr_action_prob_for_log[mask_for_action == 0] = 1.0

        curr_target_action_prob_for_log = target_mac_out.clone().detach()
        curr_target_action_prob_for_log[avail_actions == 0] = 1.0
        curr_target_action_prob_for_log[mask_for_action == 0] = 1.0


        all_log_diff = th.log(curr_action_prob_for_log) - th.log(curr_target_action_prob_for_log)

        bias_baseline = (all_log_diff * mac_out).sum(dim=3, keepdim=True).squeeze(3).detach()
        bias_baseline = bias_baseline.view(-1)
        curr_bias -= bias_baseline
        curr_bias /= self.args.reward_scale  # (32,50,5)
        alpha = 1/ self.args.reward_scale
        if  self.args.state_mixer:
            if self.args.state_obs_mixer:
                coe = self.mixer1.get_q_coeff(states,obs).view(-1)
            else:
                coe = self.mixer1.k(states).view(-1)

            # print('q_taken = {}, baseline = {}, curr_bias = {}'.format(q_taken.shape,baseline.shape,curr_bias.shape))
            advantages = (q_taken.view(-1) - baseline).detach()
            final_adv = (coe * advantages - curr_bias).detach()
        else:
            final_adv = (q_taken.view(-1) - baseline).detach()
        coma_loss = - ((final_adv * log_pi_taken) * mask).sum() / mask.sum()


        # Optimise agents
        self.p_optimiser.zero_grad()
        coma_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimiser.step()

        #compute parameters sum for debugging
        p_sum = 0.
        for p in self.agent_params:
            p_sum += p.data.abs().sum().item() / 100.0

        soft_update(self.target_mac.agent,self.mac.agent,self.args.tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("policy_loss", coma_loss.item(), t_env)
            try:
                self.logger.log_stat("agent_grad_norm", agent_grad_norm.item(), t_env)
            except:
                self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("alpha", alpha, t_env)
            self.logger.log_stat("ent", entropies.mean().item(), t_env)
    def train_actor(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # print('t_env: {} episode_num: {}'.format(t_env,episode_num))
        bs = batch.batch_size
        max_t = batch.max_seq_length
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask_init = batch["filled"][:, :-1].float()
        mask_init[:, 1:] = mask_init[:, 1:] * (1 - terminated[:, :-1])
        mask = mask_init.repeat(1, 1, self.n_agents).view(-1)
        avail_actions = batch["avail_actions"]
        avail_actions_curr = batch["avail_actions"][:, :-1]
        states = batch["state"][:,:-1]
        obs = batch['obs'][:,:-1]
        states_all = batch["state"]
        obs_all = batch['obs']
        mac = self.mac

        if not self.args.check:
            if self.args.alpha_decay:
                alpha = max(0.05, 0.5 - t_env / 200000)  # linear decay
            else:
                alpha = self.args.reg_alpha
        else:
            alpha = max(0.05, 0.5 - t_env / 200000)  # linear decay

        mac_out = []
        mac.init_hidden(batch.batch_size)
        # print('mac.hidden_states = {} mac = {}'.format(mac.hidden_states.device,next(mac.agent.parameters()).device))
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        pi = mac_out[:,:-1].clone()
        pi = pi.reshape(-1, self.n_actions)
        log_pi = th.log(pi)
        entropies = - (pi * log_pi).sum(dim=-1)

        sample_prob = mac_out.clone().detach()
        sample_prob[avail_actions == 0] = 0

        sample_prob_sum = sample_prob.clone().detach()
        # print('prob_sum_shape_1 = {}'.format(prob_sum.shape))
        sample_prob_sum = sample_prob_sum.sum(dim=-1)
        # print('prob_sum_shape_2 = {}'.format(prob_sum.shape))
        sample_prob_sum = sample_prob_sum.reshape([-1])
        # print('prob_sum_shape_3 = {}'.format(prob_sum.shape))
        for i in range(len(sample_prob_sum)):
            if sample_prob_sum[i] == 0:
                sample_prob_sum[i] = 1
        sample_prob_sum = sample_prob_sum.reshape(*sample_prob.shape[:-1]).unsqueeze(-1)


        sample_prob = sample_prob / sample_prob_sum
        sample_prob[avail_actions == 0] = 0

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        pi = mac_out[:, :-1].clone()
        pi = pi.reshape(-1, self.n_actions)
        if self.args.check:
            inputs = self.critic1._build_inputs(batch, bs, max_t)
            q_vals1 = self.critic1.forward(inputs)
            q_vals2 = self.critic2.forward(inputs)
            q_vals = th.min(q_vals1, q_vals2)

            pi = mac_out[:, :-1].reshape(-1, self.n_actions)
            entropies = - (pi * log_pi).sum(dim=-1)

            # policy target for discrete actions (from Soft Actor-Critic for Discrete Action Settings)
            pol_target = (pi * (alpha * log_pi - q_vals[:, :-1].reshape(-1, self.n_actions))).sum(dim=-1)

            policy_loss = (pol_target * mask).sum() / mask.sum()
        else:
            sample_prob = sample_prob[:, :-1]
            mask_for_action = mask_init.unsqueeze(3).repeat(1, 1, self.n_agents, avail_actions.shape[-1])
            sample_prob[mask_for_action == 0] = 1

            actions = Categorical(sample_prob).sample().long().unsqueeze(3)

            target_mac = self.target_mac

            target_mac_out = []
            target_mac.init_hidden(batch.batch_size)
            # print('target_mac.hidden_states = {} target_mac = {}'.format(target_mac.hidden_states.device,next(target_mac.agent.parameters()).device))
            for t in range(batch.max_seq_length):
                agent_outs = target_mac.forward(batch, t=t)
                target_mac_out.append(agent_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time

            # Mask out unavailable actions, renormalise (as in action selection)
            target_mac_out[avail_actions == 0] = 0
            target_mac_out = target_mac_out / target_mac_out.sum(dim=-1, keepdim=True)
            target_mac_out[avail_actions == 0] = 0


            target_pi = target_mac_out[:,:-1].clone()
            target_pi = target_pi.reshape(-1, self.n_actions)
            # target_log_pi = th.log(target_pi)

            pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
            pi_taken[mask == 0] = 1.0
            log_pi_taken = th.log(pi_taken)

            target_pi_taken = th.gather(target_pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
            target_pi_taken[mask == 0] = 1.0
            log_target_pi_taken = th.log(target_pi_taken)



            curr_bias = log_pi_taken - log_target_pi_taken
            # if self.args.check_output:
            #     for t in range(len(curr_bias)):
            #         print('t_env: {} curr_bias[{}] = {}'.format(t_env,t,curr_bias[t]))
            curr_action_prob_for_log = mac_out[:,:-1].clone().detach()
            curr_action_prob_for_log[avail_actions_curr == 0] = 1.0
            curr_action_prob_for_log[mask_for_action == 0] = 1.0

            curr_target_action_prob_for_log = target_mac_out[:,:-1].clone().detach()
            curr_target_action_prob_for_log[avail_actions_curr == 0] = 1.0
            curr_target_action_prob_for_log[mask_for_action == 0] = 1.0


            all_log_diff = th.log(curr_action_prob_for_log) - th.log(curr_target_action_prob_for_log)

            curr_pi = mac_out[:,:-1].clone().detach()
            bias_baseline = (all_log_diff * curr_pi).sum(dim=3, keepdim=True).squeeze(3).detach()


            bias_baseline = bias_baseline.view(-1)
            curr_bias -= bias_baseline
            curr_bias *= alpha  # (32,50,5)


            inputs = self.critic1._build_inputs(batch, bs, max_t)



            if self.args.double_min:
                q_vals1 = self.critic1.forward(inputs)
                q_vals2 = self.critic2.forward(inputs)
                if self.args.state_mixer:
                    if self.args.state_obs_mixer:
                        coeff1 = self.mixer1.get_q_coeff(states_all,obs_all).unsqueeze(-1)
                        coeff2 = self.mixer2.get_q_coeff( states_all,obs_all).unsqueeze(-1)
                    else:
                        coeff1 = self.mixer1.k( states_all).unsqueeze(-1)
                        coeff2 = self.mixer2.k( states_all).unsqueeze(-1)
                    # print('coeff = {} q_vals = {}'.format(coeff1.shape,q_vals1.shape))
                    q_vals1 = q_vals1 * coeff1
                    q_vals2 = q_vals2 * coeff2
                q_vals = th.min(q_vals1, q_vals2)
            else:
                q_vals = self.critic1.forward(inputs)
                if self.args.state_mixer:
                    if self.args.state_obs_mixer:
                        coeff1 = self.mixer1.get_q_coeff(states_all,obs_all).unsqueeze(-1)
                    else:
                        coeff1 = self.mixer1.k( states_all).unsqueeze(-1)

                    # print('coeff = {} q_vals = {}'.format(coeff1.shape,q_vals1.shape))
                    q_vals = q_vals * coeff1
            # if self.args.check_output:
            #     for b in range(bs):
            #         for t in range(max_t - 1):
            #             print('t_env: {} curr_pi[{},{}] = {}'.format(t_env, b, t, curr_pi[b, t]))
            #             print('t_env: {} q_vals[{},{}] = {}'.format(t_env, b, t, q_vals[b, t]))
            #             print('t_env: {} all_log_diff[{},{}] = {}'.format(t_env, b, t, all_log_diff[b, t]))
            q_vals_table =q_vals[:,:-1]
            q_vals = q_vals[:, :-1].reshape(-1, self.n_actions)

            if self.args.q_table:
                vs = curr_pi * q_vals_table
                vs = th.sum(vs,dim = -1)
                q_taken = th.gather(q_vals_table, dim=1, index=actions).squeeze(-1)

                actions_onehot = th.zeros(actions.squeeze(3).shape + (self.n_actions,))
                if self.args.use_cuda:
                    actions_onehot = actions_onehot.cuda()
                actions_onehot = actions_onehot.scatter_(3, actions, 1)

                q_taken_mix,_ = self.mixer1(q_taken, states, actions=actions_onehot, vs=vs)


                q_table = th.zeros_like(q_vals_table).to(q_vals_table.device)
                acs_eye = th.eye(self.n_actions).to(q_vals_table.device)
                acs_eye = acs_eye.unsqueeze(1).unsqueeze(1).repeat([1,q_vals_table.shape[0],q_vals_table.shape[1],1 ])
                # print('actions_onehot = {} acs_eys = {}'.format(actions_onehot.shape,acs_eye.shape))
                for a_i in range(self.n_agents):
                    for ac in range(self.n_actions):
                        ac_input = actions_onehot.clone().detach()
                        # print('u_tmp[:,:,{}:{}] = {}'.format(a_i * self.n_actions,(a_i + 1) * self.n_actions, u_tmp[:,:,a_i * self.n_actions: (a_i + 1) * self.n_actions].shape))
                        ac_input[:, :, a_i] = acs_eye[a_i]
                        q_input = q_taken.clone().detach()
                        q_input[:, :, a_i] = q_vals_table[:, :, a_i, ac]
                        mix_ret, _ = self.mixer1(q_input, states, actions=ac_input,vs = vs)
                        q_table[:, :, a_i, ac] = mix_ret.squeeze(-1)
                baseline = q_table * curr_pi
                baseline = th.sum(baseline,dim = -1)

                q_adv = q_taken_mix - baseline
                q_adv = q_adv.reshape([-1])
                adv = (q_adv - curr_bias).detach()
            else:
                q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
                baseline = th.sum(pi * q_vals, dim=-1).view(-1).detach()

                adv = (q_taken - baseline - curr_bias).detach()






            pol_target = -log_pi_taken * adv

            policy_loss = (pol_target * mask).sum() / mask.sum()
            if self.args.check_output:
                print('actor: t_env: {} policy_loss = {}'.format(t_env, t, policy_loss))
                for t in range(len(log_pi_taken)):
                    print('actor: t_env: {} log_pi_taken[{}] = {}'.format(t_env, t, log_pi_taken[t]))
                    print('actor: t_env: {} adv[{}] = {}'.format(t_env, t, adv[t]))

        # Optimise
        self.p_optimiser.zero_grad()
        policy_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimiser.step()

        # print('t_env: {} policy_loss = {}'.format(t_env,policy_loss))
        if not self.args.check:
            soft_update(self.target_mac.agent, self.mac.agent, self.args.tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval or self.args.check_output:
            self.logger.log_stat("policy_loss", policy_loss.item(), t_env)
            try:
                self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            except:
                self.logger.log_stat("agent_grad_norm", agent_grad_norm.item(), t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("alpha", alpha, t_env)
            self.logger.log_stat("ent", entropies.mean().item(), t_env)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        if self.args.DOP_func:
            self.train_actor_DOP(batch, t_env, episode_num)
            self.train_critic_DOP(batch, t_env, episode_num)
        else:
            self.train_actor(batch, t_env, episode_num)
            self.train_critic(batch, t_env)
            if self.args.soft_update_critic:
                self._soft_update_targets()
            else:
                if self.args.DOP_training:
                    target_update_interval = self.args.target_update_interval
                else:
                    target_update_interval = self.args.DOP_target_update_interval
                if (self.critic_training_steps - self.last_target_update_step) / target_update_interval >= 1.0:
                    self._update_targets()
                    self.last_target_update_step = self.critic_training_steps

    def train_critic_DOP(self, batch, t_env, episode_num):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        actions_onehot = batch['actions_onehot']
        terminated = batch["terminated"][:, :-1].float()
        terminated_all = batch["terminated"][:, :].float()
        avail_actions = batch["avail_actions"][:]
        states = batch["state"]
        obs = batch["obs"]
        mask = batch["filled"][:, :-1].float()
        mask_all = batch["filled"][:, :].float()


        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask_all[:, 1:] = mask_all[:, 1:] * (1 - terminated_all[:, :-1])
        mask_all_for_action = mask_all.unsqueeze(3).repeat(1, 1, self.n_agents, avail_actions.shape[-1])
        mask_all = mask_all.repeat(1,1,self.n_agents)


        mac_out = []
        self.mac.init_hidden(bs)
        for i in range(max_t):
            agent_outs = self.mac.forward(batch, t=i)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1).detach()
        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        target_mac_out = []
        self.target_mac.init_hidden(bs)
        for i in range(max_t):
            agent_outs = self.target_mac.forward(batch, t=i)
            target_mac_out.append(agent_outs)
        target_mac_out = th.stack(target_mac_out, dim=1).detach()
        # Mask out unavailable actions, renormalise (as in action selection)
        target_mac_out[avail_actions == 0] = 0
        target_mac_out = target_mac_out / target_mac_out.sum(dim=-1, keepdim=True)
        target_mac_out[avail_actions == 0] = 0

        pi = mac_out.clone().detach()
        target_pi = target_mac_out.clone().detach()
        pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
        target_pi_taken = th.gather(target_pi, dim=3, index=actions).squeeze(3)
        pi_taken[mask_all == 0] = 1.0
        target_pi_taken[mask_all == 0] = 1.0


        bias_sum = th.log(pi_taken) - th.log(target_pi_taken)
        bias_sum = bias_sum.sum(dim=2, keepdim=True)
        bias_sum /= self.args.reward_scale
        alpha = 1/self.args.reward_scale
        #build_target_q
        target_inputs = self.target_critic1._build_inputs(batch, bs, max_t)
        target_q_vals = self.target_critic1.forward(target_inputs).detach()

        if self.args.state_mixer:
            target_q_inputs = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)
            if self.args.state_obs_mixer:
                next_vs = target_q_vals * pi
                next_vs = th.sum(next_vs, dim=-1)
                targets_taken, _ = self.target_mixer1(target_q_inputs , states, obs=obs,
                                                      vs=next_vs)
            else:
                targets_taken = self.target_mixer1(target_q_inputs, states)
        else:
            if self.args.mean_vs:
                next_vs = target_q_vals * pi
                next_vs = th.sum(next_vs,dim = -1)
            else:
                next_vs = th.exp(target_q_vals / alpha)
                next_vs = target_pi * next_vs
                next_vs[mask_all_for_action == 0] = 1.0
                next_vs = th.log(th.sum(next_vs, dim=-1)) * alpha



            idv_q_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)
            targets_taken,_ = self.target_mixer1(idv_q_taken, states,actions=actions_onehot,
                                                      vs=next_vs)

        target_q = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda,bias_sum).detach()

        inputs = self.critic1._build_inputs(batch, bs, max_t)



        #train critic
        loss1 = 0
        masked_td_error1 = []
        q_taken1 = []
        for t in range(max_t - 1):
            mask_t = mask[:, t:t+1]
            if mask_t.sum() < 0.5:
                continue
            q_vals_t = self.critic1.forward(inputs[:, t:t + 1])
            idv_q_taken_t = th.gather(q_vals_t, 3, index=actions[:, t:t + 1]).squeeze(3)
            if self.args.state_mixer:
                if self.args.state_obs_mixer:
                    vs_t = pi[:, t:t + 1] * q_vals_t
                    vs_t = th.sum(vs_t, dim=-1)
                    q_taken_t,q_attend_regs = self.mixer1.forward(idv_q_taken_t, states[:, t:t + 1],obs = obs[:, t:t + 1],vs = vs_t)
                else:
                    q_taken_t = self.mixer1.forward(idv_q_taken_t, states[:, t:t+1])
                    q_attend_regs = 0
            else:
                if self.args.mean_vs:
                    vs_t = pi[:, t:t + 1] * q_vals_t
                    vs_t = th.sum(vs_t, dim=-1)
                else:
                    vs_t = th.exp(q_vals_t / alpha)
                    vs_t = target_pi[:, t:t + 1] * vs_t

                    vs_t[mask_all_for_action[:, t:t + 1] == 0] = 1.0

                    vs_t = th.log(th.sum(vs_t, dim=-1)) * alpha

                q_taken_t, q_attend_regs = self.mixer1(idv_q_taken_t, states[:, t:t + 1], actions=actions_onehot[:, t:t + 1],
                                                  vs=vs_t)

            target_q_t = target_q[:, t:t+1].detach()
            q_err = (q_taken_t - target_q_t) * mask_t
            critic_loss = (q_err ** 2).sum() / mask_t.sum() + q_attend_regs
            #Here introduce the loss for Qi

            #critic_loss += goal_loss
            self.c_optimiser1.zero_grad()
            critic_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params1, self.args.grad_norm_clip)
            self.c_optimiser1.step()
            self.critic_training_steps += 1
            masked_td_error1.append(q_err)
            q_taken1.append(q_taken_t)
            loss1 += critic_loss

        masked_td_error1 = th.stack(masked_td_error1, dim=1)
        q_taken1 = th.stack(q_taken1, dim=1).squeeze(-1)
        #update target network
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss1.item(), t_env)
            try:
                self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            except:
                self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
            # print('q_taken1 = {},mask = {}'.format(q_taken1.shape,mask.shape))
            self.logger.log_stat("q_taken_mean",
                                 (q_taken1 * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (target_q * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env
        if self.args.soft_update_critic:
            self._soft_update_targets()
        else:
            if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
                print_update_target = (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0
                self._update_targets(print_update_target)
                self.last_target_update_step = self.critic_training_steps
                if print_update_target:
                    self.last_target_update_episode = episode_num
    def train_critic(self, batch, t_env):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]

        terminated_all = batch["terminated"].float()
        mask_all = batch["filled"].float()
        mask_all[:,1:] =  mask_all[:,1:]*(1 - terminated_all[:, :-1])

        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask_init = batch["filled"][:, :-1].float()
        mask_init[:, 1:] = mask_init[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        states = batch["state"]
        obs = batch["obs"]
        actions_all = batch["actions"][:, :]
        actions_onehot_all = batch["actions_onehot"]
        mask_for_action = mask_init.unsqueeze(3).repeat(1, 1, self.n_agents, avail_actions.shape[-1])
        mask_all_for_action = mask_all.unsqueeze(3).repeat(1, 1, self.n_agents, avail_actions.shape[-1])
        mac = self.mac

        mixer1 = self.mixer1
        if self.args.double_min:
            mixer2 = self.mixer2

        if not self.args.check:
            if self.args.alpha_decay:
                alpha = max(0.05, 0.5 - t_env / 200000)  # linear decay
            else:
                alpha = self.args.reg_alpha
        else:
            alpha = max(0.05, 0.5 - t_env / 200000)  # linear decay

        mac_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions
        mac_out[avail_actions == 0] = 0.0

        if self.args.next_sample:
            sample_prob = mac_out.clone().detach()
            sample_prob_sum = sample_prob.clone().detach()
            # print('prob_sum_shape_1 = {}'.format(prob_sum.shape))
            sample_prob_sum = sample_prob_sum.sum(dim=-1)
            # print('prob_sum_shape_2 = {}'.format(prob_sum.shape))
            sample_prob_sum = sample_prob_sum.reshape([-1])
            # print('prob_sum_shape_3 = {}'.format(prob_sum.shape))
            for i in range(len(sample_prob_sum)):
                if sample_prob_sum[i] == 0:
                    sample_prob_sum[i] = 1
            sample_prob_sum = sample_prob_sum.reshape(*sample_prob.shape[:-1]).unsqueeze(-1)
            sample_prob = sample_prob / sample_prob_sum
            # sample_prob[avail_actions == 0] = 0

            sample_prob[mask_all_for_action == 0] = 1
            # if self.args.check_output:
            #     for b in range(bs):
            #         for t in range(max_t):
            #             print('sample_prob[{},{}] = {}'.format(b,t,sample_prob[b,t]))

        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        if self.args.check_output:
            for b in range(bs):
                for t in range(max_t - 1):
                    print('critic: t_env: {} mac_out[{},{}] = {}'.format(t_env, b, t,  mac_out[b, t]))
                    print('critic: t_env: {} sample_prob[{},{}] = {}'.format(t_env, b, t, sample_prob[b, t]))


        t_mac_out = mac_out.clone().detach()
        pi = t_mac_out

        if not self.args.check:
            target_mac = self.target_mac
            target_mac_out = []
            target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = target_mac.forward(batch, t=t)
                target_mac_out.append(agent_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time

            # Mask out unavailable actions
            target_mac_out[avail_actions == 0] = 0.0
            target_mac_out = target_mac_out / target_mac_out.sum(dim=-1, keepdim=True)
            target_mac_out[avail_actions == 0] = 0

            target_t_mac_out = target_mac_out.clone().detach()
            target_pi = target_t_mac_out


        # sample actions for next timesteps
        if self.args.next_sample:
            next_actions = Categorical(sample_prob).sample().long().unsqueeze(3)
            next_actions_onehot = th.zeros(next_actions.squeeze(3).shape + (self.n_actions,))
            if self.args.use_cuda:
                try:
                    next_actions_onehot = next_actions_onehot.cuda()
                except:
                    print('sample_prob = {}'.format(sample_prob))
                    print('sample_prob_check = {}'.format( (sample_prob>=0).all()  ))
            next_actions_onehot = next_actions_onehot.scatter_(3, next_actions, 1)
        else:
            next_actions = actions_all
            next_actions_onehot = actions_onehot_all


        if not self.args.check:
            pi_taken = th.gather(pi, dim=3, index=next_actions).squeeze(3)
            pi_taken[mask_all.expand_as(pi_taken) == 0] = 1.0
            log_pi_taken = th.log(pi_taken)
            target_pi_taken = th.gather(target_pi, dim=3, index=next_actions).squeeze(3)
            target_pi_taken[mask_all.expand_as(target_pi_taken) == 0] = 1.0
            target_log_pi_taken = th.log(target_pi_taken)
            bias_sum = log_pi_taken - target_log_pi_taken
            bias_sum = bias_sum.sum(dim=2, keepdim=True)
            bias_sum *= alpha
        else:
            pi_taken = th.gather(pi, dim=3, index=next_actions).squeeze(3)[:, 1:]
            pi_taken[mask.expand_as(pi_taken) == 0] = 1.0
            log_pi_taken = th.log(pi_taken)

        target_inputs = self.target_critic1._build_inputs(batch, bs, max_t)
        if not self.args.check and not self.args.double_min:
            target_q_vals = self.target_critic1.forward(target_inputs).detach()
        else:
            target_q_vals1 = self.target_critic1.forward(target_inputs).detach()
            target_q_vals2 = self.target_critic2.forward(target_inputs).detach()
        if self.args.check_output:
            for b in range(bs):
                for t in range(max_t ):
                    print('critic: t_env: {} target_q[{},{}] = {}'.format(t_env, b, t, target_q_vals[b, t]))

        # directly caculate the values by definition
        if not self.args.state_mixer:
            if not self.args.check:
                if not self.args.mean_vs:
                    if self.args.double_min:
                        next_vs1 = th.exp(target_q_vals1 / alpha)
                        next_vs1 = target_pi * next_vs1

                        next_vs1[mask_all_for_action == 0] = 1.0

                        next_vs1 = th.log(th.sum(next_vs1, dim=-1)) * alpha
                        next_vs2 = th.exp(target_q_vals2 / alpha)
                        next_vs2 = target_pi * next_vs2

                        next_vs2[mask_all_for_action == 0] = 1.0

                        next_vs2 = th.log(th.sum(next_vs2, dim=-1)) * alpha
                    else:
                        next_vs = th.exp(target_q_vals / alpha)
                        next_vs = target_pi * next_vs

                        next_vs[mask_all_for_action == 0] = 1.0

                        next_vs = th.log(th.sum(next_vs, dim=-1)) * alpha
                else:
                    if self.args.double_min:

                        next_vs1 = pi * target_q_vals1
                        next_vs1 = th.sum(next_vs1,dim = -1)

                        next_vs2 = pi * target_q_vals2
                        next_vs2 = th.sum(next_vs2, dim=-1)
                    else:
                        next_vs = pi * target_q_vals
                        next_vs = th.sum(next_vs, dim=-1)

            else:
                next_vs1 = th.logsumexp(target_q_vals1 / alpha, dim=-1) * alpha
                next_vs2 = th.logsumexp(target_q_vals2 / alpha, dim=-1) * alpha
        else:
            if self.args.state_obs_mixer:
                if self.args.double_min:

                    next_vs1 = pi * target_q_vals1
                    next_vs1 = th.sum(next_vs1, dim=-1)

                    next_vs2 = pi * target_q_vals2
                    next_vs2 = th.sum(next_vs2, dim=-1)
                else:
                    next_vs = pi * target_q_vals
                    next_vs = th.sum(next_vs, dim=-1)

        if not self.args.check and not self.args.double_min:
            next_chosen_qvals = th.gather(target_q_vals, dim=3, index=next_actions).squeeze(3)
            if self.args.state_mixer:
                if self.args.state_obs_mixer:
                    target_qvals, _ = self.target_mixer1(next_chosen_qvals, states, obs=obs,
                                                         vs=next_vs)
                else:
                    target_qvals = self.target_mixer1(next_chosen_qvals, states)
            else:
                target_qvals, _ = self.target_mixer1(next_chosen_qvals, states, actions=next_actions_onehot,
                                                      vs=next_vs)

        else:
            next_chosen_qvals1 = th.gather(target_q_vals1, dim=3, index=next_actions).squeeze(3)
            next_chosen_qvals2 = th.gather(target_q_vals2, dim=3, index=next_actions).squeeze(3)
            if self.args.state_mixer:
                if self.args.state_obs_mixer:
                    target_qvals1 = self.target_mixer1(next_chosen_qvals1, states,obs=obs,vs=next_vs1)
                    target_qvals2 = self.target_mixer2(next_chosen_qvals2, states,obs=obs,vs=next_vs2)
                else:
                    target_qvals1 = self.target_mixer1(next_chosen_qvals1, states)
                    target_qvals2 = self.target_mixer2(next_chosen_qvals2, states)

            else:
                target_qvals1, _ = self.target_mixer1(next_chosen_qvals1, states, actions=next_actions_onehot, vs=next_vs1)
                target_qvals2, _ = self.target_mixer2(next_chosen_qvals2, states, actions=next_actions_onehot, vs=next_vs2)

            target_qvals = th.min(target_qvals1, target_qvals2)

        if self.args.check_output:
            for b in range(bs):
                for t in range(max_t):
                    print('critic: t_env: {} next_chosen_qvals[{},{}] = {}'.format(t_env, b, t,  next_chosen_qvals[b, t]))
                    print(
                        'critic: t_env: {} target_qvals[{},{}] = {}'.format(t_env, b, t, target_qvals[b, t]))

        # Calculate td-lambda targets
        if not self.args.check:
            targets = build_td_lambda_targets(rewards, terminated, mask,target_qvals, self.n_agents, self.args.gamma,
                                               self.args.td_lambda, bias_sum=bias_sum).detach()
        else:
            target_v = build_td_lambda_targets(rewards, terminated, mask, target_qvals, self.n_agents, self.args.gamma, self.args.td_lambda)
            targets = target_v - alpha * log_pi_taken.mean(dim=-1, keepdim=True)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        if not self.args.DOP_training:
            if not self.args.check  and not self.args.double_min:
                q_vals = self.critic1.forward(inputs)
            else:
                q_vals1 = self.critic1.forward(inputs)
                q_vals2 = self.critic2.forward(inputs)

            # directly caculate the values by definition
            if not self.args.state_mixer:
                if not self.args.check:
                    if self.args.mean_vs:
                        if self.args.double_min:
                            vs1 = pi * q_vals1
                            vs1 = th.sum(vs1, dim=-1)

                            vs2 = pi * q_vals2
                            vs2 = th.sum(vs2, dim=-1)
                        else:
                            vs = pi * q_vals
                            vs = th.sum(vs, dim=-1)
                    else:
                        if self.args.double_min:
                            vs1 = th.exp(q_vals1 / alpha)
                            vs1 = target_pi * vs1

                            vs1[mask_all_for_action == 0] = 1.0

                            vs1 = th.log(th.sum(vs1, dim=-1)) * alpha

                            vs2 = th.exp(q_vals2 / alpha)
                            vs2 = target_pi * vs2

                            vs2[mask_all_for_action == 0] = 1.0

                            vs2 = th.log(th.sum(vs2, dim=-1)) * alpha

                        else:
                            vs = th.exp(q_vals / alpha)
                            vs = target_pi * vs
                            if self.args.check_output:
                                for b in range(bs):
                                    for t in range(max_t):
                                        print('critic: t_env: {} target_pi[{},{}] = {}'.format(t_env, b, t,
                                                                                                       target_pi[
                                                                                                           b, t]))
                                        print(
                                            'critic: t_env: {} vs_1[{},{}] = {}'.format(t_env, b, t,
                                                                                                vs[b, t]))

                            vs[mask_all_for_action == 0] = 1.0
                            vs = th.log(th.sum(vs, dim=-1)) * alpha
                            if self.args.check_output:
                                for b in range(bs):
                                    for t in range(max_t):
                                        print(
                                            'critic: t_env: {} vs_2[{},{}] = {}'.format(t_env, b, t,
                                                                                                vs[b, t]))

                else:
                    vs1 = th.logsumexp(q_vals1 / alpha, dim=-1) * alpha
                    vs2 = th.logsumexp(q_vals2 / alpha, dim=-1) * alpha
            else:
                if self.args.state_obs_mixer:
                    if self.args.double_min:
                        vs1 = pi * q_vals1
                        vs1 = th.sum(vs1, dim=-1)

                        vs2 = pi * q_vals2
                        vs2 = th.sum(vs2, dim=-1)
                    else:
                        vs = pi * q_vals
                        vs = th.sum(vs, dim=-1)


        if not self.args.check and not self.args.double_min:
            loss1 = 0
            masked_td_error1 = []
            q_taken1 = []
            if not self.args.DOP_training:
                q_taken = th.gather(q_vals[:, :-1], dim=3, index=next_actions[:,:-1]).squeeze(3)
                if self.args.state_mixer:
                    if self.args.state_obs_mixer:
                        q_taken = mixer1(q_taken, states[:, :-1],obs=obs[:,:-1],vs = vs)
                        q_attend_regs = 0
                    else:
                        q_taken = mixer1(q_taken, states[:, :-1])
                        q_attend_regs = 0
                else:
                    q_taken, q_attend_regs = mixer1(q_taken, states[:, :-1], actions=next_actions_onehot[:,:-1], vs=vs[:, :-1])

                td_error = q_taken - targets.detach()
                mask = mask.expand_as(td_error)
                # 0-out the targets that came from padded data
                masked_td_error = td_error * mask
                loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs

                # Optimise
                self.c_optimiser1.zero_grad()
                loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params1, self.args.grad_norm_clip)
                self.c_optimiser1.step()
                loss1 = loss
                masked_td_error1 = masked_td_error
                q_taken1 = q_taken
                self.critic_training_steps += 1
            else:
                for t in range(batch.max_seq_length - 1):
                    mask_t = mask[:, t:t + 1]
                    if mask_t.sum() < 0.5:
                        continue
                    q_vals_t = self.critic1.forward(inputs[:, t:t+1])
                    q_taken_t = th.gather(q_vals_t, 3, index=next_actions[:, t:t+1]).squeeze(3)
                    if self.args.state_mixer:
                        if self.args.state_obs_mixer:
                            vs_t = pi[:, t:t + 1] * q_vals_t
                            vs_t = th.sum(vs_t, dim=-1)
                            q_taken_t,q_attend_regs  = mixer1(q_taken_t, states[:, t:t + 1],obs = obs[:,t:t + 1],vs = vs_t)
                        else:
                            q_taken_t = mixer1(q_taken_t, states[:, t:t + 1])
                            q_attend_regs = 0
                    else:
                        if not self.args.check:
                            if self.args.mean_vs:
                                vs_t = pi[:,t:t+1] * q_vals_t
                                vs_t = th.sum(vs_t,dim = -1)
                            else:
                                vs_t = th.exp(q_vals_t / alpha)
                                vs_t = target_pi[:,t:t+1] * vs_t

                                vs_t[mask_all_for_action[:,t:t+1] == 0] = 1.0

                                vs_t = th.log(th.sum(vs_t, dim=-1)) * alpha
                        else:
                            vs_t = th.logsumexp(q_vals_t  / alpha, dim=-1) * alpha
                        q_taken_t,q_attend_regs =  mixer1(q_taken_t, states[:, t:t + 1],actions=next_actions_onehot[:,t:t + 1], vs=vs_t)


                    target_q_t = targets[:, t:t + 1].detach()
                    # print('q_taken_t = {}, target_q_t = {} mask_t = {}'.format(q_taken_t.shape,target_q_t.shape,mask_t.shape))
                    q_err = (q_taken_t - target_q_t) * mask_t
                    critic_loss = (q_err ** 2).sum() / mask_t.sum() + q_attend_regs
                    self.c_optimiser1.zero_grad()
                    critic_loss.backward()
                    grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params1, self.args.grad_norm_clip)
                    self.c_optimiser1.step()
                    loss1 += critic_loss

                    self.critic_training_steps += 1
                    masked_td_error1.append(q_err)
                    q_taken1.append(q_taken_t)
                masked_td_error1 = th.stack(masked_td_error1,dim = 1)
                q_taken1 = th.stack(q_taken1,dim = 1).squeeze(-1)
        else:
            q_taken1 = th.gather(q_vals1[:,:-1], dim=3, index=actions).squeeze(3)
            q_taken2 = th.gather(q_vals2[:,:-1], dim=3, index=actions).squeeze(3)
            if self.args.state_mixer:
                if self.args.state_obs_mixer:
                    q_taken1,q_attend_regs1 = mixer1(q_taken1, states[:, :-1],obs = obs[:,:-1],vs = vs1)
                    q_taken2,q_attend_regs2 = mixer2(q_taken2, states[:, :-1],obs = obs[:,:-1],vs = vs2)

                else:
                    q_taken1 = mixer1(q_taken1, states[:, :-1])
                    q_taken2 = mixer2(q_taken2, states[:, :-1])
                    q_attend_regs1 = 0
                    q_attend_regs2 = 0
            else:
                q_taken1, q_attend_regs1 = mixer1(q_taken1, states[:, :-1], actions=actions_onehot, vs=vs1[:, :-1])
                q_taken2, q_attend_regs2 = mixer2(q_taken2, states[:, :-1], actions=actions_onehot, vs=vs2[:, :-1])

            td_error1 = q_taken1 - targets.detach()
            td_error2 = q_taken2 - targets.detach()

            mask = mask.expand_as(td_error1)

            # 0-out the targets that came from padded data
            masked_td_error1 = td_error1 * mask
            loss1 = (masked_td_error1 ** 2).sum() / mask.sum() + q_attend_regs1
            masked_td_error2 = td_error2 * mask
            loss2 = (masked_td_error2 ** 2).sum() / mask.sum() + q_attend_regs2
        
        # Optimise
            self.c_optimiser1.zero_grad()
            loss1.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params1, self.args.grad_norm_clip)
            self.c_optimiser1.step()

            self.c_optimiser2.zero_grad()
            loss2.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params2, self.args.grad_norm_clip)
            self.c_optimiser2.step()
        if self.args.check_output:
            print('critic:  t_env {} q_loss = {}'.format(t_env,loss1))

        if t_env - self.log_stats_t >= self.args.learner_log_interval or self.args.check_output:
            self.logger.log_stat("loss", loss1.item(), t_env)
            try:
                self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            except:
                self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
            # print('q_taken1 = {},mask = {}'.format(q_taken1.shape,mask.shape))
            self.logger.log_stat("q_taken_mean",
                                 (q_taken1 * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _soft_update_targets(self, bool_print=True):
        soft_update(self.target_critic1, self.critic1, self.args.tau)
        soft_update(self.target_mixer1, self.mixer1, self.args.tau)

        if self.args.double_min:
            soft_update(self.target_critic2, self.critic2, self.args.tau)
            soft_update(self.target_mixer2, self.mixer2, self.args.tau)
        if bool_print:
            self.logger.console_logger.info("Updated target network")
    def _update_targets(self, bool_print=True):

        self.target_critic1.load_state_dict(self.critic1.state_dict())

        self.target_mixer1.load_state_dict(self.mixer1.state_dict())

        if self.args.double_min:
            self.target_critic2.load_state_dict(self.critic2.state_dict())
            self.target_mixer2.load_state_dict(self.mixer2.state_dict())
        if bool_print:
            self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        if not self.args.check:
            self.target_mac.cuda()
        self.critic1.cuda()
        self.mixer1.cuda()
        self.target_critic1.cuda()
        self.target_mixer1.cuda()
        if self.args.double_min:
            self.critic2.cuda()
            self.mixer2.cuda()
            self.target_critic2.cuda()
            self.target_mixer2.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic1.state_dict(), "{}/critic1.th".format(path))
        th.save(self.mixer1.state_dict(), "{}/mixer1.th".format(path))
        th.save(self.p_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.c_optimiser1.state_dict(), "{}/critic_opt1.th".format(path))
        if self.args.double_min:
            th.save(self.critic2.state_dict(), "{}/critic2.th".format(path))
            th.save(self.mixer2.state_dict(), "{}/mixer2.th".format(path))
            th.save(self.c_optimiser2.state_dict(), "{}/critic_opt2.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic1.load_state_dict(th.load("{}/critic1.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.mixer1.load_state_dict(th.load("{}/mixer1.th".format(path), map_location=lambda storage, loc: storage))
        self.p_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser1.load_state_dict(th.load("{}/critic_opt1.th".format(path), map_location=lambda storage, loc: storage))

        if self.args.double_min:
            self.critic_optimiser2.load_state_dict(th.load("{}/critic_opt2.th".format(path), map_location=lambda storage, loc: storage))
            self.mixer2.load_state_dict(th.load("{}/mixer2.th".format(path), map_location=lambda storage, loc: storage))
            self.critic2.load_state_dict(th.load("{}/critic2.th".format(path), map_location=lambda storage, loc: storage))
            self.target_critic2.load_state_dict(self.critic2.state_dict())
    def build_inputs(self, batch, bs, max_t, actions_onehot):
        inputs = []
        inputs.append(batch["obs"][:])
        actions = actions_onehot[:].reshape(bs, max_t, self.n_agents, -1)
        inputs.append(actions)
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
