import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dfop import DFOPMixer
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop
import numpy as np
from torch.distributions import Categorical
from modules.critics.dfop import DFOPCritic
from utils.rl_utils import build_td_lambda_targets


class DFOP_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.last_target_update_episode = 0
        self.critic_training_steps = 0
        self.alpha_start = 0.001
        self.alpha_end = 0.001
        self.alpha_decay_end = 200000

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic1 = DFOPCritic(scheme, args)
        self.mixer1 = DFOPMixer(args)

        self.mixer_dep1 = DFOPMixer(args)
        
        self.target_mixer1 = copy.deepcopy(self.mixer1)

        self.target_mixer_dep1 = copy.deepcopy(self.mixer_dep1)
     
        self.target_critic1 = copy.deepcopy(self.critic1)
        
        self.agent_params = list(mac.parameters())
        self.critic_params1 = list(self.critic1.parameters()) + list(self.mixer1.parameters()) + list(self.mixer_dep1.parameters())
 
        self.p_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.c_optimiser1 = RMSprop(params=self.critic_params1, lr=args.c_lr, alpha=args.optim_alpha, eps=args.optim_eps)


    def train_actor(self, batch: EpisodeBatch, t_env: int, episode_num: int, dep_mode: bool):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask_org = mask.repeat(1, 1, self.n_agents)
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        avail_actions = batch["avail_actions"]
        avail_agents = batch["avail_agents"]

        mac = self.mac
        alpha = max(self.alpha_end, self.alpha_start - t_env / self.alpha_decay_end) # linear decay

        mac.init_hidden_seq(bs)
        _, mac_out, chosen_actions_onehot = mac.optimize_actions(batch, dep_mode)
        agent_dep = chosen_actions_onehot

        mac_out[avail_actions == 0] = 1e-10
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 1e-10        

        pi = mac_out[:,:-1].clone()
        pi = pi.reshape(-1, self.n_actions)
        log_pi = th.log(pi)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals1 = self.critic1.forward(inputs, agent_dep, avail_agents, dep_mode=dep_mode)
        q_vals = q_vals1.detach()

        pi = mac_out[:,:-1].reshape(-1, self.n_actions)
        entropies = - (pi * log_pi).sum(dim=-1)

        pol_target = (pi * (alpha * log_pi - q_vals[:,:-1].reshape(-1, self.n_actions))).sum(dim=-1)

        policy_loss = (pol_target * mask).sum() / mask.sum()

        total_loss = policy_loss

        self.p_optimiser.zero_grad()
        total_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimiser.step()
        
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            if not dep_mode:
                self.logger.log_stat("policy_loss", policy_loss.item(), t_env)
                try:
                    self.logger.log_stat("agent_grad_norm", agent_grad_norm.item(), t_env)
                except:
                    self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
                self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
                self.logger.log_stat("alpha", alpha, t_env)
                self.logger.log_stat("ent", entropies.mean().item(), t_env)
                self.logger.log_stat('agent_utility', q_vals.mean().item(), t_env)
            else:
                self.logger.log_stat("dep_policy_loss", policy_loss.item(), t_env)
                try:
                    self.logger.log_stat("dep_agent_grad_norm", agent_grad_norm.item(), t_env)
                except:
                    self.logger.log_stat("dep_agent_grad_norm", agent_grad_norm, t_env)
                self.logger.log_stat("dep_pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
                self.logger.log_stat("dep_alpha", alpha, t_env)
                self.logger.log_stat("dep_ent", entropies.mean().item(), t_env)
                self.logger.log_stat('dep_agent_utility', q_vals.mean().item(), t_env)


    def train_critic(self, batch, t_env, dep_mode):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        avail_agents = batch['avail_agents']
        actions_onehot = batch["actions_onehot"][:, :-1]
        agent_id_onehot = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t-1, -1, -1)
        actions_onehot_dep = actions_onehot
        agent_dep = th.cat([agent_id_onehot, actions_onehot], dim=-1) #blav
        states = batch["state"]

        mac = self.mac
        mixer1 = self.mixer1
        target_mixer1 = self.target_mixer1
        
        alpha = max(self.alpha_end, self.alpha_start - t_env / self.alpha_decay_end) # linear decay

        mac.init_hidden_seq(bs)
        next_actions, mac_out, next_actions_onehot = mac.optimize_actions(batch, dep_mode)

        mac_out[avail_actions == 0] = 0.0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 1e-10

        t_mac_out = mac_out.clone().detach()
        pi = t_mac_out

        next_actions = next_actions.unsqueeze(3).detach()
        next_agent_dep = next_actions_onehot #blav
        next_actions_onehot = next_actions_onehot[:, :, :, self.n_agents:]

        pi_taken = th.gather(pi, dim=3, index=next_actions).squeeze(3)[:,1:]
        pi_taken[mask.expand_as(pi_taken) == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        target_inputs = self.target_critic1._build_inputs(batch, bs, max_t)
        target_q_vals1 = self.target_critic1.forward(target_inputs, next_agent_dep, avail_agents, dep_mode=dep_mode)

        next_vs1 = th.logsumexp(target_q_vals1 / alpha, dim=-1) * alpha         

        next_chosen_qvals1 = th.gather(target_q_vals1, dim=3, index=next_actions).squeeze(3)

        target_qvals1 = target_mixer1(next_chosen_qvals1, states, actions=next_actions_onehot, vs=next_vs1)

        target_qvals = target_qvals1

        #target_v = rewards + self.args.gamma * (1 - terminated) * target_qvals
        target_v = build_td_lambda_targets(rewards, terminated, mask, target_qvals, self.n_agents, self.args.gamma, self.args.td_lambda)
        targets = target_v - alpha * log_pi_taken.mean(dim=-1, keepdim=True)

        inputs = self.critic1._build_inputs(batch, bs, max_t)[:, :-1]
        q_vals1 = self.critic1.forward(inputs, agent_dep, avail_agents[:, :-1], dep_mode=dep_mode)

        vs1 = th.logsumexp(q_vals1 / alpha, dim=-1) * alpha

        q_taken1 = th.gather(q_vals1, dim=3, index=actions).squeeze(3)

        q_taken1 = mixer1(q_taken1, states[:, :-1], actions=actions_onehot, vs=vs1)

        td_error1 = q_taken1 - targets.detach()


        mask = mask.expand_as(td_error1)
        masked_td_error1 = td_error1 * mask
        loss1 = (masked_td_error1 ** 2).sum() / mask.sum()
        total_loss1 = loss1



        self.c_optimiser1.zero_grad()
        total_loss1.backward()
        grad_norm_1 = th.nn.utils.clip_grad_norm_(self.critic_params1, self.args.grad_norm_clip)
        self.c_optimiser1.step()
        
 

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            if not dep_mode:
                self.logger.log_stat("loss1", loss1.item(), t_env)
                try:
                    self.logger.log_stat("grad_norm_1", grad_norm_1.item(), t_env)
                except:
                    self.logger.log_stat("grad_norm_1", grad_norm_1, t_env)                    
                mask_elems = mask.sum().item()
                self.logger.log_stat("td_error_abs_1", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
                self.logger.log_stat("q_taken_mean_1",
                                     (q_taken1 * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                     t_env)
            else:
                self.log_stats_t = t_env
                self.logger.log_stat("dep_loss1", loss1.item(), t_env)
                try:
                    self.logger.log_stat("dep_grad_norm_1", grad_norm_1.item(), t_env)
                except:
                    self.logger.log_stat("dep_grad_norm_1", grad_norm_1, t_env)                   
                mask_elems = mask.sum().item()
                self.logger.log_stat("dep_td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
                self.logger.log_stat("dep_q_taken_mean_1",
                                     (q_taken1 * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("dep_target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                     t_env)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        self.train_actor(batch, t_env, episode_num, dep_mode=False)
        self.train_actor(batch, t_env, episode_num, dep_mode=True)
        self.train_critic(batch, t_env, dep_mode=False)
        self.train_critic(batch, t_env, dep_mode=True)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _update_targets(self):
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_mixer1.load_state_dict(self.mixer1.state_dict())
        self.target_mixer_dep1.load_state_dict(self.mixer_dep1.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic1.cuda()
        self.mixer1.cuda()
        self.mixer_dep1.cuda()
        self.target_critic1.cuda()
        self.target_mixer1.cuda()
        self.target_mixer_dep1.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        #th.save(self.critic1.state_dict(), "{}/critic1.th".format(path))
        #th.save(self.mixer1.state_dict(), "{}/mixer1.th".format(path))
        #th.save(self.mixer_dep1.state_dict(), "{}/mixer_dep1.th".format(path))
        #th.save(self.p_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        #th.save(self.c_optimiser1.state_dict(), "{}/critic_opt1.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic1.load_state_dict(th.load("{}/critic1.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())

        self.mixer1.load_state_dict(th.load("{}/mixer1.th".format(path), map_location=lambda storage, loc: storage))
        self.target_mixer1.load_state_dict(self.mixer1.state_dict())

        self.mixer_dep1.load_state_dict(th.load("{}/mixer_dep1.th".format(path), map_location=lambda storage, loc: storage))
        self.target_mixer_dep1.load_state_dict(self.mixer_dep1.state_dict())
        self.p_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser1.load_state_dict(th.load("{}/critic_opt1.th".format(path), map_location=lambda storage, loc: storage))

    def build_inputs(self, batch, bs, max_t, actions_onehot):
        inputs = []
        inputs.append(batch["obs"][:])
        actions = actions_onehot[:].reshape(bs, max_t, self.n_agents, -1)
        inputs.append(actions)
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
