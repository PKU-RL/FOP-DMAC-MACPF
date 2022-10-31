from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np

# This multi-agent controller shares parameters between agents
class DFOPMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY["multinomial_seq"](args)

        self.hidden_states_step = []
        self.hidden_states_seq = []


    def optimize_actions(self, ep_batch, dep_mode, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:] #blav
        b,l,a,v = avail_actions.shape
        device = avail_actions.device
        chosen_action_list = th.zeros(size=(b, l, self.n_agents), dtype=th.long, device=device)
        chosen_action_onehot_list = []
        agent_output_list = th.zeros(size=(b, l, self.n_agents, self.args.n_actions), dtype=th.float, device=device)

        self.agent.clear_dep()
        for agent_id in range(self.n_agents):
            agent_output = self.forward_seq(ep_batch, agent_id, chosen_action_onehot_list, dep_mode, test_mode=test_mode)
            agent_output_list[:, :, agent_id] = agent_output

            chosen_action = self.action_selector.optimize_action_seq(agent_output[bs], avail_actions[bs, :, agent_id])
            chosen_action_list[:, :, agent_id] = chosen_action
            
            chosen_action_onehot = th.zeros(size=(b, l, self.args.n_actions), dtype=th.float, device=device)
            chosen_action_onehot.scatter_(-1, chosen_action.unsqueeze(-1), 1.0)
            agent_id_onehot = th.zeros(size=(b, l, self.args.n_agents), dtype=th.float, device=device)
            agent_id_onehot[:, :, agent_id] = 1.0
            chosen_action_onehot_list.append(th.cat([agent_id_onehot, chosen_action_onehot], dim=-1))

        return chosen_action_list, agent_output_list, th.stack(chosen_action_onehot_list, dim=2)

    def select_actions(self, ep_batch, t_ep, t_env, dep_mode, bs=slice(None) , test_mode=False):
        with th.no_grad():
            self.agent.clear_dep()
            avail_actions = ep_batch["avail_actions"][:, t_ep]
            b,a,v = avail_actions.shape
            device = avail_actions.device
            chosen_actions = th.zeros(size=(b, self.n_agents), dtype=th.long, device=device)
            chosen_actions_onehot = []
            for agent_id in range(self.n_agents):
                agent_out = self.forward_step(ep_batch, agent_id, chosen_actions_onehot, t_ep, dep_mode, test_mode=test_mode)
                agent_action = self.action_selector.select_action_step(agent_out[bs], avail_actions[bs, agent_id], t_env, test_mode=test_mode)
                
                chosen_actions[:, agent_id] = agent_action
                
                agent_action_onehot = th.zeros(size=(b, self.args.n_actions), dtype=th.float, device=device)
                agent_action_onehot.scatter_(-1, agent_action.unsqueeze(-1), 1.0)
                agent_id_onehot = th.zeros(size=(b, self.args.n_agents), dtype=th.float, device=device)
                agent_id_onehot[:, agent_id] = 1.0
                chosen_actions_onehot.append(th.cat([agent_id_onehot, agent_action_onehot], dim=-1))
            return chosen_actions


    def forward_seq(self, ep_batch, agent_id, parents_actions, dep_mode, test_mode=False):
        agent_inputs = self._build_inputs_seq(ep_batch, agent_id)
        avail_actions = ep_batch["avail_actions"][:]
        avail_agents = ep_batch["avail_agents"][:]
        pi_logit = self.agent.forward_seq(agent_inputs, 
        								  self.hidden_states_seq[agent_id], 
        								  avail_agents[:, :, agent_id], 
        								  parents_actions, 
        								  agent_id,
                                          dep_mode)
        agent_outs = pi_logit
        agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        if not test_mode:
            epsilon_action_num = agent_outs.size(-1)
            agent_outs = ((1 - self.action_selector.epsilon) * agent_outs + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

        return agent_outs.view(ep_batch.batch_size, ep_batch.max_seq_length, -1)

    def forward_step(self, ep_batch, agent_id, parents_actions, t, dep_mode, test_mode=False):
        agent_inputs = self._build_inputs_step(ep_batch, t, agent_id)
        avail_actions = ep_batch["avail_actions"][:, t]
        avail_agents = ep_batch["avail_agents"][:, t]
        pi_logit, hidden_state = self.agent.forward_step(agent_inputs, 
        												 self.hidden_states_step[agent_id], 
        												 avail_agents[:, agent_id], 
        												 parents_actions, 
        												 agent_id,
                                                         dep_mode)
        self.hidden_states_step[agent_id] = hidden_state
        agent_outs = pi_logit
        agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        if not test_mode:
            epsilon_action_num = agent_outs.size(-1)
            agent_outs = ((1 - self.action_selector.epsilon) * agent_outs + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

        return agent_outs.view(ep_batch.batch_size, self.args.n_actions)


    def init_hidden_step(self, batch_size):
        self.hidden_states_step = []
        for _ in range(self.n_agents):
            self.hidden_states_step.append(self.agent.init_hidden().unsqueeze(0).repeat(1, batch_size, 1))  # 1bv

    def init_hidden_seq(self, batch_size):
        self.hidden_states_seq = []
        for _ in range(self.n_agents):
            self.hidden_states_seq.append(self.agent.init_hidden().unsqueeze(0).repeat(1, batch_size, 1))  # 1bv

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY["dfop"](input_shape, self.args)

    def _build_inputs_step(self, batch, t, agent_id):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t, agent_id])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t, agent_id]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1, agent_id])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)[:, agent_id])
        
        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        return inputs

    def _build_inputs_seq(self, batch, agent_id):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        seq_l = batch.max_seq_length
        inputs = []
        inputs.append(batch["obs"][:, :, agent_id])  # b1av
        if self.args.obs_last_action:
            obs_last_action = th.cat([th.zeros_like(batch["actions_onehot"][:, 0, agent_id].unsqueeze(1)), batch["actions_onehot"][:, :seq_l-1, agent_id]], dim=1)
            inputs.append(obs_last_action)
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, seq_l, -1, -1)[:, :, agent_id])

        inputs = th.cat([x.reshape(bs, seq_l, -1) for x in inputs], dim=2)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape