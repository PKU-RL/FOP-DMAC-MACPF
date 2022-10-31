import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

dep_embed_dim = 64
class DFOPCritic(nn.Module):
    def __init__(self, scheme, args):
        super(DFOPCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRU(args.rnn_hidden_dim, args.rnn_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, self.n_actions)
        self.dep_fc1 = nn.Linear(args.n_actions + args.n_agents, dep_embed_dim)
        self.dep_fc2 = nn.Linear(dep_embed_dim + args.rnn_hidden_dim, args.n_actions)

    def forward(self, inputs, joint_actions, avail_agents, dep_mode):
        b,l,a,v = inputs.shape
        avail_agents = avail_agents.float()

        #obs ---> traj
        x = F.relu(self.fc1(inputs)) #blav
        x = x.permute(0, 2, 1, 3).reshape(b * a, l, -1)
        h_in = th.zeros(size=(1, b * a, self.args.rnn_hidden_dim), dtype=th.float, device=inputs.device)
        h, _ = self.rnn(x, h_in)
        h = h.reshape(b, a, l, -1).permute(0, 2, 1, 3)

        #p_actions --> dep
        dep_in = F.elu(self.dep_fc1(joint_actions)) #blav
        #dep_in = self.dep_fc1(joint_actions)
        dep_h = th.matmul(avail_agents, dep_in)
        dep_h = dep_h / self.n_agents
        dep_h[:, :, 0] = 0.0
        dep_final = th.cat([h, dep_h], dim=-1)

        #obs ---> pure q
        q = self.fc2(h)

        #(traj, dep) --> q
        if dep_mode:
            dep = self.dep_fc2(dep_final)
            q = q.detach() + dep
        return q

    def _build_inputs(self, batch, bs, max_t):
        inputs = []
        # state, obs, action
        #inputs.append(batch["state"][:].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        inputs.append(batch["obs"][:])
        # last actions
        #if self.args.obs_last_action:
        last_action = []
        last_action.append(th.zeros_like(batch["actions_onehot"][:, 0]).unsqueeze(1))
        last_action.append(batch["actions_onehot"][:, :max_t-1])
        last_action = th.cat([x for x in last_action], dim = 1)
        inputs.append(last_action)
        #agent id
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # state
        #input_shape = scheme["state"]["vshape"]
        # observation
        input_shape = scheme["obs"]["vshape"]
        # actions and last actions
        #if self.args.obs_last_action:
        input_shape += scheme["actions_onehot"]["vshape"][0]
        # agent id
        input_shape += self.n_agents
        return input_shape
