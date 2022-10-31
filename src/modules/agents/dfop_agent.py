import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
dep_embed_dim = 64

class DFOPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DFOPAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRU(args.rnn_hidden_dim, args.rnn_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.dep_fc1 = nn.Linear(args.n_actions + args.n_agents, dep_embed_dim)
        self.dep_fc2 = nn.Linear(dep_embed_dim + args.rnn_hidden_dim, args.n_actions)

        self.dep_list = []

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def clear_dep(self):
        self.dep_list = []

    def forward_step(self, inputs, hidden_state, avail_agents, parents_actions, agent_id, dep_mode):
        b,v = inputs.shape
        avail_mask = avail_agents[:, :agent_id].unsqueeze(-1)

        #obs ---> traj
        x = F.relu(self.fc1(inputs))
        x = x.unsqueeze(1)
        h_in = hidden_state.reshape(1, b, self.args.rnn_hidden_dim)
        _, h = self.rnn(x, h_in)

        #p_actions --> dep
        if agent_id != 0:
            self.dep_list.append(parents_actions[-1])
            dep_in = F.elu(self.dep_fc1(torch.stack(self.dep_list, dim=1))) #bav
            #dep_in = self.dep_fc1(torch.stack(self.dep_list, dim=1))
            dep_h = (dep_in * avail_mask).sum(dim=1) / self.n_agents
        else:
            dep_h = torch.zeros(size=(b,dep_embed_dim), dtype=torch.float, device=h.device)

        #obs ---> pure logit
        logit = self.fc2(h)

        #(traj, dep) --> logit
        if dep_mode:
            dep_final = torch.cat([h, dep_h.unsqueeze(1)], dim=-1)
            dep_logit = self.dep_fc2(dep_final)
            logit = logit.detach() + dep_logit

        return logit, h

    def forward_seq(self, inputs, hidden_state, avail_agents, parents_actions, agent_id, dep_mode):
        b,l,v = inputs.shape
        avail_mask = avail_agents[:, :, :agent_id].unsqueeze(-1)

        #obs ---> traj
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(1, b, self.args.rnn_hidden_dim)
        h, _ = self.rnn(x, h_in)

        #p_actions --> dep
        if agent_id != 0:
            self.dep_list.append(parents_actions[-1])
            dep_in = F.elu(self.dep_fc1(torch.stack(self.dep_list, dim=2))) #blav
            #dep_in = self.dep_fc1(torch.stack(self.dep_list, dim=2))
            dep_h = (dep_in * avail_mask).sum(dim=2) / self.n_agents
        else:
            dep_h = torch.zeros(size=(b,l,dep_embed_dim), dtype=torch.float, device=h.device)


        #obs ---> pure logit
        logit = self.fc2(h)


        #(traj, dep) --> logit
        if dep_mode:
            dep_final = torch.cat([h, dep_h], dim=-1)
            dep_logit = self.dep_fc2(dep_final)
            logit = logit.detach() + dep_logit


        return logit
