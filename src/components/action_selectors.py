import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
import torch.nn.functional as F

REGISTRY = {}

'''
class GumbelSoftmax():
    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.eps = 1e-10
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False): 
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            U = th.rand(masked_policies.size()).cuda()
            y = masked_policies - th.log(-th.log(U + self.eps) + self.eps)
            y = F.softmax(y / 1, dim=-1)
            y[avail_actions == 0.0] = 0.0
            picked_actions = y.max(dim=2)[1]

        return picked_actions

REGISTRY["gumbel"] = GumbelSoftmax
'''

class MultinomialSeqActionSelector():
    
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action_step(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0
        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=1)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()
            random_numbers = th.rand_like(agent_inputs[:, 0])
            pick_random = (random_numbers < self.epsilon).long()
            random_actions = Categorical(avail_actions.float()).sample().long()
            picked_actions = pick_random * random_actions + (1 - pick_random) * picked_actions

        if not (th.gather(avail_actions, dim=1, index=picked_actions.unsqueeze(1)) > 0.99).all():
            return self.select_action(agent_inputs, avail_actions, t_env, test_mode)

        return picked_actions

    def optimize_action_seq(self, agent_inputs, avail_actions):
        masked_policies = agent_inputs.clone()
        b,l,v = avail_actions.shape
        device = avail_actions.device
        masked_policies[avail_actions == 0.0] = 0.0
        masked_policies[[avail_actions.sum(dim=-1,) == 0]] += 1e-10
        picked_actions = Categorical(masked_policies).sample().long()
        return picked_actions

        
REGISTRY["multinomial_seq"] = MultinomialSeqActionSelector

class MultinomialActionSelector():
    
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)
        if getattr(self.args, "dmac_policy", True):
            if test_mode and self.test_greedy:
                picked_actions = masked_policies.max(dim=2)[1]
            else:
                picked_actions = Categorical(masked_policies).sample().long()
        else:
            if test_mode and self.test_greedy:
                picked_actions = masked_policies.max(dim=2)[1]
            else:
                picked_actions = Categorical(masked_policies).sample().long()

                random_numbers = th.rand_like(agent_inputs[:, :, 0])
                pick_random = (random_numbers < self.epsilon).long()
                random_actions = Categorical(avail_actions.float()).sample().long()
                picked_actions = pick_random * random_actions + (1 - pick_random) * picked_actions

            if not (th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99).all():
                return self.select_action(agent_inputs, avail_actions, t_env, test_mode)

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector