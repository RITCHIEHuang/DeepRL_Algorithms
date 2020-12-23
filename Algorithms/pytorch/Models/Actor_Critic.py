#!/usr/bin/env python
# Created at 2020/3/31

import torch.nn as nn

from Algorithms.pytorch.Models.BasePolicy import BasePolicy


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Actor_Critic(nn.Module):

    def __init__(self, actor: BasePolicy, critic):
        super(Actor_Critic, self).__init__()

        self.actor = actor
        self.critic = critic

        self.apply(init_weight)

    def forward(self, inputs):
        pass

    def get_log_prob(self, state, action):
        return self.actor.get_log_prob(state, action)

    def get_action_log_prob(self, states):
        action, log_prob = self.actor.get_action_log_prob(states)
        return action, log_prob

    def get_value(self, states):
        return self.critic(states)

    def get_entropy(self, states):
        return self.actor.get_entropy(states)
