#!/usr/bin/env python
# Created at 2020/2/21
import torch.nn as nn

from Algorithms.pytorch.Models.BasePolicy import BasePolicy


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Policy(BasePolicy):
    def __init__(
        self,
        dim_state,
        dim_action,
        max_action=None,
        dim_hidden=128,
        activation=nn.LeakyReLU,
    ):
        super(Policy, self).__init__(dim_state, dim_action, dim_hidden)
        self.max_action = max_action
        self.action = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_action),
            nn.Tanh(),
        )
        self.apply(init_weight)

    def forward(self, x):
        action = self.action(x)
        return action * self.max_action

    def get_action_log_prob(self, states):
        action = self.forward(states)
        return action, None
