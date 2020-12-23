#!/usr/bin/env python
# Created at 2020/3/24
import torch
import torch.nn as nn

from Algorithms.pytorch.Models.BaseQNet import BaseQNet


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class QValue(BaseQNet):
    def __init__(self, dim_state, dim_action, dim_hidden=128, activation=nn.LeakyReLU):
        super(QValue, self).__init__(dim_state, dim_action, dim_hidden)
        self.qvalue = nn.Sequential(nn.Linear(self.dim_state + self.dim_action, self.dim_hidden),
                                    activation(),
                                    nn.Linear(self.dim_hidden, self.dim_hidden),
                                    activation(),
                                    nn.Linear(self.dim_hidden, 1))
        self.apply(init_weight)

    def forward(self, states, actions):
        states_actions = torch.cat([states, actions], dim=-1)
        q_value = self.qvalue(states_actions)

        return q_value
