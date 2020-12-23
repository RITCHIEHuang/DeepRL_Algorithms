#!/usr/bin/env python
# Created at 2020/3/3
import torch.nn as nn

from Algorithms.pytorch.Models.BaseQNet import BaseQNet


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class QNet_dqn(BaseQNet):
    def __init__(self, dim_state, dim_action, dim_hidden=64, activation=nn.LeakyReLU):
        super().__init__(dim_state, dim_action, dim_hidden)

        self.qvalue = nn.Sequential(nn.Linear(self.dim_state, self.dim_hidden),
                                    activation(),
                                    nn.Linear(self.dim_hidden, self.dim_hidden),
                                    activation(),
                                    nn.Linear(self.dim_hidden, self.dim_action))
        self.apply(init_weight)

    def forward(self, states, **kwargs):
        q_values = self.qvalue(states)
        return q_values

    def get_action(self, states):
        """
        >>>a = torch.rand(3, 4)
        tensor([[0.3643, 0.7805, 0.6098, 0.6551],
        [0.3953, 0.8059, 0.4277, 0.0126],
        [0.2667, 0.0109, 0.0467, 0.5328]])
        >>>a.max(dim=1)[1]
        tensor([1, 1, 3])
        :param states:
        :return: max_action (tensor)
        """
        q_values = self.forward(states, )
        max_action = q_values.max(dim=1)[1]  # action index with largest q values
        return max_action
