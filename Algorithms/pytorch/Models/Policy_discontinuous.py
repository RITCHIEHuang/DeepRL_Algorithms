#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午9:52
import torch
import torch.nn as nn
from torch.distributions import Categorical

from Algorithms.pytorch.Models.BasePolicy import BasePolicy


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class DiscretePolicy(BasePolicy):
    def __init__(
        self, dim_state, dim_action, dim_hidden=128, activation=nn.LeakyReLU
    ):
        super(DiscretePolicy, self).__init__(dim_state, dim_action, dim_hidden)

        self.policy = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_action),
            nn.Softmax(dim=-1),
        )
        self.apply(init_weight)

    def forward(self, x):
        action_probs = self.policy(x)
        dist = Categorical(action_probs)
        return dist

    def get_log_prob(self, state, action):
        dist = self.forward(state)
        log_prob = dist.log_prob(action)
        return log_prob

    def get_action_log_prob(self, states):
        dist = self.forward(states)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_entropy(self, states):
        dist = self.forward(states)
        return dist.entropy().mean()

    def get_kl(self, x):
        action_probs = self.policy(x)
        action_probs_old = action_probs.detach()
        kl = action_probs_old * (
            torch.log(action_probs_old) - torch.log(action_probs)
        )
        return kl.sum(dim=1, keepdim=True)
