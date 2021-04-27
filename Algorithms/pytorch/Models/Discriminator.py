#!/usr/bin/env python
# Created at 2020/5/9

import torch
import torch.nn as nn


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Discriminator(nn.Module):
    def __init__(
        self, dim_state, dim_action, dim_hidden=128, activation=nn.LeakyReLU
    ):
        super(Discriminator, self).__init__()

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden

        self.model = nn.Sequential(
            nn.Linear(self.dim_state + self.dim_action, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, 1),
        )

        self.model.apply(init_weight)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        logits = self.model(state_action)
        prob = torch.sigmoid(logits)
        return prob, logits
