#!/usr/bin/env python
# Created at 2020/3/3
import torch.nn as nn


class BaseQNet(nn.Module):
    def __init__(self, dim_state, dim_action, dim_hidden=64):
        super(BaseQNet, self).__init__()

        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_action = dim_action

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def get_action(self, states):
        raise NotImplementedError()
