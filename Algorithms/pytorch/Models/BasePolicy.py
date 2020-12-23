#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午9:52
import torch.nn as nn


class BasePolicy(nn.Module):
    def __init__(self, dim_state, dim_action, dim_hidden=128):
        super(BasePolicy, self).__init__()

        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_action = dim_action

    def forward(self, x):
        raise NotImplementedError()

    def get_action_log_prob(self, states):
        raise NotImplementedError()

    def get_log_prob(self, state, action):
        raise NotImplementedError()

    def get_entropy(self, states):
        raise NotImplementedError()
