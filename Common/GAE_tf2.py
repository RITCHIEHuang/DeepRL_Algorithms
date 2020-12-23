#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/3 下午6:48
import numpy as np
from Utils.tf2_util import NDOUBLE


def estimate_advantages(rewards, masks, values, gamma, tau):
    deltas = np.zeros((rewards.shape[0], 1), dtype=NDOUBLE)
    advantages = np.zeros((rewards.shape[0], 1), dtype=NDOUBLE)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.shape[0])):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i]
        prev_advantage = advantages[i]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    return advantages, returns
