#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/3 下午6:48
import numpy as np
import tensorflow as tf
from Utils.tf2_util import NDOUBLE, TDOUBLE


def estimate_advantages(rewards, masks, values, gamma, tau, eps=1e-8):
    batch_size = rewards.shape[0]
    deltas = np.zeros((batch_size,1), dtype=NDOUBLE)
    advantages = np.zeros((batch_size,1), dtype=NDOUBLE)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(batch_size)):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i]
        prev_advantage = advantages[i]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

    return tf.convert_to_tensor(advantages, dtype=TDOUBLE), tf.convert_to_tensor(returns, dtype=TDOUBLE)
