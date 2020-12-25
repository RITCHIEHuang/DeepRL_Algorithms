#!/usr/bin/env python
# Created at 2020/3/23

import tensorflow as tf
import numpy as np
from Utils.tf2_util import NDOUBLE


def reinforce_step(policy_net, optimizer_policy, states, actions, rewards, masks, gamma, eps=1e-6):
    """calculate cumulative reward"""
    cum_rewards = np.zeros_like(rewards, dtype=NDOUBLE)
    pre_value = 0
    for i in reversed(range(rewards.shape[0])):
        pre_value = gamma * masks[i] * pre_value + rewards[i]
        cum_rewards[i] = pre_value

    # normalize cumulative rewards
    cum_rewards = (cum_rewards - cum_rewards.mean()) / \
        (cum_rewards.std() + eps)

    """update policy"""

    with tf.GradientTape() as tape:
        log_probs = policy_net.get_log_prob(states, actions)
        policy_loss = tf.reduce_mean(-log_probs * cum_rewards)
    grads = tape.gradient(policy_loss, policy_net.trainable_variables)
    optimizer_policy.apply_gradients(
        grads_and_vars=zip(grads, policy_net.trainable_variables))

    return {"policy_loss": policy_loss}
