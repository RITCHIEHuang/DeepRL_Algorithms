#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/3/23

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow_probability.python.distributions import Categorical

from Algorithms.tf2.Models.BasePolicy import BasePolicy


class DiscretePolicy(BasePolicy):

    def __init__(self, dim_state, dim_action, dim_hidden=128, activation=tf.nn.leaky_relu):
        super(DiscretePolicy, self).__init__(dim_state, dim_action, dim_hidden)

        self.policy = tf.keras.models.Sequential([
            layers.Dense(self.dim_hidden, activation=activation),
            layers.Dense(self.dim_hidden, activation=activation),
            layers.Dense(self.dim_action, activation=tf.nn.softmax)
        ])

        self.policy.build(input_shape=(None, self.dim_state))


    def _get_dist(self, states):
        action_probs = self.policy(states)
        dist = Categorical(probs=action_probs)
        return dist

    def call(self, states, **kwargs):
        dist = self._get_dist(states)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_log_prob(self, states, actions):
        dist = self._get_dist(states)
        log_prob = dist.log_prob(actions)
        return log_prob

    def get_action_log_prob(self, states):
        return self.call(states)

    def get_entropy(self, states):
        dist = self._get_dist(states)
        return dist.entropy()

    def get_kl(self, x):
        pass
