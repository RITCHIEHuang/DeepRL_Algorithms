#!/usr/bin/env python
# Created at 2020/3/22

import tensorflow as tf
import tensorflow.keras.layers as layers

from Algorithms.tf2.Models.BaseQNet import BaseQNet


class QNet_duelingdqn(BaseQNet):
    def __init__(self, dim_state, dim_action, dim_hidden=64, activation=tf.nn.leaky_relu):
        super().__init__(dim_state, dim_action, dim_hidden)

        self.advantage = tf.keras.models.Sequential([
            layers.Dense(self.dim_hidden, activation=activation, bias_initializer=tf.zeros_initializer),
            layers.Dense(self.dim_hidden, activation=activation, bias_initializer=tf.zeros_initializer),
            layers.Dense(self.dim_action, activation=None)
        ])

        self.advantage.build(input_shape=(None, self.dim_state))

        self.value = tf.keras.models.Sequential([
            layers.Dense(self.dim_hidden, activation=activation, bias_initializer=tf.zeros_initializer),
            layers.Dense(self.dim_hidden, activation=activation, bias_initializer=tf.zeros_initializer),
            layers.Dense(1, activation=None)
        ])

        self.value.build(input_shape=(None, self.dim_state))

    def call(self, states, **kwargs):
        advantage = self.advantage(states)
        value = self.value(states)

        q_values = value + advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True)
        return q_values

    def get_action(self, states):
        q_values = self.predict_on_batch(states)
        max_action = tf.argmax(q_values, axis=-1)  # action index with largest q values
        return max_action
