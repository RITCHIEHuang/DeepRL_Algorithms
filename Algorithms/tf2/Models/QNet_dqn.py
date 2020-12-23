#!/usr/bin/env python
# Created at 2020/3/21
from Algorithms.tf2.Models.BaseQNet import BaseQNet
import tensorflow as tf
import tensorflow.keras.layers as layers

class QNet_dqn(BaseQNet):
    def __init__(self, dim_state, dim_action, dim_hidden=64, activation=tf.nn.leaky_relu):
        super().__init__(dim_state, dim_action, dim_hidden)

        self.model = tf.keras.models.Sequential([
            layers.Dense(dim_hidden, activation=activation, bias_initializer=tf.zeros_initializer),
            layers.Dense(dim_hidden, activation=activation, bias_initializer=tf.zeros_initializer),
            layers.Dense(dim_action, activation=None)
        ])

        self.model.build(input_shape=(None, dim_state))

    def call(self, states, **kwargs):
        q_values = self.model(states)
        return q_values

    def get_action(self, states):
        q_values = self.predict_on_batch(states)
        action = tf.argmax(q_values, axis=-1)
        return action
