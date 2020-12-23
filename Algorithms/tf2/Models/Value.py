#!/usr/bin/env python
# Created at 2020/3/23

import tensorflow as tf
import tensorflow.keras.layers as layers

class Value(tf.keras.Model):
    def __init__(self, dim_state, dim_hidden=128, activation=tf.nn.leaky_relu, l2_reg=1e-3):
        super(Value, self).__init__()

        self.dim_state = dim_state
        self.dim_hidden = dim_hidden

        self.value = tf.keras.models.Sequential([
            layers.Dense(self.dim_hidden, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)),
            layers.Dense(self.dim_hidden, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)),
            layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg))
        ])

        self.value.build(input_shape=(None, self.dim_state))


    def call(self, states, **kwargs):
        value = self.value(states)
        return value