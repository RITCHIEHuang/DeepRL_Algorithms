#!/usr/bin/env python
# Created at 2020/3/21
import tensorflow as tf

class BaseQNet(tf.keras.Model):
    def __init__(self, dim_state, dim_action, dim_hidden):
        super(BaseQNet, self).__init__()
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_action = dim_action

    def call(self, states, **kwargs):
        raise NotImplementedError()

    def get_action(self, states):
        raise NotImplementedError()


