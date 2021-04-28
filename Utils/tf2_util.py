#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/19 下午3:32

import tensorflow as tf
import numpy as np

TLONG = tf.int64
TFLOAT = tf.float32
TDOUBLE = tf.float64

NLONG = np.int64
NFLOAT = np.float64
NDOUBLE = np.float64


@tf.function
def get_flat(nested_tensor):
    """get flattened tensor"""
    flattened = tf.concat(
        [tf.reshape(t, [-1]) for t in nested_tensor],
        axis=0,
    )
    return flattened


def set_from_flat(model, flat_weights):
    """set model weights from flattened grads"""
    weights = []
    idx = 0
    for var in model.trainable_variables:
        n_vars = np.prod(var.shape)
        weights.append(tf.reshape(flat_weights[idx : idx + n_vars], var.shape))
        idx += n_vars
    model.set_weights(weights)


def flatgrad(grads, var_list, clip_norm=None):
    """
    calculates the gradient and flattens it
    :param loss: (float) the loss value
    :param var_list: ([TensorFlow Tensor]) the variables
    :param clip_norm: (float) clip the gradients (disabled if None)
    :return: ([TensorFlow Tensor]) flattened gradient
    """
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(
        [
            tf.reshape(
                grad if grad is not None else tf.zeros_like(v, dtype=v.dtype),
                [-1],
            )
            for (v, grad) in zip(var_list, grads)
        ],
        axis=0,
    )