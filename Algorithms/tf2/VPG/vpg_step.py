#!/usr/bin/env python
# Created at 2020/3/23

import tensorflow as tf


@tf.function
def vpg_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages):
    """update critic"""

    critic_loss_fn = tf.keras.losses.MeanSquaredError()
    value_loss = None
    for _ in range(optim_value_iternum):
        with tf.GradientTape() as tape:
            values_pred = value_net(states)
            value_loss = critic_loss_fn(returns, values_pred)

        grads = tape.gradient(value_loss, value_net.trainable_variables)
        optimizer_value.apply_gradients(
            grads_and_vars=zip(grads, value_net.trainable_variables))

    """update policy"""
    with tf.GradientTape() as tape:
        log_probs = tf.expand_dims(
            policy_net.get_log_prob(states, actions), axis=-1)
        policy_loss = - tf.reduce_mean(log_probs * advantages)

    grads = tape.gradient(policy_loss, policy_net.trainable_variables)
    optimizer_policy.apply_gradients(
        grads_and_vars=zip(grads, policy_net.trainable_variables))

    return {"critic_loss": value_loss,
            "policy_loss": policy_loss
            }
