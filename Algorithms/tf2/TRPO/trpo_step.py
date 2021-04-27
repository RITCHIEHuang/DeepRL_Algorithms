#!/usr/bin/env python
# Created at 2020/1/22
import tensorflow as tf


@tf.function
def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, old_log_probs, clip_epsilon, entropy_coeff=1e-3):
    """update critic"""
    critic_loss_fn = tf.keras.losses.MeanSquaredError()
    for _ in range(optim_value_iternum):
        with tf.GradientTape() as tape:
            values_pred = value_net(states)
            value_loss = critic_loss_fn(returns, y_pred=values_pred)

        grads = tape.gradient(value_loss, value_net.trainable_variables)
        optimizer_value.apply_gradients(
            grads_and_vars=zip(grads, value_net.trainable_variables))

    """update policy"""
    with tf.GradientTape() as tape:
        log_probs = tf.expand_dims(policy_net.get_log_prob(states, actions), axis=-1)
        ratio = tf.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(
            ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        entropy = tf.reduce_mean(policy_net.get_entropy(states))
        policy_loss = - tf.reduce_mean(tf.minimum(surr1, surr2)) - entropy_coeff * entropy

    grads = tape.gradient(policy_loss, policy_net.trainable_variables)
    # grads, grad_norm = tf.clip_by_global_norm(grads, 40)
    optimizer_policy.apply_gradients(
        grads_and_vars=zip(grads, policy_net.trainable_variables))

    return {"ratio": ratio,
            "critic_loss": value_loss,
            "policy_loss": policy_loss,
            "policy_entropy": entropy
            }
