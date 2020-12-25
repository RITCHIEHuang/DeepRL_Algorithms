#!/usr/bin/env python
# Created at 2020/3/31

import torch.nn as nn


def a2c_step(ac_net, optimizer_ac, states, actions, returns, advantages, value_loss_coeff, entropy_coeff):
    """update actor_critic net"""
    log_probs = ac_net.get_log_prob(states, actions)
    policy_loss = -(log_probs * advantages).mean()

    value = ac_net.get_value(states)
    value_loss = nn.MSELoss()(value, returns)

    entropy = ac_net.get_entropy(states)

    ac_loss = policy_loss + value_loss_coeff * value_loss - entropy_coeff * entropy

    optimizer_ac.zero_grad()
    ac_loss.backward()
    nn.utils.clip_grad_norm_(ac_net.parameters(), 20)
    optimizer_ac.step()

    return {"actor_critic_loss": ac_loss}
