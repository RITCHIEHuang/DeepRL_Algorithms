#!/usr/bin/env python
# Created at 2020/1/22
import torch
import torch.nn as nn


def vpg_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, l2_reg):
    """update critic"""
    value_loss = None
    for _ in range(optim_value_iternum):
        values_pred = value_net(states)
        value_loss = nn.MSELoss()(values_pred, returns)
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    """update policy"""
    log_probs = policy_net.get_log_prob(states, actions)
    policy_loss = -(log_probs * advantages).mean()

    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()

    return {"critic_loss": value_loss,
            "policy_loss": policy_loss
            }
