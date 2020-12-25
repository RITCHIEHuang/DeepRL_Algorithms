#!/usr/bin/env python
# Created at 2020/1/22
import torch
import torch.nn as nn


def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, old_log_probs, clip_epsilon, l2_reg, ent_coeff=0):
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
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon,
                        1.0 + clip_epsilon) * advantages
    policy_surr = -torch.min(surr1, surr2).mean()
    ent = policy_net.get_entropy(states)
    entbonous = -ent_coeff * ent.mean()
    optim_gain = policy_surr + entbonous

    optimizer_policy.zero_grad()
    optim_gain.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()

    return {"critic_loss": value_loss,
            "policy_loss": policy_surr,
            "policy_entropy": ent.mean()
            }
