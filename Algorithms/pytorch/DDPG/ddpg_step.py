#!/usr/bin/env python
# Created at 2020/1/22
import torch
import torch.nn as nn

from Utils.torch_util import get_flat_params, set_flat_params


def ddpg_step(policy_net, policy_net_target, value_net, value_net_target, optimizer_policy, optimizer_value,
              states, actions, rewards, next_states, masks, gamma, polyak):
    masks = masks.unsqueeze(-1)
    rewards = rewards.unsqueeze(-1)
    """update critic"""

    values = value_net(states, actions)

    with torch.no_grad():
        target_next_values = value_net_target(
            next_states, policy_net_target(next_states))
        target_values = rewards + gamma * masks * target_next_values
    value_loss = nn.MSELoss()(values, target_values)

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update actor"""

    policy_loss = - value_net(states, policy_net(states)).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    """soft update target nets"""
    policy_net_flat_params = get_flat_params(policy_net)
    policy_net_target_flat_params = get_flat_params(policy_net_target)
    set_flat_params(policy_net_target, polyak *
                    policy_net_target_flat_params + (1 - polyak) * policy_net_flat_params)

    value_net_flat_params = get_flat_params(value_net)
    value_net_target_flat_params = get_flat_params(value_net_target)
    set_flat_params(value_net_target, polyak *
                    value_net_target_flat_params + (1 - polyak) * value_net_flat_params)

    return {"critic_loss": value_loss,
            "policy_loss": policy_loss
            }
