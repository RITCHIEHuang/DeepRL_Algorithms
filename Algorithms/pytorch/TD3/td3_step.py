#!/usr/bin/env python
# Created at 2020/3/1
import torch
import torch.nn as nn

from Utils.torch_util import get_flat_params, set_flat_params


def td3_step(policy_net, policy_net_target, value_net_1, value_net_target_1, value_net_2, value_net_target_2,
             optimizer_policy, optimizer_value_1, optimizer_value_2, states, actions, rewards, next_states, masks,
             gamma, polyak, target_action_noise_std, target_action_noise_clip, action_high, update_policy=False):
    rewards = rewards.unsqueeze(-1)
    masks = masks.unsqueeze(-1)

    """update critic"""
    with torch.no_grad():
        target_action = policy_net_target(next_states)
        target_action_noise = torch.clamp(torch.randn_like(target_action) * target_action_noise_std,
                                          -target_action_noise_clip, target_action_noise_clip)
        target_action = torch.clamp(
            target_action + target_action_noise, -action_high, action_high)
        target_values = rewards + gamma * masks * torch.min(value_net_target_1(next_states, target_action),
                                                            value_net_target_2(next_states, target_action))

    """update value1 target"""
    values_1 = value_net_1(states, actions)
    value_loss_1 = nn.MSELoss()(target_values, values_1)

    optimizer_value_1.zero_grad()
    value_loss_1.backward()
    optimizer_value_1.step()

    """update value2 target"""
    values_2 = value_net_2(states, actions)
    value_loss_2 = nn.MSELoss()(target_values, values_2)

    optimizer_value_2.zero_grad()
    value_loss_2.backward()
    optimizer_value_2.step()

    policy_loss = None
    if update_policy:
        """update policy"""
        policy_loss = - value_net_1(states, policy_net(states)).mean()

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        """soft update target nets"""
        policy_net_flat_params = get_flat_params(policy_net)
        policy_net_target_flat_params = get_flat_params(policy_net_target)
        set_flat_params(policy_net_target,
                        polyak * policy_net_target_flat_params + (1 - polyak) * policy_net_flat_params)

        value_net_1_flat_params = get_flat_params(value_net_1)
        value_net_1_target_flat_params = get_flat_params(value_net_target_1)
        set_flat_params(value_net_target_1,
                        polyak * value_net_1_target_flat_params + (1 - polyak) * value_net_1_flat_params)

        value_net_2_flat_params = get_flat_params(value_net_2)
        value_net_2_target_flat_params = get_flat_params(value_net_target_2)
        set_flat_params(value_net_target_2,
                        polyak * value_net_2_target_flat_params + (1 - polyak) * value_net_2_flat_params)

    return {"q_value_loss_1": value_loss_1,
            "q_value_loss_2": value_loss_2,
            "policy_loss": policy_loss
            }
