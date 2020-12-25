#!/usr/bin/env python
# Created at 2020/3/25
import torch
import torch.nn as nn

from Utils.torch_util import get_flat_params, set_flat_params


def sac_step(policy_net, value_net, value_net_target, q_net_1, q_net_2, optimizer_policy, optimizer_value,
             optimizer_q_net_1, optimizer_q_net_2, states, actions, rewards, next_states, masks, gamma, polyak,
             update_target=False):
    rewards = rewards.unsqueeze(-1)
    masks = masks.unsqueeze(-1)

    """update qvalue net"""

    q_value_1 = q_net_1(states, actions)
    q_value_2 = q_net_2(states, actions)
    with torch.no_grad():
        target_next_value = rewards + gamma * \
            masks * value_net_target(next_states)

    q_value_loss_1 = nn.MSELoss()(q_value_1, target_next_value)
    optimizer_q_net_1.zero_grad()
    q_value_loss_1.backward()
    optimizer_q_net_1.step()

    q_value_loss_2 = nn.MSELoss()(q_value_2, target_next_value)
    optimizer_q_net_2.zero_grad()
    q_value_loss_2.backward()
    optimizer_q_net_2.step()

    """update policy net"""
    new_actions, log_probs = policy_net.rsample(states)
    min_q = torch.min(
        q_net_1(states, new_actions),
        q_net_2(states, new_actions)
    )
    policy_loss = (log_probs - min_q).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    """update value net"""
    target_value = (min_q - log_probs).detach()
    value_loss = nn.MSELoss()(value_net(states), target_value)
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    if update_target:
        """ update target value net """
        value_net_target_flat_params = get_flat_params(value_net_target)
        value_net_flat_params = get_flat_params(value_net)

        set_flat_params(value_net_target,
                        (1 - polyak) * value_net_flat_params + polyak * value_net_target_flat_params)

    return {"target_value_loss": value_loss,
            "q_value_loss_1": q_value_loss_1,
            "q_value_loss_2": q_value_loss_2,
            "policy_loss": policy_loss
            }
