#!/usr/bin/env python
# Created at 2020/3/27
import torch
import torch.nn as nn

from Utils.torch_util import get_flat_params, set_flat_params


def sac_alpha_step(policy_net, q_net_1, q_net_2, alpha, q_net_target_1, q_net_target_2,
                   optimizer_policy, optimizer_q_net_1, optimizer_q_net_2, optimizer_a,
                   states, actions, rewards, next_states, masks, gamma, polyak, target_entropy,
                   update_target=False):
    rewards = rewards.unsqueeze(-1)
    masks = masks.unsqueeze(-1)

    """update qvalue net"""
    with torch.no_grad():
        next_actions, next_log_probs = policy_net.rsample(next_states)
        target_q_value = torch.min(
            q_net_target_1(next_states, next_actions),
            q_net_target_2(next_states, next_actions)
        ) - alpha * next_log_probs
        target_q_value = rewards + gamma * masks * target_q_value

    q_value_1 = q_net_1(states, actions)
    q_value_loss_1 = nn.MSELoss()(q_value_1, target_q_value)
    optimizer_q_net_1.zero_grad()
    q_value_loss_1.backward()
    optimizer_q_net_1.step()

    q_value_2 = q_net_2(states, actions)
    q_value_loss_2 = nn.MSELoss()(q_value_2, target_q_value)
    optimizer_q_net_2.zero_grad()
    q_value_loss_2.backward()
    optimizer_q_net_2.step()

    """update policy net"""
    new_actions, log_probs = policy_net.rsample(states)
    min_q = torch.min(
        q_net_1(states, new_actions),
        q_net_2(states, new_actions)
    )

    policy_loss = (alpha * log_probs - min_q).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    """update alpha"""
    alpha_loss = - alpha * (log_probs.detach() + target_entropy).mean()
    optimizer_a.zero_grad()
    alpha_loss.backward()
    optimizer_a.step()

    if update_target:
        """ 
        soft update target qvalue net 
        """
        q_net_1_flat_params = get_flat_params(q_net_1)
        q_net_target_1_flat_params = get_flat_params(q_net_target_1)

        set_flat_params(q_net_target_1,
                        (1 - polyak) * q_net_1_flat_params + polyak * q_net_target_1_flat_params)

        q_net_2_flat_params = get_flat_params(q_net_2)
        q_net_target_2_flat_params = get_flat_params(q_net_target_2)

        set_flat_params(q_net_target_2,
                        (1 - polyak) * q_net_2_flat_params + polyak * q_net_target_2_flat_params)

    return {"q_value_loss_1": q_value_loss_1,
            "q_value_loss_2": q_value_loss_2,
            "policy_loss": policy_loss,
            "alpha_loss": alpha_loss
            }
