#!/usr/bin/env python
# Created at 2020/3/3
import torch
import torch.nn as nn


def dqn_step(value_net, optimizer_value, value_net_target, states, actions, rewards, next_states, masks, gamma):
    masks = masks.unsqueeze(-1)
    rewards = rewards.unsqueeze(-1)
    actions = actions.unsqueeze(-1)
    """update value net"""
    q_values = value_net(states).gather(1, actions)
    with torch.no_grad():
        q_target_next_values = value_net_target(next_states)
        q_target_values = rewards + gamma * masks * \
            q_target_next_values.max(1)[0].view(q_values.size(0), 1)

    value_loss = nn.MSELoss()(q_target_values, q_values)

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    return {"critic_loss": value_loss}
