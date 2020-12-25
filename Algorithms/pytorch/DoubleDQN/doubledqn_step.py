#!/usr/bin/env python
# Created at 2020/3/3
import torch
import torch.nn as nn

from Utils.torch_util import get_flat_params, set_flat_params


def doubledqn_step(value_net, optimizer_value, value_net_target, states, actions, rewards, next_states, masks, gamma,
                   polyak, update_target=False):
    masks = masks.unsqueeze(-1)
    rewards = rewards.unsqueeze(-1)
    actions = actions.unsqueeze(-1)
    """update q value net"""
    q_values = value_net(states).gather(1, actions)
    with torch.no_grad():
        q_target_next_values = value_net(next_states)
        q_target_actions = q_target_next_values.max(
            1)[1].view(q_values.size(0), 1)
        q_next_values = value_net_target(next_states)
        q_target_values = rewards + gamma * masks * \
            q_next_values.gather(1, q_target_actions).view(q_values.size(0), 1)

    value_loss = nn.MSELoss()(q_target_values, q_values)

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    if update_target:
        """update target q value net"""
        value_net_target.load_state_dict(value_net.state_dict())
        # value_net_target_flat_params = get_flat_params(value_net_target)
        # value_net_flat_params = get_flat_params(value_net)
        # set_flat_params(value_net_target, polyak * value_net_target_flat_params + (1 - polyak) * value_net_flat_params)

    return {"critic_loss": value_loss}
