#!/usr/bin/env python
# Created at 2020/1/22
import torch

from Utils.torch_util import device, DOUBLE


def reinforce_step(policy_net, optimizer_policy, states, actions, rewards, masks, gamma, eps=1e-6):
    """calculate cumulative reward"""
    cum_rewards = DOUBLE(rewards.size(0), 1).to(device)
    pre_value = 0
    for i in reversed(range(rewards.size(0))):
        pre_value = gamma * masks[i] * pre_value + rewards[i, 0]
        cum_rewards[i, 0] = pre_value

    # normalize cumulative rewards
    cum_rewards = (cum_rewards - cum_rewards.mean()) / \
        (cum_rewards.std() + eps)

    """update policy"""
    log_probs = policy_net.get_log_prob(states, actions)
    policy_loss = -(log_probs * cum_rewards).mean()

    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()

    return {"policy_loss": policy_loss}
