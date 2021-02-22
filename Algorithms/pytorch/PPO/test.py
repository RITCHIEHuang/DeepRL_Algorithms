#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/3 下午4:40

import click

from Algorithms.pytorch.PPO.ppo import PPO


@click.command()
@click.option("--env_id", type=str, default="Hopper-v2", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=4, help="Number of process to run environment")
@click.option("--lr_p", type=float, default=3e-4, help="Learning rate for Policy Net")
@click.option("--lr_v", type=float, default=3e-4, help="Learning rate for Value Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--tau", type=float, default=0.95, help="GAE factor")
@click.option("--epsilon", type=float, default=0.2, help="Clip rate for PPO")
@click.option("--batch_size", type=int, default=2048, help="Batch size")
@click.option("--ppo_mini_batch_size", type=int, default=0,
              help="PPO mini-batch size (default 0 -> don't use mini-batch update)")
@click.option("--ppo_epochs", type=int, default=10, help="PPO step")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
@click.option("--test_epochs", type=int, default=50, help="Trials to test model")
def main(env_id, render, num_process, lr_p, lr_v, gamma, tau, epsilon, batch_size,
         ppo_mini_batch_size, ppo_epochs, model_path, seed, test_epochs):
    ppo = PPO(env_id=env_id,
              render=render,
              num_process=num_process,
              min_batch_size=batch_size,
              lr_p=lr_p,
              lr_v=lr_v,
              gamma=gamma,
              tau=tau,
              clip_epsilon=epsilon,
              ppo_epochs=ppo_epochs,
              ppo_mini_batch_size=ppo_mini_batch_size,
              seed=seed,
              model_path=model_path)

    # for i_iter in range(1, test_epochs):
    #     ppo.eval(i_iter)

    # temp code
    import numpy as np
    from Utils.env_util import get_env_info

    n_trajs = 50
    obs_type = 0
    env, _, num_states, num_actions = get_env_info(env_id)

    states, actions, rewards, dones, next_states = [], [], [], [], []

    for i_iter in range(1, n_trajs + 1):

        state = env.reset()
        ep_reward = 0
        n_step = 0

        ep_states, ep_actions, ep_rewards, ep_dones, ep_next_states = [], [], [], [], []
        while True:
            if render:
                env.render()
            normalized_state = ppo.running_state(state)
            action = ppo.choose_action(normalized_state)
            next_state, reward, done, _ = env.step(action)
            normalized_next_state = ppo.running_state(next_state)

            ep_reward += reward
            n_step += 1

            ep_states.append(state if obs_type == 0 else normalized_state)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_next_states.append(next_state if obs_type ==
                                  0 else normalized_next_state)

            if done:
                if ep_reward > -200:
                    states.extend(ep_states)
                    actions.extend(ep_actions)
                    rewards.extend(ep_rewards)
                    dones.extend(ep_dones)
                    next_states.extend(ep_next_states)
                    print('Add success trajs !!!')
                print(
                    f"Iter: {i_iter}, step: {n_step}, episode Reward: {ep_reward}")
                break
            state = next_state

    env.close()

    states = np.r_[states].reshape((-1, num_states))
    next_states = np.r_[next_states].reshape((-1, num_states))
    actions = np.r_[actions].reshape((-1, 1))
    rewards = np.r_[rewards].reshape((-1, 1))
    dones = np.r_[dones].reshape((-1, 1))

    numpy_dict = {
        'obs': states,
        'action': actions,
        'reward': rewards,
        'done': dones,
        'next_obs': next_states
    }  # type: Dict[str, np.ndarray]

    np.savez(f"{env_id}.npz", **numpy_dict)


if __name__ == '__main__':
    main()
