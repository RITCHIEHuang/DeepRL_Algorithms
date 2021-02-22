#!/usr/bin/env python
# Created at 2020/5/13
import pickle

import click
import numpy as np
import torch

from Utils.env_util import get_env_info
from Utils.file_util import check_path


@click.command()
@click.option("--env_id", type=str, default="MountainCarContinuous-v0", help="Environment Id")
@click.option("--n_trajs", type=int, default=100,
              help="Number of trajectories to sample")
@click.option("--model_path", type=str,
              default="../PPO/trained_models/MountainCarContinuous-v0_ppo.p",
              help="Directory to load pre-trained model")
@click.option("--data_path", type=str, default="./data",
              help="Directory to store expert trajectories")
@click.option("--render", type=bool, default=True,
              help="Render environment flag")
@click.option("--seed", type=int, default=2020,
              help="Random seed for reproducing")
@click.option("--obs_type", type=int, default=0,
              help="Observation type (0 -> raw, 1 -> normalized)")
def main(env_id, n_trajs, model_path, data_path, render, seed, obs_type):
    """
    Collect trajectories from pre-trained models by PPO
    """
    if data_path is not None:
        check_path(data_path)

    env, _, num_states, num_actions = get_env_info(env_id)

    # seed
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = pickle.load(open(model_path, 'rb'))
    model.running_state.fix = True
    states, actions, rewards, dones, next_states = [], [], [], [], []

    for i_iter in range(1, n_trajs + 1):
        state = env.reset()
        ep_reward = 0
        n_step = 0

        ep_states, ep_actions, ep_rewards, ep_dones, ep_next_states = [], [], [], [], []
        while True:
            if render:
                env.render()
            normalized_state = model.running_state(state)
            action = model.choose_action(normalized_state)
            next_state, reward, done, _ = env.step(action)
            normalized_next_state = model.running_state(next_state)

            ep_reward += reward
            n_step += 1

            ep_states.append(state if obs_type == 0 else normalized_state)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_next_states.append(next_state if obs_type == 0
                                  else
                                  normalized_next_state)

            if done:
                states.extend(ep_states)
                actions.extend(ep_actions)
                rewards.extend(ep_rewards)
                dones.extend(ep_dones)
                next_states.extend(ep_next_states)
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

    save_path = f"{data_path}/{env_id}" if data_path is not None else env_id
    np.savez(f"{save_path}.npz", **numpy_dict)


if __name__ == '__main__':
    main()
