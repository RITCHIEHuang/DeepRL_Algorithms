#!/usr/bin/env python
# Created at 2020/5/13
import pickle

import click
import numpy as np
import torch

from Utils.env_util import get_env_info


@click.command()
@click.option("--env_id", type=str, default="Swimmer-v3", help="Environment Id")
@click.option("--n_trajs", type=int, default=1000, help="Number of trajectories to sample")
@click.option("--model_path", type=str, default="../PPO/trained_models/Swimmer-v3_ppo.p",
              help="Directory to load pre-trained model")
@click.option("--data_path", type=str, default="./data", help="Directory to store expert trajectories")
@click.option("--render", type=bool, default=False, help="Render environment flag")
@click.option("--seed", type=int, default=2020, help="Random seed for reproducing")
def main(env_id, n_trajs, model_path, data_path, render, seed):
    """
    Collect trajectories from pre-trained models by PPO
    """
    env, _, num_states, num_actions = get_env_info(env_id)

    # seed
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    states, actions, rewards, ep_rewards = [], [], [], []

    model = pickle.load(open(model_path, 'rb'))
    model.running_state.fix = True
    for i_iter in range(1, n_trajs + 1):

        state = env.reset()
        ep_reward = 0
        n_step = 0

        while True:
            if render:
                env.render()
            state = model.running_state(state)
            action, _ = model.choose_action(state)
            action = action.cpu().numpy()[0]
            state, reward, done, _ = env.step(action)

            ep_reward += reward
            n_step += 1

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            if done:
                ep_rewards.append(ep_reward)
                print(f"Iter: {i_iter}, step: {n_step}, episode Reward: {ep_reward}")
                break

    env.close()

    states = np.r_[states].reshape((-1, num_states))
    actions = np.r_[actions].reshape((-1, num_actions))
    rewards = np.r_[rewards].reshape((-1, 1))
    ep_rewards = np.r_[ep_rewards].reshape((n_trajs, -1))

    numpy_dict = {
        'state': states,
        'action': actions,
        'reward': rewards,
        'ep_reward': ep_rewards,
    }  # type: Dict[str, np.ndarray]

    if data_path is not None:
        np.savez(f"{data_path}/{env_id}.npz", **numpy_dict)


if __name__ == '__main__':
    main()
