#!/usr/bin/env python
# Created at 2020/2/9

import click

from Algorithms.pytorch.DQN.dqn import DQN


@click.command()
@click.option("--env_id", type=str, default="MountainCar-v0", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=1, help="Number of process to run environment")
@click.option("--lr", type=float, default=1e-3, help="Learning rate for Policy Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--epsilon", type=float, default=0.90, help="Probability controls greedy action")
@click.option("--explore_size", type=int, default=10000, help="Explore steps before execute deterministic policy")
@click.option("--memory_size", type=int, default=1000000, help="Size of replay memory")
@click.option("--step_per_iter", type=int, default=3500, help="Number of steps of interaction in each iteration")
@click.option("--batch_size", type=int, default=100, help="Batch size")
@click.option("--min_update_step", type=int, default=1000, help="Minimum interacts for updating")
@click.option("--update_target_gap", type=int, default=50, help="Steps between updating target q net")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
@click.option("--test_epochs", type=int, default=50, help="Trials to test trained model")
def main(env_id, render, num_process, lr, gamma, epsilon, explore_size, memory_size, step_per_iter, batch_size,
         min_update_step, update_target_gap, model_path, seed, test_epochs):
    dqn = DQN(env_id,
              render=render,
              num_process=num_process,
              memory_size=memory_size,
              lr_q=lr,
              gamma=gamma,
              epsilon=epsilon,
              explore_size=explore_size,
              step_per_iter=step_per_iter,
              batch_size=batch_size,
              min_update_step=min_update_step,
              update_target_gap=update_target_gap,
              seed=seed,
              model_path=model_path)

    # for i_iter in range(1, test_epochs + 1):
    #     dqn.eval(i_iter)

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
            normalized_state = dqn.running_state(state)
            action = dqn.choose_action(normalized_state)
            next_state, reward, done, _ = env.step(action)
            normalized_next_state = dqn.running_state(next_state)

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
