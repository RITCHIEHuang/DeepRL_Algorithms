#!/usr/bin/env python
# Created at 2020/3/22

import click

from Algorithms.tf2.DuelingDQN.duelingdqn import DuelingDQN


@click.command()
@click.option("--env_id", type=str, default="MountainCar-v0", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=1, help="Number of process to run environment")
@click.option("--lr", type=float, default=1e-3, help="Learning rate for Policy Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--epsilon", type=float, default=0.90, help="Probability controls greedy action")
@click.option("--explore_size", type=int, default=10000, help="Explore steps before execute deterministic policy")
@click.option("--memory_size", type=int, default=1000000, help="Size of replay memory")
@click.option("--step_per_iter", type=int, default=4000, help="Number of steps of interaction in each iteration")
@click.option("--batch_size", type=int, default=256, help="Batch size")
@click.option("--min_update_step", type=int, default=1000, help="Minimum interacts for updating")
@click.option("--update_target_gap", type=int, default=50, help="Steps between updating target q net")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
@click.option("--test_epochs", type=int, default=50, help="Trials to test trained model")
def main(env_id, render, num_process, lr, gamma, epsilon, explore_size, memory_size, step_per_iter, batch_size,
         min_update_step, update_target_gap, model_path, seed, test_epochs):
    duelingdqn = DuelingDQN(env_id,
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

    for i_iter in range(1, test_epochs + 1):
        duelingdqn.eval(i_iter)


if __name__ == '__main__':
    main()
