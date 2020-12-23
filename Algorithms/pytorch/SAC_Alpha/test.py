#!/usr/bin/env python
# Created at 2020/3/27

import click

from Algorithms.pytorch.SAC_Alpha.sac_alpha import SAC_Alpha


@click.command()
@click.option("--env_id", type=str, default="BipedalWalker-v3", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=1, help="Number of process to run environment")
@click.option("--lr_p", type=float, default=1e-3, help="Learning rate for Policy Net")
@click.option("--lr_a", type=float, default=3e-4, help="Learning rate for Temperature")
@click.option("--lr_q", type=float, default=1e-3, help="Learning rate for QValue Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--polyak", type=float, default=0.995,
              help="Interpolation factor in polyak averaging for target networks")
@click.option("--explore_size", type=int, default=10000, help="Explore steps before execute deterministic policy")
@click.option("--memory_size", type=int, default=1000000, help="Size of replay memory")
@click.option("--step_per_iter", type=int, default=4000, help="Number of steps of interaction in each iteration")
@click.option("--batch_size", type=int, default=100, help="Batch size")
@click.option("--min_update_step", type=int, default=1000, help="Minimum interacts for updating")
@click.option("--update_step", type=int, default=50, help="Steps between updating policy and critic")
@click.option("--test_epochs", type=int, default=50, help="Trials to test model")
@click.option("--target_update_delay", type=int, default=1, help="Frequency for target QValue Net update")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
def main(env_id, render, num_process, lr_p, lr_a, lr_q, gamma, polyak, explore_size, memory_size,
         step_per_iter, batch_size, min_update_step, update_step, test_epochs,
         target_update_delay, model_path, seed):
    sac_alpha = SAC_Alpha(env_id,
                    render=render,
                    num_process=num_process,
                    memory_size=memory_size,
                    lr_p=lr_p,
                    lr_a=lr_a,
                    lr_q=lr_q,
                    gamma=gamma,
                    polyak=polyak,
                    explore_size=explore_size,
                    step_per_iter=step_per_iter,
                    batch_size=batch_size,
                    min_update_step=min_update_step,
                    update_step=update_step,
                    target_update_delay=target_update_delay,
                    seed=seed,
                    model_path=model_path
                    )

    for i_iter in range(1, test_epochs + 1):
        sac_alpha.eval(i_iter)


if __name__ == '__main__':
    main()
