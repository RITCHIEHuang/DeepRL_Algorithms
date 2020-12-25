#!/usr/bin/env python
# Created at 2020/2/9

import click
import torch
from torch.utils.tensorboard import SummaryWriter

from Algorithms.pytorch.DDPG.ddpg import DDPG


@click.command()
@click.option("--env_id", type=str, default="HalfCheetah-v3", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=1, help="Number of process to run environment")
@click.option("--lr_p", type=float, default=1e-3, help="Learning rate for Policy Net")
@click.option("--lr_v", type=float, default=1e-3, help="Learning rate for Value Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--polyak", type=float, default=0.995,
              help="Interpolation factor in polyak averaging for target networks")
@click.option("--explore_size", type=int, default=10000, help="Explore steps before execute deterministic policy")
@click.option("--memory_size", type=int, default=1000000, help="Size of replay memory")
@click.option("--step_per_iter", type=int, default=1000, help="Number of steps of interaction in each iteration")
@click.option("--batch_size", type=int, default=256, help="Batch size")
@click.option("--min_update_step", type=int, default=1000, help="Minimum interacts for updating")
@click.option("--update_step", type=int, default=50, help="Steps between updating policy and critic")
@click.option("--max_iter", type=int, default=500, help="Maximum iterations to run")
@click.option("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
@click.option("--save_iter", type=int, default=50, help="Iterations to save the model")
@click.option("--action_noise", type=float, default=0.1, help="Std for noise of action")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--log_path", type=str, default="../log/", help="Directory to save logs")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
def main(env_id, render, num_process, lr_p, lr_v, gamma, polyak, explore_size, memory_size, step_per_iter, batch_size,
         min_update_step, update_step, max_iter, eval_iter, save_iter, action_noise, model_path, log_path, seed):
    base_dir = log_path + env_id + "/DDPG_exp{}".format(seed)
    writer = SummaryWriter(base_dir)

    ddpg = DDPG(env_id,
                render=render,
                num_process=num_process,
                memory_size=memory_size,
                lr_p=lr_p,
                lr_v=lr_v,
                gamma=gamma,
                polyak=polyak,
                explore_size=explore_size,
                step_per_iter=step_per_iter,
                batch_size=batch_size,
                min_update_step=min_update_step,
                update_step=update_step,
                action_noise=action_noise,
                seed=seed)

    for i_iter in range(1, max_iter + 1):
        ddpg.learn(writer, i_iter)

        if i_iter % eval_iter == 0:
            ddpg.eval(i_iter, render=render)

        if i_iter % save_iter == 0:
            ddpg.save(model_path)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
