#!/usr/bin/env python
# Created at 2020/3/31

import click
import torch
from torch.utils.tensorboard import SummaryWriter

from Algorithms.pytorch.A2C.a2c import A2C


@click.command()
@click.option("--env_id", type=str, default="BipedalWalker-v3", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=1, help="Number of process to run environment")
@click.option("--lr_ac", type=float, default=3e-4, help="Learning rate for Actor-Critic Net")
@click.option("--value_net_coeff", type=float, default=0.5, help="Coefficient for value loss")
@click.option("--entropy_coeff", type=float, default=1e-2, help="Coefficient for entropy loss")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--tau", type=float, default=0.95, help="GAE factor")
@click.option("--batch_size", type=int, default=1000, help="Batch size")
@click.option("--max_iter", type=int, default=1000, help="Maximum iterations to run")
@click.option("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
@click.option("--save_iter", type=int, default=50, help="Iterations to save the model")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--log_path", type=str, default="../log/", help="Directory to save logs")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
def main(env_id, render, num_process, lr_ac, value_net_coeff, entropy_coeff, gamma, tau, batch_size,
         max_iter, eval_iter, save_iter, model_path, log_path, seed):
    base_dir = log_path + env_id + "/A2C_exp{}".format(seed)
    writer = SummaryWriter(base_dir)

    a2c = A2C(env_id=env_id,
              render=render,
              num_process=num_process,
              min_batch_size=batch_size,
              lr_ac=lr_ac,
              value_net_coeff=value_net_coeff,
              entropy_coeff=entropy_coeff,
              gamma=gamma,
              tau=tau,
              seed=seed)

    for i_iter in range(1, max_iter + 1):
        a2c.learn(writer, i_iter)

        if i_iter % eval_iter == 0:
            a2c.eval(i_iter, render=render)

        if i_iter % save_iter == 0:
            a2c.save(model_path)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
