#!/usr/bin/env python
# Created at 2020/3/23
import click

import tensorflow as tf
from Algorithms.tf2.VPG.vpg import VPG


@click.command()
@click.option("--env_id", type=str, default="BipedalWalker-v3", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=1, help="Number of process to run environment")
@click.option("--lr_p", type=float, default=3e-4, help="Learning rate for Policy Net")
@click.option("--lr_v", type=float, default=1e-3, help="Learning rate for Value Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--tau", type=float, default=0.95, help="GAE factor")
@click.option("--batch_size", type=int, default=1000, help="Batch size")
@click.option("--vpg_epochs", type=int, default=10, help="Vanilla PG step")
@click.option("--max_iter", type=int, default=1000, help="Maximum iterations to run")
@click.option("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
@click.option("--save_iter", type=int, default=50, help="Iterations to save the model")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--log_path", type=str, default="../log/", help="Directory to save logs")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
def main(env_id, render, num_process, lr_p, lr_v, gamma, tau, batch_size,
         vpg_epochs, max_iter, eval_iter, save_iter, model_path, log_path, seed):
    base_dir = log_path + env_id + "/VPG_exp{}".format(seed)
    writer = tf.summary.create_file_writer(base_dir)

    vpg = VPG(env_id, render, num_process, batch_size, lr_p, lr_v, gamma, tau,
              vpg_epochs, seed=seed)

    for i_iter in range(1, max_iter + 1):
        vpg.learn(writer, i_iter)

        if i_iter % eval_iter == 0:
            vpg.eval(i_iter, render=render)

        if i_iter % save_iter == 0:
            vpg.save(model_path)


if __name__ == '__main__':
    main()
