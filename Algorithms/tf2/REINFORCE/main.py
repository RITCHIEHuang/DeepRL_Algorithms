#!/usr/bin/env python
# Created at 2020/3/23
import click
import tensorflow as tf

from Algorithms.tf2.REINFORCE.reinforce import REINFORCE


@click.command()
@click.option("--env_id", type=str, default="BipedalWalker-v3", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=1, help="Number of process to run environment")
@click.option("--lr_p", type=float, default=1e-3, help="Learning rate for Policy Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--batch_size", type=int, default=1000, help="Batch size")
@click.option("--reinforce_epochs", type=int, default=5, help="Reinforce step")
@click.option("--max_iter", type=int, default=1000, help="Maximum iterations to run")
@click.option("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
@click.option("--save_iter", type=int, default=50, help="Iterations to save the model")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--log_path", type=str, default="../log/", help="Directory to save logs")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
def main(env_id, render, num_process, lr_p, gamma, batch_size,
         reinforce_epochs, max_iter, eval_iter, save_iter, model_path, log_path, seed):
    base_dir = log_path + env_id + "/REINFORCE_exp{}".format(seed)
    writer = tf.summary.create_file_writer(base_dir)

    reinforce = REINFORCE(env_id, render, num_process, batch_size, lr_p, gamma,
              reinforce_epochs, seed=seed)

    for i_iter in range(1, max_iter + 1):
        reinforce.learn(writer, i_iter)

        if i_iter % eval_iter == 0:
            reinforce.eval(i_iter, render=render)

        if i_iter % save_iter == 0:
            reinforce.save(model_path)


if __name__ == '__main__':
    main()
