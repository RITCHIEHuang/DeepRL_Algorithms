#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/3 下午4:40
import pickle

import click
import torch
from torch.utils.tensorboard import SummaryWriter

from Algorithms.pytorch.PPO.ppo import PPO


@click.command()
@click.option("--env_id", type=str, default="BipedalWalker-v3", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=4, help="Number of process to run environment")
@click.option("--lr_p", type=float, default=3e-4, help="Learning rate for Policy Net")
@click.option("--lr_v", type=float, default=3e-4, help="Learning rate for Value Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--tau", type=float, default=0.95, help="GAE factor")
@click.option("--epsilon", type=float, default=0.2, help="Clip rate for PPO")
@click.option("--batch_size", type=int, default=4000, help="Batch size")
@click.option("--ppo_mini_batch_size", type=int, default=500,
              help="PPO mini-batch size (default 0 -> don't use mini-batch update)")
@click.option("--ppo_epochs", type=int, default=10, help="PPO step")
@click.option("--max_iter", type=int, default=1000, help="Maximum iterations to run")
@click.option("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
@click.option("--save_iter", type=int, default=50, help="Iterations to save the model")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--log_path", type=str, default="../log/", help="Directory to save logs")
@click.option("--seed", type=int, default=11, help="Seed for reproducing")
def main(env_id, render, num_process, lr_p, lr_v, gamma, tau, epsilon, batch_size,
         ppo_mini_batch_size, ppo_epochs, max_iter, eval_iter, save_iter, model_path, log_path, seed):
    base_dir = log_path + env_id + "/PPO_exp{}".format(seed)
    writer = SummaryWriter(base_dir)

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
              seed=seed)

    for i_iter in range(1, max_iter + 1):
        ppo.learn(writer, i_iter)

        if i_iter % eval_iter == 0:
            ppo.eval(i_iter, render=render)

        if i_iter % save_iter == 0:
            ppo.save(model_path)

            pickle.dump(ppo,
                        open('{}/{}_ppo.p'.format(model_path, env_id), 'wb'))

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
