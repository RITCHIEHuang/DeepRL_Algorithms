#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/3 下午4:40

import click

from Algorithms.pytorch.VPG.vpg import VPG


@click.command()
@click.option("--env_id", type=str, default="BipedalWalker-v2", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=4, help="Number of process to run environment")
@click.option("--lr_p", type=float, default=3e-4, help="Learning rate for Policy Net")
@click.option("--lr_v", type=float, default=3e-4, help="Learning rate for Value Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--tau", type=float, default=0.95, help="GAE factor")
@click.option("--batch_size", type=int, default=2048, help="Batch size")
@click.option("--vpg_epochs", type=int, default=10, help="VPG step")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
@click.option("--test_epochs", type=int, default=50, help="Trials to test model")
def main(env_id, render, num_process, lr_p, lr_v, gamma, tau, batch_size,
         vpg_epochs, model_path, seed, test_epochs):
    vpg = VPG(env_id, render, num_process, batch_size, lr_p, lr_v, gamma, tau,
              vpg_epochs, seed=seed, model_path=model_path)

    for i_iter in range(1, test_epochs):
        vpg.eval(i_iter)


if __name__ == '__main__':
    main()
