#!/usr/bin/env python
# Created at 2020/2/15
import click
import torch

from Algorithms.pytorch.REINFORCE.reinforce import REINFORCE


@click.command()
@click.option("--env_id", type=str, default="BipedalWalker-v3", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=4, help="Number of process to run environment")
@click.option("--lr_p", type=float, default=3e-4, help="Learning rate for Policy Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--batch_size", type=int, default=3000, help="Batch size")
@click.option("--reinforce_epochs", type=int, default=5, help="Reinforce step")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
@click.option("--test_epochs", type=int, default=50, help="Trials to test model")
def main(env_id, render, num_process, lr_p, gamma, batch_size,
         reinforce_epochs, model_path, seed, test_epochs):
    reinforce = REINFORCE(env_id, render, num_process, batch_size, lr_p, gamma,
                          reinforce_epochs, seed=seed, model_path=model_path)

    for i_iter in range(1, test_epochs + 1):
        reinforce.eval(i_iter)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
