#!/usr/bin/env python
# Created at 2020/2/9

import click
import torch
from torch.utils.tensorboard import SummaryWriter

from Algorithms.pytorch.TRPO.trpo import TRPO


@click.command()
@click.option(
    "--env_id", type=str, default="MountainCar-v0", help="Environment Id"
)
@click.option(
    "--render", type=bool, default=False, help="Render environment or not"
)
@click.option(
    "--num_process",
    type=int,
    default=1,
    help="Number of process to run environment",
)
@click.option(
    "--lr_v", type=float, default=3e-4, help="Learning rate for Value Net"
)
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--tau", type=float, default=0.95, help="GAE factor")
@click.option(
    "--max_kl", type=float, default=1e-2, help="kl constraint for TRPO"
)
@click.option("--damping", type=float, default=1e-2, help="damping for TRPO")
@click.option("--batch_size", type=int, default=1000, help="Batch size")
@click.option(
    "--max_iter", type=int, default=1000, help="Maximum iterations to run"
)
@click.option(
    "--eval_iter",
    type=int,
    default=50,
    help="Iterations to evaluate the model",
)
@click.option(
    "--save_iter", type=int, default=50, help="Iterations to save the model"
)
@click.option(
    "--model_path",
    type=str,
    default="trained_models",
    help="Directory to store model",
)
@click.option(
    "--log_path", type=str, default="../log/", help="Directory to save logs"
)
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
def main(
    env_id,
    render,
    num_process,
    lr_v,
    gamma,
    tau,
    max_kl,
    damping,
    batch_size,
    max_iter,
    eval_iter,
    save_iter,
    log_path,
    model_path,
    seed,
):
    base_dir = log_path + env_id + "/TRPO_exp{}".format(seed)
    writer = SummaryWriter(base_dir)
    trpo = TRPO(
        env_id,
        render,
        num_process,
        batch_size,
        lr_v,
        gamma,
        tau,
        max_kl,
        damping,
        seed=seed,
    )

    for i_iter in range(1, max_iter + 1):
        trpo.learn(writer, i_iter)

        if i_iter % eval_iter == 0:
            trpo.eval(i_iter, render=render)

        if i_iter % save_iter == 0:
            trpo.save(model_path)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
