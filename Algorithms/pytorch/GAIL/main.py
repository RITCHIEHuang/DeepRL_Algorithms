#!/usr/bin/env python
# Created at 2020/3/14
import time

import click
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Algorithms.pytorch.GAIL.gail import GAIL


@click.command()
@click.option("--env_id", type=str, default="BipedalWalker-v3", help="Environment Id")
@click.option("--config_path", type=str, default="./config/config.yml",
              help="Model configuration file")
@click.option("--expert_data_path", type=str, default="data/BipedalWalker-v3.npz", help="Expert data path")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=4, help="Number of process to run environment")
@click.option("--eval_model_epoch", type=int, default=50, help="Intervals for evaluating model")
@click.option("--save_model_epoch", type=int, default=50, help="Intervals for saving model")
@click.option("--save_model_path", type=str, default="trained_models", help="Path for saving trained model")
@click.option("--load_model", type=bool, default=False, help="Indicator for whether load trained model")
@click.option("--load_model_path", type=str, default="trained_models",
              help="Path for loading trained model")
@click.option("--log_path", type=str, default="./log/", help="Directory to save logs")
def main(env_id, config_path, expert_data_path, render, num_process, eval_model_epoch, save_model_epoch,
         save_model_path, load_model, load_model_path, log_path):
    base_dir = f"{log_path}/GAIL_{env_id}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"
    writer = SummaryWriter(base_dir)

    config = config_loader(path=config_path)  # load model configuration
    training_epochs = config["train"]["general"]["training_epochs"]

    gail = GAIL(env_id=env_id,
                config=config,
                expert_data_path=expert_data_path,
                render=render,
                num_process=num_process)

    if load_model:
        print(f"Loading Pre-trained GAIL model from {load_model_path}!!!")
        gail.load_model(load_model_path)

    for epoch in tqdm(range(1, training_epochs + 1)):
        gail.learn(writer, epoch)

        if epoch % eval_model_epoch == 0:
            gail.eval(epoch)

        if epoch % save_model_epoch == 0:
            gail.save_model(save_model_path)


def config_loader(path=None):
    with open(path) as f:
        config = yaml.full_load(f)
    return config


if __name__ == '__main__':
    main()
