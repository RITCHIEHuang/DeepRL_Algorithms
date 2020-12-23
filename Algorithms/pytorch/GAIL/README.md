# GAIL

This GAIL implementation is highly correlated to [PPO](../PPO) algorithm:
- Expert trajectories generated according to **PPO** pre-trained model;
- GAIL learn policy utilizing **PPO** algorithm.

 
## 1. Usage

1. Generating expert trajectories by [expert_trajectory_collector.py](expert_trajecotry_collector.py) (You should pre-train a model by specific **RL** algorithm);
2. Filling in a custom config file for gail, a template is provided in [config/config.yml](config/config.yml);
3. Training GAIL from [main.py](main.py).


## 2. Performance

Run the algorithm on [BipedalWalker-v3](http://gym.openai.com/envs/BipedalWalker-v2/) for continuous control.

Expert trajectories are collected by running [PPO](../PPO), trajectories are saved as `.npz` format,
then `GAIL` utilizes `PPO` algorithm for policy optimization.

The performance (average reward) curve looks like this:

GAIL:
![GAIL](https://tva1.sinaimg.cn/large/007S8ZIlgy1gezddkse3cj30zi0ch74y.jpg)

PPO:
![PPO](https://tva1.sinaimg.cn/large/007S8ZIlgy1gezbk63ddaj30z90c53z3.jpg)

You may see that `GAIL` is not as good as `PPO`, however for imitating, `GAIL`  is good.