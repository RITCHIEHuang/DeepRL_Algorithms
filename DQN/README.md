# DQN 系列算法实现
Deep Reinforcement Learning的入门级算法

Deep Q-learning Network估计 action-value 映射函数,使用Experience-Replay采样,可有效提升 data efficiency,并且能够处理 non-stationary situation.

## 算法细节及论文
- *Basic DQN* [Playing Atari with Deep Reinforcement Learning][2]

![1]

- *Double DQN* [Deep Reinforcement Learning with Double Q-learning][3]

使用两个非同步的网络,进行交替更新.
![4]

- *Dueling DQN* [Dueling Network Architectures for Deep Reinforcement Learning
][5]

更改Basic DQN的网络结构.
![6]

使用优化技巧使网络平衡地更新到 state value 和 advantage action(state-dependent) value.
![7]

[1]: images/DQN%20with%20Experience%20Replay.png
[2]: https://arxiv.org/abs/1312.5602
[3]: https://arxiv.org/abs/1509.06461
[4]: images/Double%20DQN%20Algorithm.png
[5]: https://arxiv.org/abs/1511.06581
[6]: images/Dueling%20DQN%20Network.png
[7]: images/Dueling%20DQN%20optimization%20for%20identifiability.png
