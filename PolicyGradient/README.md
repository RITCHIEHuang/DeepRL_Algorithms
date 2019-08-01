# Policy Gradient 系列算法

基于策略的梯度算法，包括Monte-Carlo Policy Gradient->REINFORCE, 以及DDPG, TRPO, PPO等算法.

在理解算法的过程中，有一个难点是对于策略函数$\pi(a | s, \theta)$的梯度优化计算原理，因为策略函数是一个概率分布，

其梯度计算依赖于对分布的随机采样,pytorch等计算图框架中封装了这类的算法，其梯度计算原理见[Gradient Estimation Using Stochastic Computation Graphs][1]





[1]: https://arxiv.org/abs/1506.05254