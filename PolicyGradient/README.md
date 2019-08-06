# Policy Gradient 系列算法

基于策略的梯度算法，包括Monte-Carlo Policy Gradient->REINFORCE, 以及DDPG, TRPO, PPO等算法.

在理解算法的过程中，有一个难点是对于策略函数$\pi(a | s, \theta)$的梯度优化计算原理，因为策略函数是一个概率分布，

其梯度计算依赖于对分布的随机采样,pytorch等计算图框架中封装了这类的算法，其梯度计算原理见[Gradient Estimation Using Stochastic Computation Graphs][1]

## 算法细节及论文
- *REINFORCE* 

这算是PG最入门的算法了,基于轨迹更新.当然也有time step更新的版本, 可以理解为将算法中第二步中对于i的求和式进行分解.

![2]



## 实践效果
在gym的经典游戏MountainCar-v0中的表现：

![3]

![4]



[1]: https://arxiv.org/abs/1506.05254
[2]: images/REINFORCE%20alg.png
[3]: images/reinforce-mountaincar.gif
[4]: images/Reinforce%20MountainCar-v0.png