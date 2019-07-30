import random
from collections import namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make('CartPole-v0')
# 解除环境限制
env = env.unwrapped
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

# 经验重放策略: 提升DQN的稳定性
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class MemoryReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # 保存一条转换记录
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # 采样
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Q-network　结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, num_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        activation = self.out(x)
        return activation


class DQN:
    def __init__(self, learning_rate=0.01,
                 gamma=0.90,
                 batch_size=128,
                 epsilon=0.90,
                 episodes=4000,
                 memory_size=20000,
                 update_target_gap=100,
                 enable_gpu=False):
        # if enable_gpu:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.batch_size = batch_size
        self.episodes = episodes
        self.update_target_gap = update_target_gap
        self.epsilon = epsilon

        self.num_learn_step = 0

        self.memory = MemoryReplay(memory_size)
        self.eval_net, self.target_net = Net(), Net()
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    # greedy 策略动作选择
    def choose_action(self, state):
        state = torch.unsqueeze(torch.tensor(state), 0)
        if np.random.randn() <= self.epsilon:  # greedy policy
            action_val = self.eval_net.forward(state.float())
            action = torch.max(action_val, 1)[1].data.numpy()
            return action[0]
        else:
            action = np.random.randint(0, num_actions)
            return action

    def learn(self):
        # 更新目标网络 target_net
        if self.num_learn_step % self.update_target_gap:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.num_learn_step += 1

        # 从Memory中采batch
        sample = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*sample))
        batch_state = torch.cat(batch.state)
        batch_action = torch.stack(batch.action, 0)
        batch_reward = torch.stack(batch.reward, 0)
        batch_next_state = torch.cat(batch.next_state)

        # 训练eval_net
        q_eval = self.eval_net(batch_state.float()).gather(1, batch_action)
        # 不更新 target_net参数
        q_next = self.target_net(batch_next_state.float()).detach()
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        # 计算误差
        loss = self.loss_func(q_eval, q_target)

        # 更新梯度
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run():
    episodes = 400
    memory_size = 2000
    dqn = DQN(enable_gpu=True)

    sample_memory_counter = 0
    # 迭代所有episodes进行采样
    for i in range(episodes):
        # 当前episode开始
        state = env.reset()
        episode_reward = 0

        while True:
            env.render()
            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action)
            dqn.memory.push(torch.tensor([state]),
                            torch.tensor([action]),
                            torch.tensor([reward]),
                            torch.tensor([next_state]))
            sample_memory_counter += 1
            episode_reward += reward

            if len(dqn.memory) >= memory_size:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(episode_reward, 3)))
            # 当前episode　结束
            if done:
                break
            state = next_state


if __name__ == '__main__':
    run()
