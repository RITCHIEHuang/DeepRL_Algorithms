import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DQN.Net import Net
from DQN.experience_replay import MemoryReplay, Transition

env = gym.make('CartPole-v0')
# 解除环境限制
env = env.unwrapped
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n


class DoubleQN:
    def __init__(self, learning_rate=0.01,
                 gamma=0.90,
                 batch_size=32,
                 epsilon=0.90,
                 memory_size=20000,
                 update_target_gap=100,
                 enable_gpu=False):
        # if enable_gpu:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_gap = update_target_gap
        self.epsilon = epsilon

        self.num_learn_step = 0

        self.memory = MemoryReplay(memory_size)
        self.eval_net, self.target_net = Net(num_states, num_actions), Net(num_states, num_actions)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    # greedy 策略动作选择
    def choose_action(self, state):
        state = torch.unsqueeze(torch.tensor(state), 0)
        if np.random.uniform() <= self.epsilon:  # greedy policy
            action_val = self.eval_net(state.float())
            action = torch.max(action_val, 1)[1].numpy()
            return action[0]
        else:
            action = np.random.randint(0, num_actions)
            return action

    def learn(self):
        # 更新目标网络 target_net
        if self.num_learn_step % self.update_target_gap == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.num_learn_step += 1

        # 从Memory中采batch
        sample = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*sample))
        batch_state = torch.cat(batch.state)
        batch_action = torch.stack(batch.action, 0)
        batch_reward = torch.stack(batch.reward, 0)
        batch_next_state = torch.cat(batch.next_state)

        # 训练网络 eval_net
        q_eval_ = self.eval_net(batch_state.float()).gather(1, batch_action)
        q_eval = self.eval_net(batch_next_state.float()).detach()
        max_a = q_eval.max(1)[1].view(self.batch_size, 1)
        # 不更新 target_net参数
        q_next = self.target_net(batch_next_state.float()).detach()
        # current_reward + gamma * Q_target(next_state, argmax_a q_eval)
        q_target = batch_reward + self.gamma * q_next.gather(1, max_a)
        # 计算误差
        loss = self.loss_func(q_eval_, q_target)

        # 更新训练网络 eval_net 梯度
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run():
    episodes = 400
    memory_size = 2000
    dqn = DoubleQN(enable_gpu=False, memory_size=memory_size)

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

            x, x_dot, theta, theta_hot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            dqn.memory.push(torch.tensor([state]),
                            torch.tensor([action]),
                            torch.tensor([r]),
                            torch.tensor([next_state]))
            episode_reward += r

            if len(dqn.memory) >= memory_size:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(episode_reward, 3)))
            # 当前episode　结束
            if done:
                break
            state = next_state
    env.close()


if __name__ == '__main__':
    run()
