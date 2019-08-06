import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.distributions.categorical import Categorical

from PolicyGradient.Net import PolicyNet
from Utils.env_utils import get_env_space


class REINFORCE:
    def __init__(self,
                 num_states,
                 num_actions,
                 learning_rate=0.01,
                 gamma=0.90,
                 enable_gpu=False
                 ):

        if enable_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda")

        self.policy = PolicyNet(num_states, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma

        self.rewards = []  # 记录轨迹的每个 time step 对应的及时回报 r_t
        self.log_probs = []  # 记录轨迹的每个 time step 对应的 log_probability
        self.cum_rewards = []  # 记录轨迹的每个 time step 对应的 累计回报 G_t

    def calc_cumulative_rewards(self):
        R = 0.0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            self.cum_rewards.insert(0, R)

    def choose_action(self, state):
        state = torch.tensor(state).unsqueeze(0).to(self.device).float()
        probs = self.policy(state)

        # 对action进行采样,并计算log probability
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        self.log_probs.append(log_prob)

        return action.item()

    def update_episode(self):
        self.calc_cumulative_rewards()

        # 梯度上升更新策略参数
        loss = -torch.tensor(self.log_probs, requires_grad=True).to(self.device).mul(
            torch.tensor(self.cum_rewards).to(self.device)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rewards.clear()
        self.log_probs.clear()
        self.cum_rewards.clear()


if __name__ == '__main__':
    env_id = 'MountainCar-v0'
    alg_id = 'REINFORCE'
    env, num_states, num_actions = get_env_space(env_id)

    agent = REINFORCE(num_states, num_actions, enable_gpu=True)
    episodes = 400

    writer = SummaryWriter()
    iterations_ = []
    rewards_ = []

    # 迭代所有episodes进行采样
    for i in range(episodes):
        # 当前episode开始
        state = env.reset()
        episode_reward = 0

        while True:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            episode_reward += reward

            agent.rewards.append(reward)
            if done:
                iterations_.append(i)
                rewards_.append(episode_reward)

                writer.add_scalar(alg_id, episode_reward, i)
                print("episode: {} , the episode reward is {}".format(i, round(episode_reward, 3)))

            # 当前episode　结束
            if done:
                break
            state = next_state

        agent.update_episode()

    env.close()
