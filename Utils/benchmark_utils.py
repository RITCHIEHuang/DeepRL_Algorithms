from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import torch

from DQN.DoubleDQN import DoubleDQN
from DQN.DuelingDQN import DuelingDQN
from DQN.NaiveDQN import NaiveDQN
from Utils.env_utils import get_env_space
from Utils.plot_util import Plot


class Bechmark:
    def __init__(self):
        self.plot = None

    def setup_plot(self, plot_refresh=0.001, x_label=None, y_label=None, title=None):
        self.plot = Plot(plot_refresh)
        self.plot.set_label_and_title(x_label, y_label, title)

    def run(self, env_id, alg_id, enalbe_gpu=True, num_episodes=1000, num_memory=5000):
        episodes = num_episodes
        memory_size = num_memory

        env, num_states, num_actions = get_env_space(env_id)

        if alg_id == 'DQN':
            agent = NaiveDQN(gamma=0.99, num_states=num_states, num_actions=num_actions, enable_gpu=enalbe_gpu,
                             memory_size=memory_size)

        elif alg_id == 'DoubleDQN':
            agent = DoubleDQN(gamma=0.99, num_states=num_states, num_actions=num_actions, enable_gpu=enalbe_gpu,
                              memory_size=memory_size)

        elif alg_id == 'DuelingDQN':
            agent = DuelingDQN(gamma=0.99, num_states=num_states, num_actions=num_actions, enable_gpu=enalbe_gpu,
                               memory_size=memory_size)

        iterations_, rewards_ = [], []
        # 迭代所有episodes进行采样
        for i in range(episodes):
            # 当前episode开始
            state = env.reset()
            episode_reward = 0

            while True:
                env.render()
                action = agent.choose_action(state, num_actions)
                next_state, reward, done, info = env.step(action)

                x, x_dot, theta, theta_hot = next_state
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2
                agent.memory.push(torch.tensor([state]),
                                  torch.tensor([action]),
                                  torch.tensor([r]),
                                  torch.tensor([next_state]))
                episode_reward += r

                if len(agent.memory) >= memory_size:
                    agent.learn()
                    if done:
                        iterations_.append(i)
                        rewards_.append(episode_reward)

                        self.plot.add_plot(iterations_, rewards_, str(alg_id))
                        print("episode: {} , the episode reward is {}".format(i, round(episode_reward, 3)))
                # 当前episode　结束
                if done:
                    break
                state = next_state
        env.close()


if __name__ == '__main__':
    bench = Bechmark()
    env_id = 'CartPole-v0'
    bench.setup_plot(0.001, 'Iterations', 'Rewards', env_id)

    envs = [env_id, env_id, env_id]
    algs = ['DQN', 'DoubleDQN', 'DuelingDQN']

    list(map(bench.run, envs, algs))
    # with ProcessPoolExecutor() as pool:
    #     pool.map(bench.run, envs, algs, gpus)
    # bench.run(env_id, 'DQN', True)
    # bench.run(env_id, 'DoubleDQN', True)
    # bench.run(env_id, 'DuelingDQN', True)
