import torch
from tensorboardX import SummaryWriter

from DQN.DoubleDQN import DoubleDQN
from DQN.DuelingDQN import DuelingDQN
from DQN.NaiveDQN import NaiveDQN
from Utils.env_utils import get_env_space

model_data = {}


class Bechmark:
    def __init__(self, writer):
        self.writer = writer

    def run(self, env_id, alg_id, enalbe_gpu=True, num_episodes=400, num_memory=2000):
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

                        writer.add_scalar(alg_id, episode_reward, i)
                        # model_data[alg_id + '_x'] = iterations_
                        # model_data[alg_id + '_y'] = rewards_
                        # plot.add_plot(iterations_, rewards_, color, label=alg_id, x_label='Iterations',
                        #               y_label='Rewards', title=env_id)
                        print("episode: {} , the episode reward is {}".format(i, round(episode_reward, 3)))
                # 当前episode　结束
                if done:
                    break
                state = next_state
        env.close()


if __name__ == '__main__':
    writer = SummaryWriter()
    bench = Bechmark(writer)
    env_id = 'CartPole-v0'

    # envs = [env_id, env_id, env_id]
    envs = [env_id]
    # algs = ['DQN', 'DoubleDQN', 'DuelingDQN']
    algs = ['DuelingDQN']

    list(map(bench.run, envs, algs))
    writer.close()
