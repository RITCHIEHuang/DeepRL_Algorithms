#!/usr/bin/env python
# Created at 2020/3/22
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as optim

from Algorithms.tf2.DoubleDQN.doubledqn_step import doubledqn_step
from Algorithms.tf2.Models.QNet_dqn import QNet_dqn
from Common.replay_memory import Memory
from Utils.env_util import get_env_info
from Utils.file_util import check_path
from Utils.tf2_util import NDOUBLE, NLONG
from Utils.zfilter import ZFilter


class DoubleDQN:
    def __init__(self,
                 env_id,
                 render=False,
                 num_process=1,
                 memory_size=100000,
                 explore_size=10000,
                 step_per_iter=3000,
                 lr_q=1e-3,
                 gamma=0.99,
                 polyak=0.995,
                 batch_size=128,
                 min_update_step=1000,
                 epsilon=0.90,
                 update_target_gap=50,
                 seed=1,
                 model_path=None
                 ):
        self.env_id = env_id
        self.render = render
        self.num_process = num_process
        self.memory = Memory(size=memory_size)
        self.explore_size = explore_size
        self.step_per_iter = step_per_iter
        self.lr_q = lr_q
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.min_update_step = min_update_step
        self.update_target_gap = update_target_gap
        self.epsilon = epsilon
        self.seed = seed
        self.model_path = model_path

        self._init_model()

    def _init_model(self):
        """init model from parameters"""
        self.env, env_continuous, num_states, self.num_actions = get_env_info(
            self.env_id)
        assert not env_continuous, "DoubleQN is only applicable to discontinuous environment !!!!"

        tf.keras.backend.set_floatx('float64')

        # seeding
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.env.seed(self.seed)

        # initialize networks
        self.value_net = QNet_dqn(num_states, self.num_actions)
        self.value_net_target = QNet_dqn(num_states, self.num_actions)
        self.running_state = ZFilter((num_states,), clip=5)

        # load model if necessary
        if self.model_path:
            print("Loading Saved Model {}_double_dqn_tf2.p".format(self.env_id))
            self.running_state = pickle.load(
                open('{}/{}_double_dqn_tf2.p'.format(self.model_path, self.env_id), "rb"))
            self.value_net.load_weights(
                "{}/{}_double_dqn_tf2".format(self.model_path, self.env_id))

        self.value_net_target.set_weights(self.value_net.get_weights())

        self.optimizer = optim.Adam(lr=self.lr_q)

    def choose_action(self, state):
        state = np.expand_dims(NDOUBLE(state), 0)
        if np.random.uniform() <= self.epsilon:
            action = self.value_net.get_action(state)
            action = action.numpy()[0]
        else:  # choose action greedy
            action = np.random.randint(0, self.num_actions)
        return action

    def eval(self, i_iter, render=False):
        """evaluate model"""
        state = self.env.reset()
        test_reward = 0
        while True:
            self.env.render()
            state = self.running_state(state)
            action = self.choose_action(state)
            state, reward, done, _ = self.env.step(action)

            test_reward += reward
            if done:
                break
        print(f"Iter: {i_iter}, test Reward: {test_reward}")
        self.env.close()

    def learn(self, writer, i_iter):
        """interact"""
        global_steps = (i_iter - 1) * self.step_per_iter + 1
        log = dict()
        num_steps = 0
        num_episodes = 0
        total_reward = 0
        min_episode_reward = float('inf')
        max_episode_reward = float('-inf')

        while num_steps < self.step_per_iter:
            state = self.env.reset()
            state = self.running_state(state)
            episode_reward = 0

            for t in range(10000):
                if self.render:
                    self.env.render()

                if global_steps < self.explore_size:  # explore
                    action = self.env.action_space.sample()
                else:  # choose according to target net
                    action = self.choose_action(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = self.running_state(next_state)
                mask = 0 if done else 1
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                self.memory.push(state, action, reward, next_state, mask, None)

                episode_reward += reward
                global_steps += 1
                num_steps += 1

                if global_steps >= self.min_update_step:
                    batch = self.memory.sample(
                        self.batch_size)  # random sample batch
                    self.update(batch, global_steps)

                if done or num_steps >= self.step_per_iter:
                    break

                state = next_state

            num_episodes += 1
            total_reward += episode_reward
            min_episode_reward = min(episode_reward, min_episode_reward)
            max_episode_reward = max(episode_reward, max_episode_reward)

        self.env.close()

        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_episode_reward'] = max_episode_reward
        log['min_episode_reward'] = min_episode_reward

        print(f"Iter: {i_iter}, num steps: {log['num_steps']}, total reward: {log['total_reward']: .4f}, "
              f"min reward: {log['min_episode_reward']: .4f}, max reward: {log['max_episode_reward']: .4f}, "
              f"average reward: {log['avg_reward']: .4f}")

        # record reward information
        with writer.as_default():
            tf.summary.scalar("total reward", log['total_reward'], step=i_iter)
            tf.summary.scalar("average reward", log['avg_reward'], step=i_iter)
            tf.summary.scalar(
                "min reward", log['min_episode_reward'], step=i_iter)
            tf.summary.scalar(
                "max reward", log['max_episode_reward'], step=i_iter)
            tf.summary.scalar("num steps", log['num_steps'], step=i_iter)

    def update(self, batch, global_steps):
        batch_state = NDOUBLE(batch.state)
        batch_action = NLONG(batch.action)
        batch_reward = NDOUBLE(batch.reward)
        batch_next_state = NDOUBLE(batch.next_state)
        batch_mask = NDOUBLE(batch.mask)

        alg_step_stats = doubledqn_step(self.value_net, self.optimizer, self.value_net_target, batch_state, batch_action,
                                        batch_reward, batch_next_state, batch_mask, self.gamma, self.polyak,
                                        global_steps % self.update_target_gap == 0)

    def save(self, save_path):
        """save model"""
        check_path(save_path)
        pickle.dump((self.running_state),
                    open('{}/{}_double_dqn.p'.format(save_path, self.env_id), 'wb'))
        self.value_net.save_weights(
            '{}/{}_double_dqn_tf2'.format(save_path, self.env_id))
