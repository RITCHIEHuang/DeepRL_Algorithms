#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/3/23
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as optim

from Common.MemoryCollector_tf2 import MemoryCollector
from Algorithms.tf2.Models.Policy import Policy
from Algorithms.tf2.Models.Policy_discontinuous import DiscretePolicy
from Algorithms.tf2.REINFORCE.reinforce_step import reinforce_step
from Utils.env_util import get_env_info
from Utils.file_util import check_path
from Utils.tf2_util import NDOUBLE
from Utils.zfilter import ZFilter


class REINFORCE:
    def __init__(self,
                 env_id,
                 render=False,
                 num_process=1,
                 min_batch_size=2048,
                 lr_p=3e-4,
                 gamma=0.99,
                 reinforce_epochs=5,
                 seed=1,
                 model_path=None
                 ):
        self.env_id = env_id
        self.render = render
        self.num_process = num_process
        self.min_batch_size = min_batch_size
        self.lr_p = lr_p
        self.gamma = gamma
        self.reinforce_epochs = reinforce_epochs
        self.model_path = model_path
        self.seed = seed

        self._init_model()

    def _init_model(self):
        """init model from parameters"""
        self.env, env_continuous, num_states, num_actions = get_env_info(
            self.env_id)
        tf.keras.backend.set_floatx('float64')
        # seeding
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.env.seed(self.seed)

        if env_continuous:
            self.policy_net = Policy(num_states, num_actions)  # current policy
        else:
            self.policy_net = DiscretePolicy(num_states, num_actions)

        self.running_state = ZFilter((num_states,), clip=5)

        if self.model_path:
            print("Loading Saved Model {}_reinforce_tf2.p".format(self.env_id))
            self.running_state = pickle.load(
                open('{}/{}_reinforce_tf2.p'.format(self.model_path, self.env_id), "rb"))
            self.policy_net.load_weights(
                "{}/{}_reinforce_tf2".format(self.model_path, self.env_id))

        self.collector = MemoryCollector(self.env, self.policy_net, render=self.render,
                                         running_state=self.running_state,
                                         num_process=self.num_process)

        self.optimizer_p = optim.Adam(lr=self.lr_p, clipnorm=20)

    def choose_action(self, state):
        """select action"""
        state = np.expand_dims(NDOUBLE(state), 0)
        action, log_prob = self.policy_net.get_action_log_prob(state)

        action = action.numpy()[0]
        return action

    def eval(self, i_iter, render=False):
        """init model from parameters"""
        state = self.env.reset()
        test_reward = 0
        while True:
            if render:
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
        """learn model"""
        memory, log = self.collector.collect_samples(self.min_batch_size)

        print(f"Iter: {i_iter}, num steps: {log['num_steps']}, total reward: {log['total_reward']: .4f}, "
              f"min reward: {log['min_episode_reward']: .4f}, max reward: {log['max_episode_reward']: .4f}, "
              f"average reward: {log['avg_reward']: .4f}, sample time: {log['sample_time']: .4f}")

        # record reward information
        with writer.as_default():
            tf.summary.scalar("total reward", log['total_reward'], i_iter)
            tf.summary.scalar("average reward", log['avg_reward'], i_iter)
            tf.summary.scalar("min reward", log['min_episode_reward'], i_iter)
            tf.summary.scalar("max reward", log['max_episode_reward'], i_iter)
            tf.summary.scalar("num steps", log['num_steps'], i_iter)

        batch = memory.sample()  # sample all items in memory

        batch_state = NDOUBLE(batch.state)
        batch_action = NDOUBLE(batch.action)
        batch_reward = NDOUBLE(batch.reward)
        batch_mask = NDOUBLE(batch.mask)

        log_stats = {}
        for _ in range(self.reinforce_epochs):
            log_stats = reinforce_step(self.policy_net, self.optimizer_p, batch_state, batch_action, batch_reward,
                                       batch_mask, self.gamma)
        with writer.as_default():
            tf.summary.scalar("policy loss", log_stats["policy_loss"], i_iter)
        return log_stats

    def save(self, save_path):
        """save model"""
        check_path(save_path)
        pickle.dump(self.running_state,
                    open('{}/{}_reinforce_tf2.p'.format(save_path, self.env_id), 'wb'))
        self.policy_net.save_weights(
            "{}/{}_reinforce_tf2".format(save_path, self.env_id))
