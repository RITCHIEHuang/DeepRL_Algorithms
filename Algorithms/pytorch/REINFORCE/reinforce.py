#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午10:30
import pickle

import torch
import torch.optim as optim

from Algorithms.pytorch.Models.Policy import Policy
from Algorithms.pytorch.Models.Policy_discontinuous import DiscretePolicy
from Algorithms.pytorch.REINFORCE.reinforce_step import reinforce_step
from Common.MemoryCollector import MemoryCollector
from Utils.env_util import get_env_info
from Utils.file_util import check_path
from Utils.torch_util import FLOAT, device
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

        # seeding
        torch.manual_seed(self.seed)
        self.env.seed(self.seed)

        if env_continuous:
            self.policy_net = Policy(num_states, num_actions).to(
                device)  # current policy
        else:
            self.policy_net = DiscretePolicy(
                num_states, num_actions).to(device)

        self.running_state = ZFilter((num_states,), clip=5)

        if self.model_path:
            print("Loading Saved Model {}_reinforce.p".format(self.env_id))
            self.policy_net, self.running_state = pickle.load(
                open('{}/{}_reinforce.p'.format(self.model_path, self.env_id), "rb"))

        self.collector = MemoryCollector(self.env, self.policy_net, render=self.render,
                                         running_state=self.running_state,
                                         num_process=self.num_process)

        self.optimizer_p = optim.Adam(
            self.policy_net.parameters(), lr=self.lr_p)

    def choose_action(self, state):
        """select action"""
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action_log_prob(state)
    
        action = action.cpu().numpy()[0]
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
        writer.add_scalar("total reward", log['total_reward'], i_iter)
        writer.add_scalar("average reward", log['avg_reward'], i_iter)
        writer.add_scalar("min reward", log['min_episode_reward'], i_iter)
        writer.add_scalar("max reward", log['max_episode_reward'], i_iter)
        writer.add_scalar("num steps", log['num_steps'], i_iter)

        batch = memory.sample()  # sample all items in memory

        batch_state = FLOAT(batch.state).to(device)
        batch_action = FLOAT(batch.action).to(device)
        batch_reward = FLOAT(batch.reward).to(device)
        batch_mask = FLOAT(batch.mask).to(device)

        alg_step_stats = {}
        for _ in range(self.reinforce_epochs):
            alg_step_stats = reinforce_step(self.policy_net, self.optimizer_p, batch_state, batch_action, batch_reward,
                                            batch_mask,
                                            self.gamma)
        return alg_step_stats

    def save(self, save_path):
        """save model"""
        check_path(save_path)
        pickle.dump((self.policy_net, self.running_state),
                    open('{}/{}_reinforce.p'.format(save_path, self.env_id), 'wb'))
