#!/usr/bin/env python
# Created at 2020/3/31
import pickle

import torch
import torch.optim as optim

from Algorithms.pytorch.A2C.a2c_step import a2c_step
from Algorithms.pytorch.Models.Actor_Critic import Actor_Critic
from Algorithms.pytorch.Models.Policy import Policy
from Algorithms.pytorch.Models.Policy_discontinuous import DiscretePolicy
from Algorithms.pytorch.Models.Value import Value
from Common.GAE import estimate_advantages
from Common.MemoryCollector import MemoryCollector
from Utils.env_util import get_env_info
from Utils.file_util import check_path
from Utils.torch_util import FLOAT, device
from Utils.zfilter import ZFilter


class A2C():
    def __init__(self,
                 env_id,
                 render=False,
                 num_process=4,
                 min_batch_size=2048,
                 lr_ac=3e-4,
                 value_net_coeff=0.5,
                 entropy_coeff=1e-2,
                 gamma=0.99,
                 tau=0.95,
                 seed=1,
                 model_path=None
                 ):

        self.env_id = env_id
        self.gamma = gamma
        self.tau = tau
        self.render = render
        self.num_process = num_process
        self.lr_ac = lr_ac
        self.value_net_coeff = value_net_coeff
        self.entropy_coeff = entropy_coeff
        self.min_batch_size = min_batch_size
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
            self.policy_net = Policy(num_states, num_actions).to(device)
        else:
            self.policy_net = DiscretePolicy(
                num_states, num_actions).to(device)

        self.value_net = Value(num_states).to(device)

        self.ac_net = Actor_Critic(self.policy_net, self.value_net).to(device)

        self.running_state = ZFilter((num_states,), clip=5)

        if self.model_path:
            print("Loading Saved Model {}_a2c.p".format(self.env_id))
            self.ac_net, self.running_state = pickle.load(
                open('{}/{}_a2c.p'.format(self.model_path, self.env_id), "rb"))

        self.collector = MemoryCollector(self.env, self.ac_net, render=self.render,
                                         running_state=self.running_state,
                                         num_process=self.num_process)

        self.optimizer_ac = optim.Adam(self.ac_net.parameters(), lr=self.lr_ac)

    def choose_action(self, state):
        """select action"""
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = self.ac_net.get_action_log_prob(state)

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

        with torch.no_grad():
            batch_value = self.ac_net.get_value(batch_state)

        batch_advantage, batch_return = estimate_advantages(batch_reward, batch_mask, batch_value, self.gamma,
                                                            self.tau)

        alg_step_stats = a2c_step(self.ac_net, self.optimizer_ac, batch_state, batch_action, batch_return,
                                  batch_advantage, self.value_net_coeff, self.entropy_coeff)

        return alg_step_stats

    def save(self, save_path):
        """save model"""
        check_path(save_path)
        pickle.dump((self.ac_net, self.running_state),
                    open('{}/{}_a2c.p'.format(save_path, self.env_id), 'wb'))
