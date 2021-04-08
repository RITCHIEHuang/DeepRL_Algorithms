#!/usr/bin/env python
# Created at 2020/5/9

import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Algorithms.pytorch.GAIL.dataset.expert_dataset import ExpertDataset      
from Algorithms.pytorch.Models.ConfigPolicy import Policy
from Algorithms.pytorch.Models.Discriminator import Discriminator
from Algorithms.pytorch.Models.Value import Value
from Algorithms.pytorch.PPO.ppo_step import ppo_step
from Common.GAE import estimate_advantages
from Common.MemoryCollector import MemoryCollector
from Utils.env_util import get_env_info
from Utils.file_util import check_path
from Utils.torch_util import FLOAT, to_device, device, resolve_activate_function


class GAIL:
    def __init__(self,
                 render=False,
                 num_process=4,
                 config=None,
                 expert_data_path=None,
                 env_id=None):

        self.render = render
        self.env_id = env_id
        self.num_process = num_process
        self.expert_data_path = expert_data_path
        self.config = config

        self._load_expert_trajectory()
        self._init_model()

    def _load_expert_trajectory(self):
        self.expert_dataset = ExpertDataset(expert_data_path=self.expert_data_path,
                                            train_fraction=self.config["expert_data"]["train_fraction"],
                                            traj_limitation=self.config["expert_data"]["traj_limitation"],
                                            shuffle=self.config["expert_data"]["shuffle"],
                                            batch_size=self.config["expert_data"]["batch_size"])

    def _init_model(self):
        # seeding
        seed = self.config["train"]["general"]["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env, env_continuous, num_states, num_actions = get_env_info(
            self.env_id)

        # check env
        assert num_states == self.expert_dataset.num_states and num_actions == self.expert_dataset.num_actions, \
            "Expected corresponding expert dataset and env"

        dim_dict = {
            "dim_state": num_states,
            "dim_action": num_actions
        }

        self.config["value"].update(dim_dict)
        self.config["policy"].update(dim_dict)
        self.config["discriminator"].update(dim_dict)

        self.value = Value(dim_state=self.config["value"]["dim_state"],
                           dim_hidden=self.config["value"]["dim_hidden"],
                           activation=resolve_activate_function(
                               self.config["value"]["activation"])
                           )
        self.policy = Policy(config=self.config["policy"])

        self.discriminator = Discriminator(dim_state=self.config["discriminator"]["dim_state"],
                                           dim_action=self.config["discriminator"]["dim_action"],
                                           dim_hidden=self.config["discriminator"]["dim_hidden"],
                                           activation=resolve_activate_function(
                                               self.config["discriminator"]["activation"])
                                           )

        self.discriminator_func = nn.BCELoss()
        self.running_state = None

        self.collector = MemoryCollector(self.env, self.policy, render=self.render,
                                         running_state=self.running_state,
                                         num_process=self.num_process)

        print("Model Structure")
        print(self.policy)
        print(self.value)
        print(self.discriminator)
        print()

        self.optimizer_policy = optim.Adam(
            self.policy.parameters(), lr=self.config["policy"]["learning_rate"])
        self.optimizer_value = optim.Adam(
            self.value.parameters(), lr=self.config["value"]["learning_rate"])
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(),
                                                  lr=self.config["discriminator"]["learning_rate"])

        to_device(self.value, self.policy,
                  self.discriminator, self.discriminator_func)

    def choose_action(self, state):
        """select action"""
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = self.policy.get_action_log_prob(state)
        return action, log_prob

    def learn(self, writer, i_iter):
        memory, log = self.collector.collect_samples(
            self.config["train"]["generator"]["sample_batch_size"])

        self.policy.train()
        self.value.train()
        self.discriminator.train()

        print(f"Iter: {i_iter}, num steps: {log['num_steps']}, total reward: {log['total_reward']: .4f}, "
              f"min reward: {log['min_episode_reward']: .4f}, max reward: {log['max_episode_reward']: .4f}, "
              f"average reward: {log['avg_reward']: .4f}, sample time: {log['sample_time']: .4f}")

        # record reward information
        writer.add_scalar("gail/average reward", log['avg_reward'], i_iter)
        writer.add_scalar("gail/num steps", log['num_steps'], i_iter)

        # collect generated batch
        # gen_batch = self.collect_samples(self.config["ppo"]["sample_batch_size"])
        gen_batch = memory.sample()
        gen_batch_state = FLOAT(gen_batch.state).to(
            device)  # [batch size, state size]
        gen_batch_action = FLOAT(gen_batch.action).to(
            device)  # [batch size, action size]
        gen_batch_old_log_prob = FLOAT(
            gen_batch.log_prob).to(device)  # [batch size, 1]
        gen_batch_mask = FLOAT(gen_batch.mask).to(device)  # [batch, 1]

        ####################################################
        # update discriminator
        ####################################################
        d_optim_i_iters = self.config["train"]["discriminator"]["optim_step"]
        if i_iter % d_optim_i_iters == 0:
            for step, (expert_batch_state, expert_batch_action) in enumerate(self.expert_dataset.train_loader):
                if step >= d_optim_i_iters:
                    break
                # calculate probs and logits
                gen_prob, gen_logits = self.discriminator(
                    gen_batch_state, gen_batch_action)
                expert_prob, expert_logits = self.discriminator(expert_batch_state.to(device),
                                                                expert_batch_action.to(device))

                # calculate accuracy
                gen_acc = torch.mean((gen_prob < 0.5).float())
                expert_acc = torch.mean((expert_prob > 0.5).float())

                # calculate regression loss
                expert_labels = torch.ones_like(expert_prob)
                gen_labels = torch.zeros_like(gen_prob)
                e_loss = self.discriminator_func(
                    expert_prob, target=expert_labels)
                g_loss = self.discriminator_func(gen_prob, target=gen_labels)
                d_loss = e_loss + g_loss

                # calculate entropy loss
                logits = torch.cat([gen_logits, expert_logits], 0)
                entropy = ((1. - torch.sigmoid(logits)) * logits -
                           torch.nn.functional.logsigmoid(logits)).mean()
                entropy_loss = - \
                    self.config["train"]["discriminator"]["ent_coeff"] * entropy

                total_loss = d_loss + entropy_loss

                self.optimizer_discriminator.zero_grad()
                total_loss.backward()
                self.optimizer_discriminator.step()

        writer.add_scalar('discriminator/d_loss', d_loss.item(), i_iter)
        writer.add_scalar("discriminator/e_loss", e_loss.item(), i_iter)
        writer.add_scalar("discriminator/g_loss", g_loss.item(), i_iter)
        writer.add_scalar("discriminator/ent", entropy.item(), i_iter)
        writer.add_scalar('discriminator/expert_acc', gen_acc.item(), i_iter)
        writer.add_scalar('discriminator/gen_acc', expert_acc.item(), i_iter)

        ####################################################
        # update policy by ppo [mini_batch]
        ####################################################

        with torch.no_grad():
            gen_batch_value = self.value(gen_batch_state)
            d_out, _ = self.discriminator(gen_batch_state, gen_batch_action)
            gen_batch_reward = -torch.log(1 - d_out + 1e-6)

        gen_batch_advantage, gen_batch_return = estimate_advantages(gen_batch_reward, gen_batch_mask,
                                                                    gen_batch_value,
                                                                    self.config["train"]["generator"]["gamma"],
                                                                    self.config["train"]["generator"]["tau"])

        ppo_optim_i_iters = self.config["train"]["generator"]["optim_step"]
        ppo_mini_batch_size = self.config["train"]["generator"]["mini_batch_size"]

        for _ in range(ppo_optim_i_iters):
            if ppo_mini_batch_size > 0:
                gen_batch_size = gen_batch_state.shape[0]
                optim_iter_num = int(
                    math.ceil(gen_batch_size / ppo_mini_batch_size))
                perm = torch.randperm(gen_batch_size)

                for i in range(optim_iter_num):
                    ind = perm[slice(i * ppo_mini_batch_size,
                                     min((i + 1) * ppo_mini_batch_size, gen_batch_size))]
                    mini_batch_state, mini_batch_action, mini_batch_advantage, mini_batch_return, \
                        mini_batch_old_log_prob = gen_batch_state[ind], gen_batch_action[ind], \
                        gen_batch_advantage[ind], gen_batch_return[ind], gen_batch_old_log_prob[
                            ind]

                    v_loss, p_loss, ent_loss = ppo_step(policy_net=self.policy,
                                                        value_net=self.value,
                                                        optimizer_policy=self.optimizer_policy,
                                                        optimizer_value=self.optimizer_value,
                                                        optim_value_iternum=self.config["value"]["optim_value_iter"],
                                                        states=mini_batch_state,
                                                        actions=mini_batch_action,
                                                        returns=mini_batch_return,
                                                        old_log_probs=mini_batch_old_log_prob,
                                                        advantages=mini_batch_advantage,
                                                        clip_epsilon=self.config["train"]["generator"]["clip_ratio"],
                                                        l2_reg=self.config["value"]["l2_reg"])
            else:
                v_loss, p_loss, ent_loss = ppo_step(policy_net=self.policy,
                                                    value_net=self.value,
                                                    optimizer_policy=self.optimizer_policy,
                                                    optimizer_value=self.optimizer_value,
                                                    optim_value_iternum=self.config["value"]["optim_value_iter"],
                                                    states=gen_batch_state,
                                                    actions=gen_batch_action,
                                                    returns=gen_batch_return,
                                                    old_log_probs=gen_batch_old_log_prob,
                                                    advantages=gen_batch_advantage,
                                                    clip_epsilon=self.config["train"]["generator"]["clip_ratio"],
                                                    l2_reg=self.config["value"]["l2_reg"])

        writer.add_scalar('generator/p_loss', p_loss, i_iter)
        writer.add_scalar('generator/v_loss', v_loss, i_iter)
        writer.add_scalar('generator/ent_loss', ent_loss, i_iter)

        print(f" Training episode:{i_iter} ".center(80, "#"))
        print('d_gen_prob:', gen_prob.mean().item())
        print('d_expert_prob:', expert_prob.mean().item())
        print('d_loss:', d_loss.item())
        print('e_loss:', e_loss.item())
        print("d/bernoulli_entropy:", entropy.item())

    def eval(self, i_iter, render=False):
        self.policy.eval()
        self.value.eval()
        self.discriminator.eval()

        state = self.env.reset()
        test_reward = 0
        while True:
            if render:
                self.env.render()
            if self.running_state:
                state = self.running_state(state)
            action, _ = self.choose_action(state)
            action = action.cpu().numpy()[0]
            state, reward, done, _ = self.env.step(action)

            test_reward += reward
            if done:
                break
        print(f"Iter: {i_iter}, test Reward: {test_reward}")
        self.env.close()

    def save_model(self, save_path):
        check_path(save_path)
        # torch.save((self.discriminator, self.policy, self.value), f"{save_path}/{self.exp_name}.pt")
        torch.save(self.discriminator,
                   f"{save_path}/{self.env_id}_Discriminator.pt")
        torch.save(self.policy, f"{save_path}/{self.env_id}_Policy.pt")
        torch.save(self.value, f"{save_path}/{self.env_id}_Value.pt")

    def load_model(self, model_path):
        # load entire model
        # self.discriminator, self.policy, self.value = torch.load(model_path, map_location=device)
        self.discriminator = torch.load(
            f"{model_path}_Discriminator.pt", map_location=device)
        self.policy = torch.load(
            f"{model_path}_Policy.pt", map_location=device)
        self.value = torch.load(f"{model_path}_Value.pt", map_location=device)
