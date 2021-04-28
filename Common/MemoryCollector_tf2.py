#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/19 下午2:47
import math
import multiprocessing
import time

import numpy as np
import tensorflow as tf
from Common.replay_memory import Memory
from Utils.tf2_util import NDOUBLE, TDOUBLE


def collect_samples(
    pid, queue, env, policy, render, running_state, min_batch_size
):
    log = dict()
    memory = Memory()
    num_steps = 0
    num_episodes = 0

    min_episode_reward = float("inf")
    max_episode_reward = float("-inf")
    total_reward = 0

    while num_steps < min_batch_size:
        state = env.reset()
        episode_reward = 0
        if running_state:
            state = running_state(state)

        for t in range(10000):
            if render:
                env.render()

            state_tensor = tf.expand_dims(
                tf.convert_to_tensor(state, dtype=TDOUBLE), axis=0
            )
            action, log_prob = policy.get_action_log_prob(state_tensor)
            action = action.numpy()[0]
            log_prob = log_prob.numpy()[0]
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            if running_state:
                next_state = running_state(next_state)

            mask = 0 if done else 1
            # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
            memory.push(state, action, reward, next_state, mask, log_prob)
            num_steps += 1

            if done or num_steps >= min_batch_size:
                break

            state = next_state

        # num_steps += (t + 1)
        num_episodes += 1
        total_reward += episode_reward
        min_episode_reward = min(episode_reward, min_episode_reward)
        max_episode_reward = max(episode_reward, max_episode_reward)

    log["num_steps"] = num_steps
    log["num_episodes"] = num_episodes
    log["total_reward"] = total_reward
    log["avg_reward"] = total_reward / num_episodes
    log["max_episode_reward"] = max_episode_reward
    log["min_episode_reward"] = min_episode_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log["total_reward"] = sum([x["total_reward"] for x in log_list])
    log["num_episodes"] = sum([x["num_episodes"] for x in log_list])
    log["num_steps"] = sum([x["num_steps"] for x in log_list])
    log["avg_reward"] = log["total_reward"] / log["num_episodes"]
    log["max_episode_reward"] = max(
        [x["max_episode_reward"] for x in log_list]
    )
    log["min_episode_reward"] = min(
        [x["min_episode_reward"] for x in log_list]
    )

    return log


class MemoryCollector:
    def __init__(
        self, env, policy, render=False, running_state=None, num_process=1
    ):
        self.env = env
        self.policy = policy
        self.running_state = running_state
        self.render = render
        self.num_process = num_process

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        process_batch_size = int(math.floor(min_batch_size / self.num_process))
        queue = multiprocessing.Queue()
        workers = []

        # don't render other parallel processes
        for i in range(self.num_process - 1):
            worker_args = (
                i + 1,
                queue,
                self.env,
                self.policy,
                False,
                self.running_state,
                process_batch_size,
            )
            workers.append(
                multiprocessing.Process(
                    target=collect_samples, args=worker_args
                )
            )

        for worker in workers:
            worker.start()

        memory, log = collect_samples(
            0,
            None,
            self.env,
            self.policy,
            self.render,
            self.running_state,
            process_batch_size,
        )

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)

        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log

        # concat all memories
        for worker_memory in worker_memories:
            memory.append(worker_memory)

        if self.num_process > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)

        t_end = time.time()
        log["sample_time"] = t_end - t_start

        return memory, log
