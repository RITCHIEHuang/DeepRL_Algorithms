from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


# 经验重放策略: 提升DQN的稳定性
class MemoryReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # 保存一条转换记录
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # 采样
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
