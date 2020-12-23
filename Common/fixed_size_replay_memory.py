import random
from Common.replay_memory import Memory, Transition


class FixedMemory(Memory):
    def __init__(self, size=None):
        super(FixedMemory, self).__init__()
        self._cur_pos = 0
        self._size = size

    # save item
    def push(self, *args):
        if len(self.memory) < self._size:
            self.memory.append(None)
        self.memory[self._cur_pos] = Transition(*args)
        self._cur_pos = (self._cur_pos + 1) % self._size

    def append(self, other):
        for memory in other.memory:
            if len(self.memory) < self._size:
                self.memory.append(None)
            self.memory[self._cur_pos] = memory
            self._cur_pos = (self._cur_pos + 1) % self._size

    # sample a mini_batch
    def sample(self, batch_size=None):
        # sample all transitions
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else: # sample with size: batch_size
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)