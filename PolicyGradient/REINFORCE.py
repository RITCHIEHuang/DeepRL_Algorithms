import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

if __name__ == '__main__':
    x = torch.linspace(0, 10, 100)
    y = x.exp() + torch.rand(x.numel())
    print(x, y)