import torch.nn as nn


# Policy-network　结构
class PolicyNet(nn.Module):
    def __init__(self, num_states, num_actions):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states, 50),
            nn.ReLU(),
            nn.Linear(50, num_actions),
            nn.Softmax(dim=1)  # 输出每个action对应的概率
        )

        # 初始化网络参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)

    def forward(self, x):
        probs = self.net(x)
        return probs
