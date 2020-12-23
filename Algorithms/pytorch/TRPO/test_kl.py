#!/usr/bin/env python
# Created at 2020/2/9
import unittest

import torch
from torch.distributions import Normal, kl_divergence


class Test_KL(unittest.TestCase):

    def setUp(self) -> None:
        self.mean1 = torch.randn(4, 2)
        self.log_std1 = torch.randn(4, 2)
        self.mean2 = torch.randn(4, 2)
        self.log_std2 = torch.randn(4, 2)

        self.dist_1 = Normal(self.mean1, torch.exp(self.log_std1))
        self.dist_2 = Normal(self.mean2, torch.exp(self.log_std2))

    def test_kl_simple(self):
        kl_torch = kl_divergence(self.dist_1, self.dist_2)
        kl_custom = self.log_std2 - self.log_std1 + (self.log_std1.exp().pow(2) + (self.mean1 - self.mean2).pow(2)) / (
                2 * self.log_std2.exp().pow(2)) - 1 / 2

        print(f"pytorch result: {kl_torch}, \nmath equation result: {kl_custom}")
        self.assertTrue(torch.all(kl_torch - kl_custom < torch.tensor(0.001)))


if __name__ == '__main__':
    unittest.main()
