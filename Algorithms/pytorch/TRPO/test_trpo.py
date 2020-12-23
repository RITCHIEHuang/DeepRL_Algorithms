#!/usr/bin/env python
# Created at 2020/2/17
import unittest

import torch

from Algorithms.pytorch.TRPO.trpo_step import conjugate_gradient


def conjugate_gradients(Avp_f, b, nsteps=10, rdotr_tol=1e-10):
    x = torch.zeros(b.size(), device=b.device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x


class TRPOTest(unittest.TestCase):
    def setUp(self) -> None:
        self.A = torch.randn((4, 4))
        self.A_f = lambda x: self.A @ x
        self.b = torch.randn(4)

    def test_cg(self):
        x = conjugate_gradient(self.A_f, self.b)
        x2 = conjugate_gradients(self.A_f, self.b)
        print("A", self.A)
        print("b", self.b)
        print("x", x)
        print("x2", x2)
        print("Ax", self.A @ x)
        print("Ax2", self.A @ x2)
        print("A^(-1) b", self.A.inverse() @ self.b)


if __name__ == '__main__':
    unittest.main()
