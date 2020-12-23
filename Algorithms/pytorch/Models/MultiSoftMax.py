#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/16 下午5:25

from typing import Tuple

import torch
import torch.nn as nn


class MultiSoftMax(nn.Module):
    r"""customized module to deal with multiple softmax case.
    softmax feature: [dim_begin, dim_end)
    sections define sizes of each softmax case

    Examples::

        >>> m = MultiSoftMax(dim_begin=0, dim_end=5, sections=(2, 3))
        >>> input = torch.randn((2, 5))
        >>> output = m(input)
    """

    def __init__(self, dim_begin: int, dim_end: int, sections: Tuple = None):
        super().__init__()
        self.dim_begin = dim_begin
        self.dim_end = dim_end
        self.sections = sections

        if sections:
            assert dim_end - dim_begin == sum(sections), "expected same length of sections and customized" \
                                                         "dims"

    def forward(self, input_tensor: torch.Tensor):
        x = input_tensor[..., self.dim_begin:self.dim_end]
        res = input_tensor.clone()
        res[..., self.dim_begin:self.dim_end] = torch.cat([
            xx.softmax(dim=-1) for xx in torch.split(x, self.sections, dim=-1)], dim=-1)
        return res
