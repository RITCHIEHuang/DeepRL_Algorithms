#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/5/9
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal

from Algorithms.pytorch.Models.BasePolicy import BasePolicy
from Algorithms.pytorch.Distributions.MultiOneHotCategorical import (
    MultiOneHotCategorical,
)
from Algorithms.pytorch.Distributions.MultiSoftMax import MultiSoftMax
from Utils.torch_util import resolve_activate_function


def init_weight(m, gain=1):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)


class Policy(BasePolicy):
    """
    Deal with comprehensive policy with discrete action and continuous action by configuration file,
    If both discrete and continuous action are provide, place discrete action on the left side for simplicity
    """

    def __init__(self, config=None):
        assert config, "Expected config not none!!!!"

        super().__init__(config["dim_state"], config["dim_action"])

        self.dim_hidden = config["dim_hidden"]
        self.activation = resolve_activate_function(config["activation"])
        self.dim_disc_action = config["dim_disc_action"]
        self.action_log_std = config["action_log_std"]
        self.disc_action_sections = config["disc_action_sections"]  # tuple
        self.use_multivariate_distribution = config[
            "use_multivariate_distribution"
        ]

        self.common = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_hidden),
            self.activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            self.activation(),
        )
        self.common.apply(partial(init_weight, gain=np.sqrt(2)))

        self.dim_cont_action = (
            self.dim_action - self.dim_disc_action
        )  # dimension of continuous action

        if self.dim_disc_action:
            self.disc_action = nn.Sequential(
                nn.Linear(self.dim_hidden, self.dim_disc_action),
                MultiSoftMax(
                    0, self.dim_disc_action, self.disc_action_sections
                ),
            )
            self.disc_action.apply(partial(init_weight, gain=np.sqrt(2)))
        self.cont_action_mean = nn.Linear(
            self.dim_hidden, self.dim_cont_action
        )  # mean of continuous action approximation
        self.cont_action_mean.apply(partial(init_weight, gain=0.01))
        self.cont_action_log_std = nn.Parameter(
            torch.ones(1, self.dim_cont_action) * self.action_log_std,
            requires_grad=True,
        )  # log_std of continuous action approximation

        # check disc actions
        assert (
            sum(self.disc_action_sections) == self.dim_disc_action
        ), "Wrong discrete action configuration !!!!"

    def forward(self, x):
        x = self.common(x)  # [batch_size, dim_disc_action + dim_cont_action]
        cont_mean = self.cont_action_mean(x)
        cont_log_std = self.cont_action_log_std.expand_as(cont_mean)
        cont_std = torch.exp(cont_log_std)

        if self.use_multivariate_distribution:
            dist_cont = MultivariateNormal(
                cont_mean, torch.diag_embed(cont_std)
            )
        else:
            dist_cont = Normal(cont_mean, cont_std)  # faster convergence

        dist_disc = None
        if self.dim_disc_action:
            disc_probs = self.disc_action(x)
            dist_disc = MultiOneHotCategorical(
                disc_probs, sections=self.disc_action_sections
            )
        return dist_disc, dist_cont

    def get_log_prob(self, state, action):
        dist_disc, dist_cont = self.forward(state)
        if self.use_multivariate_distribution:
            log_prob = dist_cont.log_prob(action[..., self.dim_disc_action :])
        else:
            log_prob = dist_cont.log_prob(
                action[..., self.dim_disc_action :]
            ).sum(dim=-1)
        if dist_disc:
            discrete_log_prob = dist_disc.log_prob(
                action[..., : self.dim_disc_action]
            )
            log_prob = log_prob + discrete_log_prob
        return log_prob.unsqueeze(-1)  # [batch_size, 1]

    def get_action_log_prob(self, states):
        dist_disc, dist_cont = self.forward(states)
        action = dist_cont.sample()
        if self.use_multivariate_distribution:
            log_prob = dist_cont.log_prob(
                action
            )  # use multivariate normal distribution
        else:
            log_prob = dist_cont.log_prob(action).sum(dim=-1)  # [batch_size]

        if dist_disc:
            discrete_action = dist_disc.sample()
            discrete_log_prob = dist_disc.log_prob(
                discrete_action
            )  # [batch_size]
            action = torch.cat([discrete_action, action], dim=-1)
            log_prob = log_prob + discrete_log_prob  # [batch_size]

        return action, log_prob.unsqueeze(-1)  # [batch_size, 1]

    def get_entropy(self, states):
        dist_discrete, dist_continuous = self.forward(states)
        ents = []
        if dist_discrete:
            ent_discrete = dist_discrete.entropy()
            ents.append(ent_discrete)
        if dist_continuous:
            ent_continuous = dist_continuous.entropy()
            ents.append(ent_continuous)
        ent = torch.cat(ents, dim=-1)
        return ent

    def get_kl(self, states):
        pass
