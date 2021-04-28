#!/usr/bin/env python
# Created at 2020/1/22
import numpy as np
import scipy.optimize as opt
import torch
import torch.autograd as autograd
import torch.nn as nn

from Utils.torch_util import (
    device,
    set_flat_params,
    get_flat_grad_params,
    get_flat_params,
    FLOAT,
)


def trpo_step(
    policy_net,
    value_net,
    states,
    actions,
    returns,
    advantages,
    old_log_probs,
    max_kl,
    damping,
    l2_reg,
    optimizer_value=None,
):
    """
    Update by TRPO algorithm
    """
    """update critic"""

    def value_objective_func(value_net_flat_params):
        """
        get value_net loss
        :param value_net_flat_params: numpy
        :return:
        """
        set_flat_params(value_net, FLOAT(value_net_flat_params))
        values_pred = value_net(states)
        value_loss = nn.MSELoss()(values_pred, returns)
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg

        objective_value_loss = value_loss.item()
        # print("Current value loss: ", objective_value_loss)
        return objective_value_loss

    def value_objective_grad_func(value_net_flat_params):
        """
        objective function for scipy optimizing
        """
        set_flat_params(value_net, FLOAT(value_net_flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(states)
        value_loss = nn.MSELoss()(values_pred, returns)
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg

        value_loss.backward()  # to get the grad
        objective_value_loss_grad = (
            get_flat_grad_params(value_net)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        return objective_value_loss_grad

    if optimizer_value is None:
        """
        update by scipy optimizing, for detail about L-BFGS-B: ref:
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
        """
        value_net_flat_params_old = (
            get_flat_params(value_net)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )  # initial guess
        res = opt.minimize(
            value_objective_func,
            value_net_flat_params_old,
            method="L-BFGS-B",
            jac=value_objective_grad_func,
            options={"maxiter": 30, "disp": False},
        )
        # print("Call L-BFGS-B, result: ", res)
        value_net_flat_params_new = res.x
        set_flat_params(value_net, FLOAT(value_net_flat_params_new))

    else:
        """
        update by gradient descent
        """
        for _ in range(10):
            values_pred = value_net(states)
            value_loss = nn.MSELoss()(values_pred, returns)
            # weight decay
            for param in value_net.parameters():
                value_loss += param.pow(2).sum() * l2_reg
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

    """update policy"""
    update_policy(
        policy_net, states, actions, old_log_probs, advantages, max_kl, damping
    )


def conjugate_gradient(Hvp_f, b, steps=10, rdotr_tol=1e-10):
    """
    reference <<Numerical Optimization>> Page 112
    :param Hvp_f: function Hvp_f(x) = A @ x
    :param b: equation
    :param steps: steps to run Conjugate Gradient Descent
    :param rdotr_tol: the threshold to stop algorithm
    :return: update direction
    """
    x = torch.zeros_like(b, device=device)  # initialization approximation of x
    r = -b.clone()  # Hvp(x) - b : residual
    p = b.clone()  # b - Hvp(x) : steepest descent direction
    rdotr = r.t() @ r  # r.T @ r
    for i in range(steps):
        Hvp = Hvp_f(p)  # A @ p
        alpha = rdotr / (p.t() @ Hvp)  # step length
        x += alpha * p  # update x
        r += alpha * Hvp  # new residual
        new_rdotr = r.t() @ r
        betta = new_rdotr / rdotr  # beta
        p = -r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:  # satisfy the threshold
            break
    return x


def line_search(
    model,
    f,
    x,
    step_dir,
    expected_improve,
    max_backtracks=10,
    accept_ratio=0.1,
):
    """
    max f(x) <=> min -f(x)
    line search sufficient condition: -f(x_new) <= -f(x) + -e coeff * step_dir
    perform line search method for choosing step size
    :param model:
    :param f:
    :param x:
    :param step_dir: direction to update model parameters
    :param expected_improve:
    :param max_backtracks:
    :param accept_ratio:
    :return:
    """
    f_val = f(False).item()

    for step_coefficient in [0.5 ** k for k in range(max_backtracks)]:
        x_new = x + step_coefficient * step_dir
        set_flat_params(model, x_new)
        f_val_new = f(False).item()
        actual_improve = f_val_new - f_val
        improve = expected_improve * step_coefficient
        ratio = actual_improve / improve
        if ratio > accept_ratio:
            return True, x_new
    return False, x


def update_policy(
    policy_net: nn.Module,
    states,
    actions,
    old_log_probs,
    advantages,
    max_kl,
    damping,
):
    def get_loss(grad=True):
        log_probs = policy_net.get_log_prob(states, actions)
        if not grad:
            log_probs = log_probs.detach()
        ratio = torch.exp(log_probs - old_log_probs)
        loss = (ratio * advantages).mean()
        return loss

    def Hvp(v):
        """
        compute vector product of second order derivative of KL_Divergence Hessian and v
        :param v: vector
        :return: \nabla \nabla H @ v
        """
        # compute kl divergence between current policy and old policy
        kl = policy_net.get_kl(states)
        kl = kl.mean()

        # first order gradient kl
        grads = torch.autograd.grad(
            kl, policy_net.parameters(), create_graph=True
        )
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()  # flag_grad_kl.T @ v
        # second order gradient of kl
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = torch.cat(
            [grad.contiguous().view(-1) for grad in grads]
        ).detach()

        return flat_grad_grad_kl + v * damping

    # compute first order approximation to Loss
    loss = get_loss()
    loss_grads = autograd.grad(loss, policy_net.parameters())
    loss_grad = torch.cat(
        [grad.view(-1) for grad in loss_grads]
    ).detach()  # g.T

    # conjugate gradient solve : Hx = g
    # apply vector product strategy here to compute Hx by `Hvp`
    # approximation solution of x'= H^(-1)g
    step_dir = conjugate_gradient(Hvp, loss_grad)
    # g.T H^(-1) g; another implementation: Hvp(step_dir) @ step_dir
    shs = Hvp(step_dir).t() @ step_dir
    lm = torch.sqrt(2 * max_kl / shs)
    step = lm * step_dir  # update direction for policy nets
    expected_improve = loss_grad.t() @ step

    """
    line search for step size 
    """
    current_flat_parameters = get_flat_params(policy_net)  # theta
    success, new_flat_parameters = line_search(
        policy_net,
        get_loss,
        current_flat_parameters,
        step,
        expected_improve,
        10,
    )
    set_flat_params(policy_net, new_flat_parameters)
    # success indicating whether TRPO works as expected
    return success
