import tensorflow as tf
import numpy as np

from Utils.tf2_util import flatgrad, get_flat, set_from_flat


def conjugate_gradient(
    f_ax, b_vec, cg_iters=10, callback=None, residual_tol=1e-10
):
    """
    conjugate gradient calculation (Ax = b), bases on
    https://epubs.siam.org/doi/book/10.1137/1.9781611971446 Demmel p 312
    :param f_ax: (function) The function describing the Matrix A dot the vector x
                 (x being the input parameter of the function)
    :param b_vec: (numpy float) vector b, where Ax = b
    :param cg_iters: (int) the maximum number of iterations for converging
    :param callback: (function) callback the values of x while converging
    :param verbose: (bool) print extra information
    :param residual_tol: (float) the break point if the residual is below this value
    :return: (numpy float) vector x, where Ax = b
    """
    first_basis_vect = tf.identity(b_vec)  # the first basis vector
    residual = -tf.identity(b_vec)  # the residual
    x_var = np.zeros_like(b_vec)  # vector x, where Ax = b
    residual_dot_residual = tf.reduce_sum(
        residual * residual
    )  # L2 norm of the residual

    fmt_str = "CG %10i %10.3g %10.3g"
    title_str = "CG %10s %10s %10s"

    print(title_str % ("iter", "residual norm", "soln norm"))
    for i in range(cg_iters):
        if callback is not None:
            callback(x_var)

        print(fmt_str % (i, residual_dot_residual, tf.norm(x_var)))

        z_var = f_ax(first_basis_vect)
        v_var = residual_dot_residual / tf.reduce_sum(first_basis_vect * z_var)
        x_var += v_var * first_basis_vect
        residual += v_var * z_var
        new_residual_dot_residual = tf.reduce_sum(residual * residual)
        mu_val = new_residual_dot_residual / residual_dot_residual
        first_basis_vect = -residual + mu_val * first_basis_vect

        residual_dot_residual = new_residual_dot_residual
        if residual_dot_residual < residual_tol:
            break

    if callback is not None:
        callback(x_var)

    print(fmt_str % (i + 1, residual_dot_residual, tf.norm(x_var)))
    return x_var


def trpo_step(
    policy_net,
    value_net,
    opt_v,
    states,
    actions,
    old_log_probs,
    gae,
    returns,
    max_kl=0.01,
    cg_iters=10,
    ent_coeff=0.0,
    cg_damping=1e-2,
    vf_iters=3,
):
    """update critc"""
    for _ in range(vf_iters):
        with tf.GradientTape() as tape:
            values = value_net(states)
            value_loss = tf.reduce_mean(
                tf.square(values - tf.stop_gradient(returns))
            )

        grads = tape.gradient(value_loss, value_net.trainable_variables)
        opt_v.apply_gradients(zip(grads, value_net.trainable_variables))

    """update policy"""

    def get_losses():
        log_probs = tf.expand_dims(
            policy_net.get_log_prob(states, actions), axis=-1
        )
        ent = tf.reduce_mean(policy_net.get_entropy(states))
        ratio = tf.exp(log_probs - tf.stop_gradient(old_log_probs))
        kl = policy_net.get_kl(states)
        # kl = -log_probs + tf.reduce_mean(old_log_probs)
        mean_kl = tf.reduce_mean(kl)
        surr_gain = tf.reduce_mean(ratio * gae)
        mean_ent = tf.reduce_mean(ent)
        ent_bonus = ent_coeff * mean_ent
        optim_gain = surr_gain + ent_bonus

        return [optim_gain, mean_kl, ent_bonus, surr_gain, mean_ent]

    with tf.GradientTape() as tape:
        optim_gain, mean_kl, ent_bonus, surr_gain, mean_ent = get_losses()

    var_list = policy_net.trainable_variables
    grads = tape.gradient(optim_gain, var_list)
    flat_grads = get_flat(grads)

    def fisher_vector_product(vec):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                (
                    optim_gain,
                    mean_kl,
                    ent_bonus,
                    surr_gain,
                    mean_ent,
                ) = get_losses()

            klgrads = t1.gradient(mean_kl, var_list)
            flatten_kl_grads = get_flat(klgrads)
            gvp = tf.reduce_sum(flatten_kl_grads * vec)
        gvp_grads = t2.gradient(gvp, var_list)
        fvp = flatgrad(gvp_grads, var_list)
        return fvp + cg_damping * vec

    """conjugate gradient -> stepdir"""
    stepdir = conjugate_gradient(
        fisher_vector_product, flat_grads, cg_iters=cg_iters
    )
    shs = 0.5 * tf.reduce_sum(stepdir * fisher_vector_product(stepdir))
    # abs(shs) to avoid taking square root of negative values
    lagrange_multiplier = np.sqrt(abs(shs) / max_kl)
    fullstep = stepdir / lagrange_multiplier
    expectedimprove = tf.reduce_sum(flat_grads * fullstep)

    surrbefore = get_losses()[0]

    """line search -> stepsize"""
    stepsize = 1.0
    thbefore = get_flat(var_list)
    for _ in range(10):
        thnew = thbefore + fullstep * stepsize
        set_from_flat(policy_net, thnew)
        mean_losses = (
            surr,
            kl_loss,
            ent_bonus,
            surr_gain,
            mean_ent,
        ) = get_losses()
        improve = surr - surrbefore
        print("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
        if not np.isfinite(mean_losses).all():
            print("Got non-finite value of losses -- bad!")
        elif kl_loss > max_kl * 1.5:
            print("violated KL constraint. shrinking step.")
        elif improve < 0:
            print("surrogate didn't improve. shrinking step.")
        else:
            print("Stepsize OK!")
            break
        stepsize *= 0.5
    else:
        print("couldn't compute a good step")
        set_from_flat(policy_net, thbefore)

    log_metrics = {
        "critic_loss": value_loss,
        "gae": tf.reduce_mean(gae),
        "ent_loss": ent_bonus,
        "optim_gain": optim_gain,
        "mean_kl": mean_kl,
        "surr_gain": surr_gain,
        "entropy": mean_ent,
    }

    return log_metrics
