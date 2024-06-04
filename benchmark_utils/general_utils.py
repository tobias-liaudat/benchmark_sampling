
import torch
import deepinv as dinv


def get_best_device():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    return device


def compute_step_size(x_init, y, physics, likelihood, prior, scale_step, sigma_noise_lvl):

    # Get likelihood norm
    likelihood_term_norm = likelihood.norm
    # Get norm of the physics operator
    physics_norm = physics.compute_norm(x_init)
    # Full likelihood norm
    likelihood_lips = (physics_norm + likelihood_term_norm)

    # Compute prior lipschitz
    spectral_norm_op = dinv.loss.regularisers.JacobianSpectralNorm(
        max_iter=10, tol=1e-3, eval_mode=False, verbose=True
    )
    output = prior.grad(
        x_init.requires_grad_(), sigma_denoiser=sigma_noise_lvl
    ).requires_grad_()
    prior_lips = spectral_norm_op(output, x_init).detach()
    # We need to detach the variables after the gradient calculations in the
    # calculation of the spectral norm
    x_init = x_init.detach()
    y = y.detach()

    # Compute step size
    step_size = (
        scale_step / (likelihood_lips + prior_lips)
    ).detach()

    print("likelihood_lips: ", likelihood_lips)
    print("prior_lips: ", prior_lips)
    print("step_size: ", step_size)

    return step_size

