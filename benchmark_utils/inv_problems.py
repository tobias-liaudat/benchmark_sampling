import deepinv as dinv



def define_physics(inv_problem, noise_model, **kwargs):


    if noise_model == "gaussian":
        noise = dinv.physics.GaussianNoise(sigma=kwargs['sigma'])

    if noise_model == "poisson":
        noise = dinv.physics.Poisson(gain=kwargs['gain'])

    if inv_problem == "denoising":

        physics = dinv.physics.Denoising(noise=noise)


    return physics


def define_prior_model(prior_model, device, **kwargs):

    if prior_model == "dncnn_lipschitz_gray":
        prior = dinv.optim.ScorePrior(
            denoiser=dinv.models.DnCNN(
                pretrained="download_lipschitz",
                in_channels=1,
                out_channels=1,
                device=device
            )
        )
    else:
        raise NotImplementedError("Prior model {:s} not yet implemented.".format(prior_model))


    return prior 
