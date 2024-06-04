import deepinv as dinv



def define_physics(inv_problem, noise_model, **kwargs):


    if noise_model == "gaussian":
        noise = dinv.physics.GaussianNoise(sigma=kwargs['sigma'])

    if noise_model == "poisson":
        noise = dinv.physics.Poisson(gain=kwargs['gain'])

    if inv_problem == "denoising":

        physics = dinv.physics.Denoising(noise=noise)
    
    if inv_problem == "deblurring":
        physics = dinv.physics.BlurFFT(
            (1,180,180),
            filter = dinv.physics.blur.gaussian_blur(sigma=(3, 3)),
            noise_model = noise
        )


    return physics


def define_prior_model(prior_model, **kwargs):

    if prior_model == "dncnn_lipschitz_gray":
        prior = dinv.optim.ScorePrior(
            denoiser=dinv.models.DnCNN(
                pretrained="download_lipschitz",
                in_channels=1,
                out_channels=1,
                device="cpu"
            )
        )
    else:
        raise NotImplementedError("Prior model {:s} not yet implemented.".format(prior_model))


    return prior 
