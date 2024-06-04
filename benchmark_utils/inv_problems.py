import deepinv as dinv



def define_physics(inv_problem, noise_model, **kwargs):
    if noise_model == "gaussian":
        noise = dinv.physics.GaussianNoise(sigma=kwargs['sigma'])

    if noise_model == "poisson":
        noise = dinv.physics.PoissonNoise(gain=kwargs['gain'])

    if inv_problem == "denoising":

        physics = dinv.physics.Denoising(noise=noise)
    
    if inv_problem == "gaussian_deblurring":
        physics = dinv.physics.BlurFFT(
            kwargs['img_size'],
            filter = dinv.physics.blur.gaussian_blur(sigma=kwargs['blur_sd']),
            noise_model = noise
        )
    
    if inv_problem == "inpainting":
        physics = dinv.physics.Inpainting(
            kwargs['img_size'],
            mask = kwargs['prop_inpaint'],
            noise_model = noise
        )
    
    if inv_problem == "super_resolution":
        physics = dinv.physics.Downsampling(
            img_size=kwargs['img_size'],
            noise_model = noise,
            padding='circular'
        )
 


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
