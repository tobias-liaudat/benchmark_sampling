import deepinv as dinv

from benchmark_utils import general_utils


def define_physics(inv_problem, noise_model, **kwargs):

    device = general_utils.get_best_device()

    if noise_model == "gaussian":
        noise = dinv.physics.GaussianNoise(sigma=kwargs["sigma"])
    elif noise_model == "poisson":
        noise = dinv.physics.PoissonNoise(gain=kwargs["gain"])
    else:
        raise NotImplementedError(
            "The noise model {:s} has not been implemented yet.".format(noise_model)
        )

    if inv_problem == "denoising":
        physics = dinv.physics.Denoising(noise=noise)

    elif inv_problem == "gaussian_deblurring":
        physics = dinv.physics.BlurFFT(
            kwargs["img_size"],
            filter=dinv.physics.blur.gaussian_blur(sigma=kwargs["blur_sd"]),
            noise_model=noise,
            device=device,
        )

    elif inv_problem == "motion_deblurring":
        psf_size = 31
        motion_generator = dinv.physics.generator.MotionBlurGenerator(
            (psf_size, psf_size), device=device)
        physics = dinv.physics.BlurFFT(
            kwargs["img_size"],
            filter = motion_generator.step(batch_size=1)['filter'],
            noise_model=noise,
            device=device,
        )

    elif inv_problem == "inpainting":
        physics = dinv.physics.Inpainting(
            kwargs["img_size"],
            mask=kwargs["prop_inpaint"],
            noise_model=noise,
            device=device,
        )

    elif inv_problem == "super_resolution":
        physics = dinv.physics.Downsampling(
            img_size=kwargs["img_size"],
            padding="circular",
            noise_model=noise,
            device=device,
        )

    else:
        raise NotImplementedError(
            "The inverse problem {:s} has not been implemented yet.".format(inv_problem)
        )

    return physics


def define_prior_model(prior_model, device, **kwargs):

    if prior_model == "dncnn_lipschitz_gray":
        prior = dinv.optim.ScorePrior(
            denoiser=dinv.models.DnCNN(
                pretrained="download_lipschitz",
                in_channels=1,
                out_channels=1,
                device=device,
            )
        )
    elif prior_model == "TV":
        prior = dinv.optim.ScorePrior(
            denoiser = dinv.models.TVDenoiser(
            ).to(device)
        )
    elif prior_model == "gsdrunet":
        # Specify the Denoising prior
        prior = GSPnP(
            denoiser=dinv.models.GSDRUNet(pretrained="download", train=False).to(device)
        )


    else:
        raise NotImplementedError(
            "Prior model {:s} not yet implemented.".format(prior_model)
        )

    return prior


class GSPnP(dinv.optim.prior.RED):
    r"""
    Gradient-Step Denoiser prior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def g(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return self.denoiser.potential(x, *args, **kwargs)


