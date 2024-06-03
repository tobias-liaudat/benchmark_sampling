import deepinv as dinv



def define_physics(inv_problem, noise_model, **kwargs):


    if noise_model == "gaussian":
        noise = dinv.physics.GaussianNoise(sigma=kwargs['sigma'])

    if noise_model == "poisson":
        noise = dinv.physics.Poisson(gain=kwargs['gain'])

    if inv_problem == "denoising":

        physics = dinv.physics.Denoising(noise=noise)


    return physics

