from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import NoCriterion

import os

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    # import your reusable functions here
    import deepinv as dinv
    from benchmark_utils import general_utils


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = "pnp-ula"
    stopping_criterion = NoCriterion(strategy="callback")

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "scale_step": [0.99],
        "burnin": [100],
        "stats_window_length": [20],
        "thinning_step": [10],
        "iterations": [100],
        "alpha": [1.],
        "eta": [0.05],
        "inner_iter": [10]
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []

    def get_next(self, stop_val):
        return stop_val + 1

    def set_objective(self, y, physics, prior, likelihood):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.y, self.physics, self.prior, self.likelihood = (
            y,
            physics,
            prior,
            likelihood,
        )

    def run(self, callback):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        # Get initial x
        x_init = self.physics.A_adjoint(self.y)

        sigma_noise_lvl = 0.1#self.physics.noise_model.sigma

        # Compute automatically the step size taking into account the Lipschitz constants
        step_size = general_utils.compute_step_size(
            x_init=x_init.clone(),
            y=self.y.clone(),
            physics=self.physics,
            likelihood=self.likelihood,
            prior=self.prior,
            scale_step=self.scale_step,
            sigma_noise_lvl=sigma_noise_lvl,
        )

        # Define the sampler
        sampler = dinv.sampling.langevin.inner_iter(
            step_size, self.alpha, self.inner_iter, self.eta, sigma_noise_lvl
        )

        # Initialise the chain with a burnin period
        burnin_x = x_init
        for i_ in range(self.burnin):
            burnin_x = sampler.forward(
                burnin_x, self.y, self.physics, self.likelihood, self.prior
            )
            burnin_x = burnin_x.clamp(min=-1.,max=2.)
       
        # Initialise the empty list
        self.x_window = []
        temp = burnin_x
        # Fill the list with the length of the window
        for it in range(self.stats_window_length):
            for k_ in range(self.thinning_step):
                temp = sampler.forward(
                    temp, self.y, self.physics, self.likelihood, self.prior
                )
                temp = temp.clamp(min=-1.,max=2.)

            self.x_window.append(temp)

        # Now that the window is full carry out the benchmark
        while callback():
            # Draw a new sample
            temp = self.x_window[-1]
            # Remove the last added sample
            _ = self.x_window.pop(0)
            for k_ in range(self.thinning_step):
                temp = sampler.forward(
                    temp, self.y, self.physics, self.likelihood, self.prior
                )
                temp = temp.clamp(min=-1.,max=2.)
            # Add the sample to the list
            self.x_window.append(temp)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(x_window=self.x_window)
