from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import deepinv as dinv
    import torch
    from benchmark_utils import inv_problems, general_utils, eval_tools


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Sampling PSNR"

    # URL of the main repo for this benchmark.
    url = "https://github.com/tobias-liaudat/benchmark_sampling"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        'prior_model' : ["dncnn_lipschitz_gray"],
    }

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = []

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self, x_true, y, physics):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.x_true, self.y = x_true, y
        self.physics = physics

    def evaluate_result(self, x_window):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.

        # Compute posterior mean
        x_post_mean = torch.mean(torch.stack(x_window, dim=0), dim=0)

        # Compute the PSNR over the batch
        psnr_mean = dinv.utils.metric.cal_psnr(
            x_post_mean, self.x_true, mean_batch=True, to_numpy=True
        ).item()

        # Compute acf and ess on the batch
        ess_slow, ess_med, ess_fast, lowest_median_acf, lowest_slow_acf, lowest_fast_acf = eval_tools.compute_acf_and_ess(x_window)

        return dict(
            PSNR_posterior_mean = psnr_mean,
            ESS_slow = ess_slow,
            ESS_med = ess_med,
            ESS = ess_fast,
            ACF_med = lowest_median_acf,
            ACF_slow = lowest_slow_acf, 
            ACF_fast = lowest_fast_acf,
            value=1,
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(x_window = self.x_true)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.

        device = general_utils.get_best_device()

        self.prior = inv_problems.define_prior_model(
            self.prior_model,
            device=device
        )

        self.likelihood = dinv.optim.L2(sigma=self.physics.noise_model.sigma)

        return dict(
            y=self.y,
            physics = self.physics,
            prior = self.prior,
            likelihood = self.likelihood
        )
