from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import deepinv as dinv
    import torch
    from benchmark_utils import inv_problems, general_utils, eval_tools
    from torchvision.utils import save_image
    import os
    import datetime


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
        "prior_model": ["dncnn_lipschitz_gray"],
        "compute_PSNR": [True],
        "compute_lpips": [True],
        "compute_ssim": [False],
        "compute_acf_ess": [False],
        "compute_posterior_std_dev_pearson_cc" : [False],
        "compute_metric_last_sample": [True],
        "compute_metric_sample_means": [[1, 50]],
        "save_image": [True],
        "save_every_iter": [100],
    }

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = [
        "pytorch:pytorch",
        "numpy",
        "pip:deepinv",
        "pip:arviz",
        "pip:statsmodels",
        "pip:pyiqa",
    ]

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


        # Initialise results dictionary
        results_dict = dict(value=1)

        # Check the window size
        assert len(x_window) >= np.max(self.compute_metric_sample_means)

        # Compute posterior mean
        x_post_mean = torch.mean(torch.stack(x_window, dim=0), dim=0)

        if self.save_image:
            if self.it % self.save_every_iter == 0:
                save_image(
                    x_post_mean,
                    os.path.join(self.im_folder, "mean_iter_{}.png".format(self.it)),
                    nrow=2,
                )
                save_image(
                    x_window[-1],
                    os.path.join(self.im_folder, "sample_iter_{}.png".format(self.it)),
                    nrow=2,
                )
            self.it += 1

        # Iterate over the metrics over the posterior mean
        for metric, metric_name in zip(self.metrics_list, self.metrics_list_name):
            results_dict[metric_name + "_posterior_mean"] = metric(
                x_post_mean, self.x_true
            )

        if self.compute_metric_last_sample:
            # Get last sample and compute metrics
            x_last_sample = x_window[-1]
            # Iterate over the metrics over the last sample
            for metric, metric_name in zip(self.metrics_list, self.metrics_list_name):
                results_dict[metric_name + "_one_sample"] = metric(
                    x_last_sample, self.x_true
                )

        if 0 not in self.compute_metric_sample_means:
            # Compute over an average of the last samples
            for avrg_num in self.compute_metric_sample_means:
                # Compute posterior mean
                x_mean = torch.mean(torch.stack(x_window, dim=0)[-avrg_num:, :], dim=0)
                # Iterate over the metrics over the last sample
                for metric, metric_name in zip(
                    self.metrics_list, self.metrics_list_name
                ):
                    results_dict[metric_name + "_" + str(avrg_num) + "_samples"] = (
                        metric(x_mean, self.x_true)
                    )

        # Compute Pearson correlation coefficient between the posterior mean and the error
        if self.compute_posterior_std_dev_pearson_cc:
            a=1

        # Compute acf and ess on the batch
        if self.compute_acf_ess:
            (
                ess_slow,
                ess_med,
                ess_fast,
                lowest_median_acf,
                lowest_slow_acf,
                lowest_fast_acf,
            ) = eval_tools.compute_acf_and_ess(x_window)
            # Store results
            results_dict["ESS_slow"] = ess_slow
            results_dict["ESS_med"] = ess_med
            results_dict["ESS_fast"] = ess_fast
            results_dict["ACF_med"] = lowest_median_acf
            results_dict["ACF_slow"] = lowest_slow_acf
            results_dict["ACF_fast"] = lowest_fast_acf

        return results_dict

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(x_window=self.x_true)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.

        device = general_utils.get_best_device()

        if self.save_image:
            self.it = 0  # Image index for saving
            ## Create folder for images
            date = datetime.date.today()
            now = datetime.datetime.now()
            date = "{day}_{month}__{hr}_{mn}_{s}".format(
                day=date.day, month=date.month, hr=now.hour, mn=now.minute, s=now.second
            )
            self.im_folder = "./benchmark_sampling/output_ims/" + date
            if not os.path.exists(self.im_folder):
                os.makedirs(self.im_folder, exist_ok=True)

        self.prior = inv_problems.define_prior_model(self.prior_model, device=device)

        self.likelihood = dinv.optim.L2(sigma=self.physics.noise_model.sigma)

        # Define metrics list
        self.metrics_list = []
        self.metrics_list_name = []
        if self.compute_PSNR:
            psnr_calc = lambda x_est, x_true: dinv.utils.metric.cal_psnr(
                x_est, x_true, mean_batch=True, to_numpy=True
            ).item()
            self.metrics_list.append(psnr_calc)
            self.metrics_list_name.append("PSNR")

        if self.compute_lpips:
            self.lpips = dinv.loss.LPIPS(train=False, device=device)
            # We apply the mean over the set of images of the dataset
            lpips_calc = lambda x_est, x_true: torch.mean(
                self.lpips(x_true, x_est)
            ).item()
            self.metrics_list.append(lpips_calc)
            self.metrics_list_name.append("LPIPS")

        if self.compute_ssim:
            self.ssim = dinv.loss.SSIM(multiscale=False, train=False, device=device)
            ssim_calc = lambda x_est, x_true: torch.mean(
                self.ssim(x_true, x_est)
            ).item()
            self.metrics_list.append(ssim_calc)
            self.metrics_list_name.append("SSIM")

        return dict(
            y=self.y, physics=self.physics, prior=self.prior, likelihood=self.likelihood
        )
