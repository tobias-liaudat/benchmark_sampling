from benchopt import BaseSolver, safe_import_context
import deepinv as dinv
import torch

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    # import your reusable functions here
    from benchmark_utils import gradient_ols


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'pnp-ula'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'scale_step': [0.99],
        'burnin' : 100,
        'thinning_step' : 10,
      }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []

    def set_objective(self, X, y, physics, prior, likelihood):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X, self.y, self.physics, self.prior, self.likelihood = X, y, physics, prior, likelihood

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        #fixme
        L = np.linalg.norm(self.X, ord=2) ** 2
        step_size = self.scale_step / L

        #need access to prior and likelihood, noise level, physics

        # run sampler for burnin iterations the first time,
        # then for thinning_step iterations the next times
        iterations = self.parameters['thinning_step']
        f = dinv.sampling.ULA(
            prior=prior,
            data_fidelity=likelihood,
            max_iter=iterations,
            alpha=self.parameters['regularization'],
            step_size=step_size,
            verbose=True,
            sigma=sigma_denoiser,
        )
        self.beta = f(self.y, physics)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(beta=self.beta)
