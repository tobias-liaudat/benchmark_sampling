from benchopt import BaseSolver, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    # import your reusable functions here
    import deepinv as dinv
    from deepinv.sampling.utils import Welford
    import torch


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'pnp-ula'
    sampling_strategy = "callback"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'scale_step': [0.99],
        'burnin' : [100],
        'iterations': [100],
        'thinning_step' : [10],
        'alpha' : [1],
        'eta' : [0.05],
        'inner_iter' : [10],
      }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []

    def set_objective(self, y, physics, prior, likelihood):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.y, self.physics, self.prior, self.likelihood = y, physics, prior, likelihood

    def run(self, callback):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html
    
        self.statistics = Welford(x=self.y)

        noise_lvl = self.physics.noise_model.sigma
        step_size = self.scale_step / noise_lvl**2
        
        sampler = dinv.sampling.langevin.SKRockIterator(
            step_size, self.alpha, self.eta, self.inner_iter, noise_lvl
        )

        burnin_x = self.y

        for i_ in range(self.burnin):
            burnin_x = sampler.forward(burnin_x, self.y, self.physics, self.likelihood, self.prior)

        self.x = [burnin_x]
        for i_ in range(self.iterations):
            while callback():
                temp = self.x
                for k_ in range(self.thinning_step):
                    temp = sampler.forward(temp, self.y, self.physics, self.likelihood, self.prior)
                self.x.append(temp)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        self.statistics.update(self.x[-1])

        return dict(x_est=self.statistics.mean())
