import torch.nn as nn
import torch
import numpy as np

class IMLAIterator(nn.Module):
    def __init__(self, step_size, alpha, sigma, theta=0.5, 
                 hist_size=5, tolerance=1e-5, max_iter=500,
                 verbose=0):
        super().__init__()
        self.step_size = step_size
        self.alpha = alpha
        self.noise_std = np.sqrt(2 * step_size)
        self.theta = theta
        self.sigma = sigma
        self.hist_size = hist_size
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.verbose = verbose

    def forward(self, x, y, physics, likelihood, prior):

        noise = torch.randn_like(x) * self.noise_std


        # optimization settings
        n_iter_max = 1

        # initial value
        x_ = x.detach().clone()
        x_.requires_grad = True
        
        # set up lbfgs optimizer
        optimizer = torch.optim.LBFGS([x_], lr=1, max_iter=self.max_iter,
                    max_eval=None, tolerance_grad=self.tolerance,
                    tolerance_change=self.tolerance,
                    history_size=self.hist_size,
                    line_search_fn='strong_wolfe')

        def closure():
            # set up posterior
            posterior = lambda u: likelihood(u, y, physics) + self.alpha * prior(
                u, self.sigma
            )
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            loss = ((1/self.theta) * 
                posterior(self.theta * x_ + (1-self.theta)*x)
                + (1/(2*self.step_size))*torch.linalg.matrix_norm(
                    x_ - x - noise, 'fro')**2).sum()
            
            # Backward pass
            loss.backward()

            return loss

        for _ in range(n_iter_max):
            optimizer.step(closure)
            if self.verbose > 0:
                print(optimizer.state_dict())
        
        return x_.detach().clone() # new sample produced by the ILA algorithm
