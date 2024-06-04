# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax

from . import inv_problems
from . import eval_tools
from . import general_utils

def gradient_ols(X, y, beta):
    return X.T @ (X @ beta - y)
