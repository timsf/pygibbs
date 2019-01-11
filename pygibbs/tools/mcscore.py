from typing import Callable, Tuple

import numpy as np
from scipy.special import logsumexp


def eval_loglik_matrix(eval_loglik: Callable[[tuple, tuple], np.ndarray], data: tuple, draws: tuple) -> np.ndarray:
    """Evaluate the log likelihood for each data point and sample.

    :param eval_loglik: function that evaluates the log likelihood for given parameters
    :param data: observed values
    :param draws: parameter sample arrays
    :returns: [ndraws, nobs] log likelihood for given parameters and data
    """

    def dropna(x):
        return x[~np.isnan(x)]

    return np.array([dropna(eval_loglik(*data, *x)) for x in zip(*draws)])


def eval_waic_score(eval_loglik: Callable[[tuple, tuple], np.ndarray], data: tuple, draws: tuple) -> Tuple[float, float]:
    """Evaluates the WAIC approximation to the out of sample predictive density.

    :param eval_loglik: function that evaluates the log likelihood for given parameters
    :param data: observed values
    :param draws: parameter sample arrays
    :returns: estimated out of sample predictive density, standard error of estimate
    """

    L = eval_loglik_matrix(eval_loglik, data, draws)
    
    lpd = np.apply_along_axis(lambda x: logsumexp(x) - np.log(len(x)), 0, L)
    dim = np.var(L, 0)
    elpd = lpd - dim
    
    return float(np.mean(elpd)), float(np.std(elpd) / np.sqrt(len(elpd)))


def eval_loo_score(eval_loglik: Callable[[tuple, tuple], np.ndarray], data: tuple, draws: tuple) -> Tuple[float, float]:
    """Evaluates the LOO approximation to the out of sample predictive density.

    :param eval_loglik: function that evaluates the log likelihood for given parameters
    :param data: observed values
    :param draws: parameter sample arrays
    :returns: estimated out of sample predictive density, standard error of estimate
    """

    L = eval_loglik_matrix(eval_loglik, data, draws)

    # truncate weights to stabilize variance
    trunc_vals = np.apply_along_axis(lambda x: 0.5 * np.log(len(x)) - logsumexp(-x), 0, L)
    trunc_L = np.where(L > trunc_vals, L, trunc_vals)

    lpd = np.apply_along_axis(lambda x: logsumexp(x) - np.log(len(x)), 0, L)
    elpd = np.apply_along_axis(lambda x: np.log(len(x)) - logsumexp(-x), 0, trunc_L)
    dim = lpd - elpd
    
    return float(np.mean(elpd)), float(np.std(elpd) / np.sqrt(len(elpd)))
