import numpy as np
from scipy.special import logsumexp


def eval_loglik_matrix(eval_loglik, data, draws):
    """Evaluate the log likelihood for each data point and sample.

    Parameters
    ----------
    eval_loglik : func
        function that evaluates the log likelihood for given parameters
    data : tuple
        observed values
    draws : tuple
        parameter sample arrays

    Returns
    -------
    np.ndarray in R+^(ndraws, nobs)
        log likelihood for given parameters and data
    """

    dropna = lambda x: x[~np.isnan(x)]

    return np.array([
        dropna(eval_loglik(*data, *[x[t] for x in draws]))
        for t in range(draws[0].shape[0])])


def eval_waic_score(eval_loglik, data, draws):
    """Evaluates the WAIC approximation to the out of sample predictive density.

    Parameters
    ----------
    eval_loglik : func
        function that evaluates the log likelihood for given parameters
    data : tuple
        observed values
    draws : tuple
        parameter sample arrays

    Returns
    -------
    tuple (float, float)
        estimated out of sample predictive density, standard error of estimate
    """

    L = eval_loglik_matrix(eval_loglik, data, draws)
    
    lpd = np.apply_along_axis(lambda x: logsumexp(x) - np.log(len(x)), 0, L)
    dim = np.var(L, 0)
    elpd = lpd - dim
    
    return (np.mean(elpd), np.std(elpd) / np.sqrt(len(elpd)))


def eval_loo_score(eval_loglik, data, draws):
    """Evaluates the LOO approximation to the out of sample predictive density.

    Parameters
    ----------
    eval_loglik : func
        function that evaluates the log likelihood for given parameters
    data : tuple
        observed values
    draws : tuple
        parameter sample arrays

    Returns
    -------
    tuple (float, float)
        estimated out of sample predictive density, standard error of estimate
    """

    L = eval_loglik_matrix(eval_loglik, data, draws)

    # truncate weights to stabilize variance
    trunc_vals = np.apply_along_axis(
        lambda x: 0.5 * np.log(len(x)) - logsumexp(-x), 0, L)
    trunc_L = np.where(L > trunc_vals, L, trunc_vals)

    lpd = np.apply_along_axis(lambda x: logsumexp(x) - np.log(len(x)), 0, L)
    elpd = np.apply_along_axis(
        lambda x: np.log(len(x)) - logsumexp(-x), 0, trunc_L)
    dim = lpd - elpd
    
    return (np.mean(elpd), np.std(elpd) / np.sqrt(len(elpd)))
