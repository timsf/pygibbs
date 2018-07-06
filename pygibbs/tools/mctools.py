import numpy as np
import pandas as pd
from itertools import product


def summ_sample(x, quants, name="param"):
    """Produce a tabular summary of the sample's distribution.

    Parameters
    ----------
    x : np.ndarray
        sample array where the first dimension is the sample dimension
    quants : np.ndarray
        quantiles to be evaluated
    name : str, default "param"
        name of the parameter

    Returns
    -------
    pd.DataFrame
        summary
    """

    rows = [
        name + "%s" % str(list(tup)).replace(" ", "")
        for tup in product(*(range(1, i + 1) for i in list(x.shape)[1:]))]
    cols = ["Q" + str(quant).zfill(2) for quant in quants]
    
    dat = np.array([np.percentile(x, quant, 0).flatten() for quant in quants]).T

    return pd.DataFrame(dat, index=rows, columns=cols)


def diag_sample(x, name="param"):
    """Produce a tabular summary of the sample's quality.

    Parameters
    ----------
    x : np.ndarray
        sample array where the first dimension is the sample dimension
    name : str, default "param"
        name of the parameter

    Returns
    -------
    pd.DataFrame
        summary
    """

    rows = [
        name + "%s" % str(list(tup)).replace(" ", "")
        for tup in product(*(range(1, i + 1) for i in list(x.shape)[1:]))]
    cols = ["StdErr", "ESS"]
    
    ess = np.array(
        [np.apply_along_axis(est_ess, 0, x, x.shape[0] // 10)]).flatten()
    stderr = np.std(x, 0).flatten() / np.sqrt(ess)

    dat = np.vstack((stderr, ess)).T

    return pd.DataFrame(dat, index=rows, columns=cols)


def est_ess(series, max_lag, tradeoff_par=6):
    """Compute the effective sample size of a series of MC draws.

    Parameters
    ----------
    series : np.ndarray in R^ndraws
        1-d sample array
    max_lag : int in N+
        number of lags that enter into autocorrelation computations
    tradeoff_par : int in N+, default 6
        governs the bias-variance tradeoff in the estimation

    Returns
    -------
    float
        estimate of the effective sample size
    """

    int_autocor = est_int_autocor(series, max_lag, tradeoff_par)

    if int_autocor < 0:
        return np.nan
    return np.min((len(series), len(series) / 2 / int_autocor))


def est_int_autocor(series, max_lag, tradeoff_par):
    """Estimate the integrated autocorrelation time.
    Since there is a bias-variance tradeoff involved in the estimation, the acf is integrated up to lag l such that l is the smallest int for which
        l >= tradeoff_par * (0.5 + sum[t = 1][l](acf(t))

    Parameters
    ----------
    series : array_like in R^ndraws
        time series
    tradeoff_par : int {1, .., np.inf}, default 6
        governs the bias-variance tradeoff in the estimation

    Returns
    -------
    float
        estimate of the acf's integrated autocorrelation time
    """

    acf = est_acf(series, max_lag)

    int_autocor = 0.5
    for i in range(len(acf)):
        int_autocor += acf[i]
        if i + 1 >= tradeoff_par * int_autocor:
            return int_autocor
    return int_autocor


def est_exp_autocor(series, max_lag):
    """Estimate the exponential autocorrelation time.
    This method is very sensitive to post-decay noise in the acf.

    Parameters
    ----------
    series : array_like in R^ndraws
        time series

    Returns
    -------
    float
        estimate of the acf's exponential autocorrelation time
    """

    acf = est_acf(series, max_lag)

    return np.nanmax([
        (lag + 1) / -np.log(np.abs(acf[lag]))
        for lag in range(len(acf))])


def est_acf(series, max_lag):
    """Estimate the autocorrelation function of a given time series. This is a biased estimator.

    Parameters
    ----------
    series : array_like in R^d
        time series

    Returns
    -------
    np.ndarray
        acf
    """

    mean = np.mean(series)
    var = np.mean(series ** 2) - mean ** 2

    if var == 0:
        return np.array([np.nan])

    demeaned = series - mean
    acf = np.array([
        sum(demeaned[lag:] * demeaned[:-lag])
        for lag in range(1, min(len(series), max_lag))]) / var / len(series)

    return acf
