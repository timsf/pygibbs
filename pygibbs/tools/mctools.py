import numpy as np
import pandas as pd
from itertools import product


def summ_sample(x: np.ndarray, quants: np.ndarray, name: str = 'param') -> pd.DataFrame:
    '''Produce a tabular summary of the sample's distribution.

    :param x: sample array where the first dimension is the sample dimension
    :param quants: quantiles to be evaluated
    :param name: name of the parameter
    :returns: summary

    >>> np.random.seed(666)
    >>> summ_sample(np.random.standard_normal((100, 2)), np.array([10, 50, 90]), 'x')
               Q10       Q50       Q90
    x[1] -1.278403 -0.043071  1.135279
    x[2] -1.017850  0.091727  1.379015
    '''

    rows = [
        name + '%s' % str(list(tup)).replace(' ', '')
        for tup in product(*(range(1, i + 1) for i in list(x.shape)[1:]))]
    cols = ['Q' + str(quant).zfill(2) for quant in quants]
    
    dat = np.array([np.percentile(x, quant, 0).flatten() for quant in quants]).T

    return pd.DataFrame(dat, index=rows, columns=cols)


def diag_sample(x: np.ndarray, name: str = 'param') -> pd.DataFrame:
    '''Produce a tabular summary of the sample's quality.

    :param x: sample array where the first dimension is the sample dimension
    :param name: name of the parameter
    :returns: summary

    >>> np.random.seed(666)
    >>> diag_sample(np.random.standard_normal((100, 2)), 'x')
            StdErr         ESS
    x[1]  0.102626  100.000000
    x[2]  0.110409   77.125764
    '''

    rows = [
        name + '%s' % str(list(tup)).replace(' ', '')
        for tup in product(*(range(1, i + 1) for i in list(x.shape)[1:]))]
    cols = ['StdErr', 'ESS']
    
    ess = np.array(
        [np.apply_along_axis(est_ess, 0, x, x.shape[0] // 10)]).flatten()
    stderr = np.std(x, 0).flatten() / np.sqrt(ess)

    dat = np.vstack((stderr, ess)).T

    return pd.DataFrame(dat, index=rows, columns=cols)


def est_ess(series, max_lag, tradeoff_par=6):
    '''Compute the effective sample size of a series of MC draws.

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
    '''

    int_autocor = est_int_autocor(series, max_lag, tradeoff_par)

    if int_autocor < 0:
        return np.nan
    return np.min((len(series), len(series) / 2 / int_autocor))


def est_int_autocor(series, max_lag, tradeoff_par):
    '''Estimate the integrated autocorrelation time.
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
    '''

    acf = est_acf(series, max_lag)

    int_autocor = 0.5
    for i in range(len(acf)):
        int_autocor += acf[i]
        if i + 1 >= tradeoff_par * int_autocor:
            return int_autocor
    return int_autocor


def est_acf(series, max_lag):
    '''Estimate the autocorrelation function of a given time series. This is a biased estimator.

    Parameters
    ----------
    series : array_like in R^d
        time series

    Returns
    -------
    np.ndarray
        acf
    '''

    mean = np.mean(series)
    var = np.mean(series ** 2) - mean ** 2

    if var == 0:
        return np.array([np.nan])

    demeaned = series - mean
    acf = np.array([
        sum(demeaned[lag:] * demeaned[:-lag])
        for lag in range(1, min(len(series), max_lag))]) / var / len(series)

    return acf
