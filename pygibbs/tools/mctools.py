from typing import List, Tuple
from itertools import product

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from pygibbs.tools.linalg import logdet_pd


def summ_sample(x: np.ndarray, quants: np.ndarray, name: str = 'param') -> pd.DataFrame:
    """Produce a tabular summary of the sample's distribution.

    :param x: sample array where the first dimension is the sample dimension
    :param quants: quantiles to be evaluated
    :param name: name of the parameter
    :returns: summary

    >>> np.random.seed(666)
    >>> summ_sample(np.random.standard_normal((100, 2)), np.array([10, 50, 90]), 'x')
               Q10       Q50       Q90
    x[1] -1.278403 -0.043071  1.135279
    x[2] -1.017850  0.091727  1.379015
    """

    rows = [
        name + '%s' % str(list(tup)).replace(' ', '')
        for tup in product(*(range(1, i + 1) for i in list(x.shape)[1:]))]
    cols = ['Q' + str(quant).zfill(2) for quant in quants]

    dat = np.array([np.percentile(x, quant, 0).flatten() for quant in quants]).T

    return pd.DataFrame(dat, index=rows, columns=cols)


def diag_sample(x: np.ndarray, name: str = 'param') -> pd.DataFrame:
    """Produce a tabular summary of the sample's quality.

    :param x: sample array where the first dimension is the sample dimension
    :param name: name of the parameter
    :returns: summary

    >>> np.random.seed(666)
    >>> diag_sample(np.random.standard_normal((1000, 2)), 'x')
            StdErr          ESS
    x[1]  0.029115  1139.621801
    x[2]  0.026836  1382.019899
    """

    rows = [
        name + '%s' % str(list(tup)).replace(' ', '')
        for tup in product(*(range(1, i + 1) for i in list(x.shape)[1:]))]
    cols = ['StdErr', 'ESS']

    x_flat = np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    ess, _ = est_ess(x_flat)
    stderr = np.std(x_flat, 0) / np.sqrt(ess)

    dat = np.vstack((stderr, ess)).T

    return pd.DataFrame(dat, index=rows, columns=cols)


def est_ess(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute the effective sample size of a series of MC draws.

    :param x: time series array
    :returns: effective sample size for each time series dimension, effective sample size for entire process

    >>> np.random.seed(666)
    >>> est_ess(np.random.standard_normal((1000, 2)))
    (array([1139.62180093, 1382.01989909]), 239.87952224266587)
    """

    # set truncation point for autocorrelation computation
    trunc_point = np.floor(x.shape[0] ** (1 / 3))

    iid_cov = np.cov(x.T)
    raw_mc_cov = est_lugsail_cov(x, trunc_point, 0.5, 3)
    mc_cov = correct_pd(raw_mc_cov, x.shape[0], np.sqrt(np.log(x.shape[0]) / x.shape[1]), 0.5)
    marginal_ess = x.shape[0] * np.diag(iid_cov) / np.diag(mc_cov)
    joint_ess = x.shape[0] * (logdet_pd(iid_cov) / logdet_pd(mc_cov)) ** (1 / x.shape[1])

    return marginal_ess, joint_ess


def est_lugsail_cov(x: np.ndarray, b: int, c: float = 0.5, r: float = 3) -> np.ndarray:
    """Use the "lugsail" estimator to estimate the monte carlo covariance.

    :param x: time series array
    :param b: batch size of underlying batch means estimator
    :param c: bias-variance trade-off parameter
    :param r: bias-variance trade-off parameter
    :returns: "lugsail" monte carlo covariance estimate

    >>> np.random.seed(666)
    >>> est_lugsail_cov(np.random.standard_normal((1000, 2)), 9)
    array([[ 0.84855114, -0.29886087],
           [-0.29886087,  0.72087742]])
    """

    # enforce equally sized batches
    b -= b % r
    if -int(x.shape[0] % b) != 0:
        x = x[:-int(x.shape[0] % b)]

    return (est_batch_cov(x, b) - c * est_batch_cov(x, np.floor(b / r))) / (1 - c)


def est_batch_cov(x: np.ndarray, batch_size: int) -> np.ndarray:
    """Use the "batch means" estimator to estimate the monte carlo covariance.

    :param x: time series array
    :param batch_size:
    :returns: "batch means" monte carlo covariance estimate

    >>> np.random.seed(666)
    >>> est_batch_cov(np.random.standard_normal((1000, 2)), 10)
    array([[ 0.91857231, -0.12105535],
           [-0.12105535,  0.93242508]])
    """

    # enforce equally sized batches
    if -int(x.shape[0] % batch_size) != 0:
        x = x[:-int(x.shape[0] % batch_size)]

    return batch_size * np.cov(est_batch_means(x, batch_size), rowvar=False)


def est_batch_means(x: np.ndarray, batch_size: int) -> List[np.ndarray]:
    """Split an array into batches and compute batch means.

    :param x: time series array
    :param batch_size:
    :returns: mean for each batch

    >>> np.random.seed(666)
    >>> x = np.random.standard_normal((4, 2))
    >>> x
    array([[ 0.82418808,  0.479966  ],
           [ 1.17346801,  0.90904807],
           [-0.57172145, -0.10949727],
           [ 0.01902826, -0.94376106]])
    >>> est_batch_means(x, 2)
    [array([0.99882805, 0.69450704]), array([-0.27634659, -0.52662917])]
    """

    n_batches = x.shape[0] / batch_size
    batches = np.split(x, n_batches)

    return [np.mean(batch, 0) for batch in batches]


def correct_pd(s: np.ndarray, n_samples: int, e: float = None, r: float = None) -> np.ndarray:
    """Ensure positive definiteness of a covariance matrix estimate by inflating its negative eigenvalues

    :param s: symmetric covariance matrix estimate
    :param n_samples: size of estimation sample
    :param e: eigenvalue threshold scale
    :param r: eigenvalue threshold power
    :returns: corrected covariance matrix estimate

    >>> np.random.seed(666)
    >>> x = np.random.standard_normal((100, 2))
    >>> est_lugsail_cov(x, 9)
    array([[ 1.49949716, -1.22971603],
           [-1.22971603,  0.73786048]])
    >>> correct_pd(est_lugsail_cov(x, 9), x.shape[0])
    array([[ 1.74003486, -1.06098403],
           [-1.06098403,  0.85622233]])
    """

    # set defaults for threshold scaling
    if e is None:
        e = np.sqrt(np.log(n_samples) / s.shape[0])
    if r is None:
        r = 0.5

    sd = np.sqrt(np.diag(s))
    eigval, eigvec = eigh((s / sd).T / sd)
    threshold = e * n_samples ** (-r)
    pd_eigval = np.where(eigval > threshold, eigval, threshold)

    return (eigvec.T @ np.diag(pd_eigval) @ eigvec * sd).T * sd
