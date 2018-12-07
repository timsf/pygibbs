"""
Provides routines for estimating multivariate normals, ie:
    x[i·]|(mu, sig) ~ Mv-Normal(mu, sig)
    where mu is the mean vector and sig is the covariance matrix.

The prior distribution over (sig,) is inv-wishart:
    sig|(v, s) ~ Inv-Wishart(v, s)

The posterior distribution over (sig,) is inv-wishart:
    sig|(x, vN, sN) ~ Inv-Wishart(vN, sN)

Data
----
nobs : int(>0) # number of observations
nvar : int(>0) # number of variables
x : np.ndarray[nobs, nvar] # data matrix
w : np.ndarray(>0)[nobs] # observation weights

Parameters
----------
mu : np.ndarray[nvar] # mean vector (known)
sig : np.ndarray(PSD)[nvar, nvar] # variance matrix

Hyperparameters
---------------
v : float(>0) # variance dof
s : np.ndarray(PSD)[nvar, nvar] # variance location matrix
"""

import numpy as np
from scipy.special import multigammaln
from scipy.stats import invwishart

from pygibbs.tools.densities import eval_mvnorm as eval_loglik, eval_matt


def update(x: np.ndarray, mu: np.ndarray, v: float, s: np.ndarray, w: np.ndarray=None) -> (float, np.ndarray):
    """Compute the hyperparameters of the posterior parameter distribution.

    :param x:
    :param mu:
    :param v:
    :param s:
    :param w:
    :returns: posterior hyperparameters

    >>> data, param = _generate_fixture(3, 2, seed=666)
    >>> x, mu = data
    >>> v, s = param
    >>> update(x, mu, v, s)
    (7, array([[6.38317859, 1.52492303],
           [1.52492303, 5.06872541]]))
    >>> w = np.ones(3)
    >>> update(x, mu, v, s, w=w)
    (7, array([[6.38317859, 1.52492303],
           [1.52492303, 5.06872541]]))
    """

    nobs, nvar = x.shape
    if nobs != 0:
        if w is not None:
            x = np.diag(w) @ x
        x_cov = np.mean([np.outer(x_i, x_i) for x_i in x - mu], 0)
    else:
        x_cov = np.zeros((nvar, nvar))

    vN = v + nobs
    sN = s + nobs * x_cov

    return (vN, sN)


def marginalize(v: float, s: np.ndarray) -> 3 * (np.ndarray,):
    """Compute the parameters of the marginal likelihood.

    :param v:
    :param s:
    :returns: parameters of marginal likelihood

    >>> data, param = _generate_fixture(3, 2, seed=666)
    >>> _, mu = data
    >>> v, s = param
    >>> marginalize(v, s)
    (array([[1., 0.],
           [0., 1.]]), 4)
    """

    return (s / v, v)


def sample_param(ndraws: int, v: float, s: np.ndarray) -> (np.ndarray,):
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: number of samples to be drawn
    :param v:
    :param s:
    :returns: samples from the parameter distribution

    >>> data, param = _generate_fixture(3, 2, seed=666)
    >>> x, _ = data
    >>> v, s = param
    >>> sample_param(1, v, s)
    (array([[[ 2.77468306, -0.06137196],
            [-0.06137196,  3.8732027 ]]]),)
    """

    sig = invwishart.rvs(v, s, ndraws).reshape((ndraws, *s.shape))

    return (sig,)


def sample_data(ndraws: int, mu: np.ndarray, v: float, s: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: number of samples to be drawn
    :param mu:
    :param v:
    :param s:
    :returns: samples from the data distribution

    >>> data, param = _generate_fixture(3, 2, seed=666)
    >>> _, mu = data
    >>> v, s = param
    >>> sample_data(1, mu, v, s)
    array([[[ 0.03683928, -1.81367702]]])
    """

    z = np.random.standard_normal((ndraws, mu.shape[0]))
    sig = sample_param(ndraws, v, s)
    x = mu + np.array([np.linalg.cholesky(sig_i) @ z_i for z_i, sig_i in zip(z, sig)])

    return x


def eval_logmargin(x: np.ndarray, mu: np.ndarray, v: float, s: np.ndarray, w: np.ndarray=None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters. You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param x:
    :param mu:
    :param v:
    :param s:
    :param w:
    :returns: log marginal likelihood

    >>> data, param = _generate_fixture(3, 2, seed=666)
    >>> x, mu = data
    >>> v, s = param
    >>> eval_logmargin(x, mu, v, s)
    -7.7817037865996035
    >>> w = np.ones(3)
    >>> eval_logmargin(x, mu, v, s, w=w)
    -7.7817037865996035
    """

    nobs, nvar = x.shape
    def nc(v, s):
        return v / 2 * np.prod(np.linalg.slogdet(s)) - multigammaln(v / 2, nvar)

    return nc(v, s) - nc(*update(x, mu, v, s, w)) - nobs * nvar / 2 * np.log(np.pi)


def get_ev(v: float, s: np.ndarray) -> (np.ndarray,):
    """Evaluate the expectation of parameters given hyperparameters.

    :param v:
    :param s:
    :returns: parameter expectations

    >>> _, param = _generate_fixture(3, 2, seed=666)
    >>> v, s = param
    >>> get_ev(v, s)
    (array([[4., 0.],
           [0., 4.]]),)
    """

    return (s / (v - s.shape[0] - 1),)


def get_mode(v: float, s: np.ndarray) -> (np.ndarray,):
    """Evaluate the mode of parameters given hyperparameters.

    :param v:
    :param s:
    :returns: parameter modes

    >>> _, param = _generate_fixture(3, 2, seed=666)
    >>> v, s = param
    >>> get_mode(v, s)
    (array([[0.57142857, 0.        ],
           [0.        , 0.57142857]]),)
    """

    return (s / (v + s.shape[0] + 1),)


def _eval_logmargin_check(x: np.ndarray, mu: np.ndarray, v: np.ndarray, s: np.ndarray, w: np.ndarray=None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters. You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param x:
    :param mu:
    :param v:
    :param s:
    :param w:
    :returns: log marginal likelihood

    >>> data, param = _generate_fixture(3, 2, seed=666)
    >>> x, mu = data
    >>> v, s = param
    >>> l1 = eval_logmargin(x, mu, v, s)
    >>> l2 = _eval_logmargin_check(x, mu, v, s)
    >>> np.isclose(l1, l2)
    True
    >>> l1
    >>> l2
    """

    nobs, nvar = x.shape
    if w is not None:
        x = np.diag(w) @ x
    sig, nu = marginalize(v, s)

    return eval_matt(x, mu, np.ones(nobs), sig, nu)


def _generate_fixture(nobs: int, nvar: int, seed: int = 666) -> (2 * (np.ndarray,), 4 * (np.ndarray,)):
    """Generate a set of input data.

    :param nobs:
    :param nvar:
    :param seed: random number generator seed
    :returns: generated data, generated hyperparameters

    >>> data, param = _generate_fixture(3, 2, seed=666)
    >>> data
    (array([[ 0.82418808,  0.479966  ],
           [ 1.17346801,  0.90904807],
           [-0.57172145, -0.10949727]]), array([0., 0.]))
    >>> param
    (4, array([[4., 0.],
           [0., 4.]]))
    """

    # ensure deterministic output
    np.random.seed(seed)

    # set input
    x = np.random.standard_normal((nobs, nvar))
    mu = np.zeros(nvar)

    # set hyperparameters
    v = nvar + 2
    s = np.diag(np.ones(nvar)) * v

    return ((x, mu), (v, s))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
