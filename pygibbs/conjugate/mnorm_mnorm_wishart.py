"""
Provides routines for estimating multivariate normals, ie:
    x[i·]|(mu, sig) ~ Mv-Normal(mu, sig)
    where mu is the mean vector and sig is the covariance matrix.

The prior distribution over (mu, sig) is normal-inv-wishart:
    mu|(sig, m, l) ~ Mv-Normal(m, sig / l)
    sig|(v, s) ~ Inv-Wishart(v, s)

The posterior distribution over (mu, sig) is normal-inv-wishart:
    mu|(x, sig, mN, lN) ~ Mv-Normal(mN, sig / lN)
    sig|(x, vN, sN) ~ Inv-Wishart(vN, sN)

Data
----
nobs : int(>0) # number of observations
nvar : int(>0) # number of variables
x : np.ndarray[nobs, nvar] # data matrix
w : np.ndarray(>0)[nobs] # observation weights

Parameters
----------
mu : np.ndarray[nvar] # mean vector
sig : np.ndarray(PSD)[nvar, nvar] # variance matrix

Hyperparameters
---------------
m : np.ndarray[nvar] # coefficient location vector
l : float(>0) # mean dof
v : float(>0) # variance dof
s : np.ndarray(PSD)[nvar, nvar] # variance location matrix
"""

import numpy as np
from scipy.special import multigammaln
from scipy.stats import invwishart

from pygibbs.tools.densities import eval_mvnorm as eval_loglik


def update(x: np.ndarray, m: np.ndarray, l: float, v: float, s: np.ndarray, w: np.ndarray=None) -> (
        np.ndarray, float, float, np.ndarray):
    """Compute the hyperparameters of the posterior parameter distribution.

    :param x:
    :param m:
    :param l:
    :param v:
    :param s:
    :param w:
    :returns: posterior hyperparameters

    >>> data, param = _generate_fixture(3, 2, seed=666)
    >>> x, = data
    >>> m, l, v, s = param
    >>> update(x, m, l, v, s)
    (array([0.35648366, 0.3198792 ]), 4, 7, array([[6.01128646, 1.22262499],
           [1.22262499, 4.82887539]]))
    >>> w = np.ones(3)
    >>> update(x, m, l, v, s, w=w)
    (array([0.35648366, 0.3198792 ]), 4, 7, array([[6.01128646, 1.22262499],
           [1.22262499, 4.82887539]]))
    """

    nobs, nvar = x.shape
    if nobs != 0:
        if w is not None:
            x = np.diag(w) @ x
        x_mean = np.mean(x, 0)
        x_cov = np.cov(x.T, ddof=0)
    else:
        x_mean = np.zeros(nvar)
        x_cov = np.zeros((nvar, nvar))
    
    lN = l + nobs
    mN = (l * m + nobs * x_mean) / lN
    vN = v + nobs
    sN = s + nobs * (x_cov + l / lN * np.dot((x_mean - m), (x_mean - m).T))

    return (mN, lN, vN, sN)


def marginalize(m: np.ndarray, l: float, v: float, s: np.ndarray) -> 3 * (np.ndarray,):
    """Compute the parameters of the marginal likelihood.

    :param m:
    :param l:
    :param v:
    :param s:
    :returns: parameters of marginal likelihood

    >>> _, param = _generate_fixture(3, 2, seed=666)
    >>> m, l, v, s = param
    >>> marginalize(m, l, v, s)
    (array([0., 0.]), array([[1., 0.],
           [0., 1.]]), 1.0, 4)
    """

    return (m, s / v, 1 / l, v)


def sample_param(ndraws: int, m: np.ndarray, l: float, v: float, s: np.ndarray) -> 2 * (np.ndarray,):
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: number of samples to be drawn
    :param m:
    :param l:
    :param v:
    :param s:
    :returns: samples from the parameter distribution

    >>> _, param = _generate_fixture(3, 2, seed=666)
    >>> m, l, v, s = param
    >>> sample_param(1, m, l, v, s)
    (array([[ 0.03683928, -1.81367702]]), array([[[ 3.74820974, -2.7929328 ],
            [-2.7929328 ,  5.66331323]]]))
    """

    z = np.random.standard_normal((ndraws, m.shape[0]))
    sig = invwishart.rvs(v, s, ndraws).reshape((ndraws, *s.shape))
    mu = m + np.array([np.linalg.cholesky(sig_i / l) @ z_i for z_i, sig_i in zip(z, sig)])

    return (mu, sig)


def sample_data(ndraws: int, m: np.ndarray, l: float, v: float, s: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: number of samples to be drawn
    :param m:
    :param l:
    :param v:
    :param s:
    :returns: samples from the data distribution

    >>> _, param = _generate_fixture(3, 2, seed=666)
    >>> m, l, v, s = param
    >>> sample_data(1, m, l, v, s)
    array([[ 1.29691632, -3.82150683]])
    """

    z = np.random.standard_normal((ndraws, m.shape[0]))
    mu, sig = sample_param(ndraws, m, l, v, s)
    x = mu + np.array([np.linalg.cholesky(sig_i) @ z_i for z_i, sig_i in zip(z, sig)])

    return x


def eval_logmargin(x: np.ndarray, m: np.ndarray, l: float, v: float, s: np.ndarray, w: np.ndarray=None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters. You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param x:
    :param m:
    :param l:
    :param v:
    :param s:
    :param w:
    :returns: log marginal likelihood

    >>> data, param = _generate_fixture(3, 2, seed=666)
    >>> x, = data
    >>> m, l, v, s = param
    >>> eval_logmargin(x, m, l, v, s)
    -8.864244605232614
    >>> w = np.ones(3)
    >>> eval_logmargin(x, m, l, v, s, w=w)
    -8.864244605232614
    """

    nobs, nvar = x.shape
    def nc(_, l, v, s):
        return (v * np.prod(np.linalg.slogdet(s)) + nvar * np.log(l)) / 2 - multigammaln(v / 2, nvar)

    return nc(m, l, v, s) - nc(*update(x, m, l, v, s, w)) - nobs * nvar / 2 * np.log(np.pi)


def get_ev(m: np.ndarray, _, v: float, s: np.ndarray) -> 2 * (np.ndarray,):
    """Evaluate the expectation of parameters given hyperparameters.

    :param m:
    :param l:
    :param v:
    :param s:
    :returns: parameter expectations

    >>> _, param = _generate_fixture(3, 2, seed=666)
    >>> m, l, v, s = param
    >>> get_ev(m, l, v, s)
    (array([0., 0.]), array([[4., 0.],
           [0., 4.]]))
    """

    return (m, s / (v - len(m) - 1))


def get_mode(m: np.ndarray, _, v: float, s: np.ndarray) -> 2 * (np.ndarray,):
    """Evaluate the mode of parameters given hyperparameters.

    :param m:
    :param l:
    :param v:
    :param s:
    :returns: parameter modes

    >>> _, param = _generate_fixture(3, 2, seed=666)
    >>> m, l, v, s = param
    >>> get_mode(m, l, v, s)
    (array([0., 0.]), array([[0.57142857, 0.        ],
           [0.        , 0.57142857]]))
    """

    return (m, s / (v + len(m) + 1))


def _generate_fixture(nobs: int, nvar: int, seed: int=666) -> ((np.ndarray,), 4 * (np.ndarray,)):
    """Generate a set of input data.

    :param nobs:
    :param nvar:
    :param seed: random number generator seed
    :returns: generated data, generated hyperparameters

    >>> data, param = _generate_fixture(3, 2, seed=666)
    >>> data
    (array([[ 0.82418808,  0.479966  ],
           [ 1.17346801,  0.90904807],
           [-0.57172145, -0.10949727]]),)
    >>> param
    (array([0., 0.]), 1, 4, array([[4., 0.],
           [0., 4.]]))
    """

    # ensure deterministic output
    np.random.seed(seed)

    # set input
    x = np.random.standard_normal((nobs, nvar))

    # set hyperparameters
    m = np.zeros(nvar)
    l = 1
    v = nvar + 2
    s = np.diag(np.ones(nvar)) * v

    return ((x,), (m, l, v, s))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
