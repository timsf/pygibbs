"""
Provides routines for estimating normals, ie:
    x[·j]|(mu, sig) ~ Normal(mu[j], sig[j])
    where mu is the mean and sig is the standard deviation.

The prior distribution over (sig,) is scaled-inv-chi^2:
    sig[j]|(v, s) ~ Scaled-Inv-Chi^2(v[j], s[j])

The posterior distribution over (sig,) is scaled-inv-chi^2:
    sig[j]|(x[·j], mu, vN, sN) ~ Scaled-Inv-Chi^2(vN[j], sN[j])

Data
----
nobs : int(>0) # number of observations
nvar : int(>0) # number of variables
x : np.ndarray[nobs, nvar] # data matrix
w : np.ndarray(>0)[nobs] # observation weights

Parameters
----------
mu : np.ndarray[nres, nvar] # mean vector (known)
sig : np.ndarray(<0)[nres] # variance vector

Hyperparameters
---------------
v : np.ndarray(>0)[nres] # variance dof vector
s : np.ndarray(>0)[nres] # variance location vector
"""

import numpy as np
from scipy.special import gammaln

from pygibbs.tools.densities import eval_norm as eval_loglik, eval_mvt


def update(x: np.ndarray, mu: np.ndarray, v: np.ndarray, s: np.ndarray, w: np.ndarray=None) -> 2 * (np.ndarray,):
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
    (array([6., 6.]), array([5.38317859, 4.06872541]))
    >>> w = np.ones(3)
    >>> update(x, mu, v, s, w=w)
    (array([6., 6.]), array([5.38317859, 4.06872541]))
    """

    nobs, nvar = np.sum(np.isfinite(x), 0), x.shape[1]
    if w is not None:
        x = np.diag(w) @ x
    x_var = np.where(nobs != 0, np.nanmean(np.square(x - mu), 0), np.zeros(nvar))

    vN = v + nobs
    sN = s + nobs * x_var

    return (vN, sN)


def marginalize(v: np.ndarray, s: np.ndarray) -> 3 * (np.ndarray,):
    """Compute the parameters of the marginal likelihood.

    :param v:
    :param s:
    :returns: parameters of marginal likelihood

    >>> data, param = _generate_fixture(3, 2, seed=666)
    >>> _, mu = data
    >>> v, s = param
    >>> marginalize(v, s)
    (array([1., 1.]), array([3., 3.]))
    """

    return (s / v, v)


def sample_param(ndraws: int, v: np.ndarray, s: np.ndarray) -> (np.ndarray,):
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: number of samples to be drawn
    :param v:
    :param s:
    :returns: samples from the parameter distribution

    >>> data, param = _generate_fixture(3, 2, seed=666)
    >>> x, _ = data
    >>> v, s = param
    >>> sample_param(1, v, s)
    (array([[1.26332761, 3.61132912]]),)
    """

    sig = s / np.random.chisquare(v, (ndraws, s.shape[0]))

    return (sig,)


def sample_data(ndraws: int, mu: np.ndarray, v: np.ndarray, s: np.ndarray) -> np.ndarray:
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
    array([[[ 0.01646105, -1.62380457]]])
    """

    z = np.random.standard_normal((ndraws, mu.shape[0]))
    sig = sample_param(ndraws, v, s)
    sqrt_sig2 = np.sqrt(sig)
    x = mu + np.array([sqrt_sig2_i * z_i for z_i, sqrt_sig2_i in zip(z, sqrt_sig2)])

    return x


def eval_logmargin(x: np.ndarray, mu: np.ndarray, v: np.ndarray, s: np.ndarray, w: np.ndarray=None) -> float:
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
    -7.770320350482517
    >>> w = np.ones(3)
    >>> eval_logmargin(x, mu, v, s, w=w)
    -7.770320350482517
    """

    nobs = np.sum(np.isfinite(x), 0)
    def nc(v, s):
        return v / 2 * np.log(s) - gammaln(v / 2)

    return float(np.sum(nc(v, s) - nc(*update(x, mu, v, s, w)) - nobs / 2 * np.log(np.pi)))


def get_ev(v: np.ndarray, s: np.ndarray) -> (np.ndarray,):
    """Evaluate the expectation of parameters given hyperparameters.

    :param v:
    :param s:
    :returns: parameter expectations

    >>> _, param = _generate_fixture(3, 2, seed=666)
    >>> v, s = param
    >>> get_ev(v, s)
    (array([3., 3.]),)
    """

    return (s / (v - 2),)


def get_mode(v: np.ndarray, s: np.ndarray) -> (np.ndarray,):
    """Evaluate the mode of parameters given hyperparameters.

    :param v:
    :param s:
    :returns: parameter modes

    >>> _, param = _generate_fixture(3, 2, seed=666)
    >>> v, s = param
    >>> get_mode(v, s)
    (array([0.6, 0.6]),)
    """

    return (s / (v + 2),)


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
    """

    nobs, nvar = x.shape
    if w is not None:
        x = np.diag(w) @ x
    sig, nu = marginalize(v, s)

    return float(np.sum([eval_mvt(x.T[[j]], mu[j], np.repeat(sig[j], nobs), nu[j]) for j in range(nvar)]))


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
    (array([3., 3.]), array([3., 3.]))
    """

    # ensure deterministic output
    np.random.seed(seed)

    # set input
    x = np.random.standard_normal((nobs, nvar))
    mu = np.zeros(nvar)

    # set hyperparameters
    v = np.ones(nvar) + 2
    s = np.ones(nvar) * v

    return ((x, mu), (v, s))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
