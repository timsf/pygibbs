"""
Implements Monte Carlo inference for Gaussian hierarchical linear models of the following form:
    y[i·]|(x, bet, tau) ~ Mv-Normal(bet[i·] * x, tau * I) [emission distribution]
    bet[i·]|(mu, sig) ~ Mv-Normal(mu, sig) [loading distribution]
    where I is the identity matrix.

Alternatively, the model may be expressed in its non-centered parameterization:
    y[i·]|(x, bet, tau) ~ Mv-Normal((gam[i·] + mu) * x, tau * I) [emission distribution]
    gam[i·]|(mu, sig) ~ Mv-Normal(0, sig) [loading distribution]
    where I is the identity matrix.

The algorithm will automatically switch to the appropriate parameterization.
"""

from typing import Tuple

import numpy as np

from pygibbs.tools.gibbs import em, gibbs
from pygibbs.conjugate import norm_chisq as nc, mnorm_wishart as mnw, mnorm_mnorm_wishart as mnmnw, \
    lm_mnorm_chisq as lmnc, mlm_mnorm as mlmn


Data = Tuple[np.ndarray, np.ndarray]
Eta = Tuple[np.ndarray, np.ndarray, np.ndarray]
Theta = Tuple[np.ndarray, np.ndarray, float]
Param = Tuple[Eta, Theta]
HyperLoad = Tuple[float, np.ndarray]
HyperEmit = Tuple[float, float]
Hyper = Tuple[HyperLoad, HyperEmit]


def estimate(niter: int, y: np.ndarray, x: np.ndarray, load: HyperLoad, emit: HyperEmit) -> Param:
    """Compute the parameter MAP estimator.

    :param niter: (>0) maximum number of iterations
    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param load:
    :param emit:
    :returns: model parameter modes by block
    """

    init = np.random.standard_normal((y.shape[0], x.shape[0]))
    init_eta = init, init, np.ones(2)

    return em(niter, (y, x), (load, emit), init_eta, eval_logobserved, estimate_eta, estimate_theta)


def sample(ndraws: int, y: np.ndarray, x: np.ndarray, load: HyperLoad, emit: HyperEmit) -> Param:
    """Draw Gibbs samples from the joint posterior.

    :param ndraws: (>0) number of draws
    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param load:
    :param emit:
    :returns: model parameter samples by block
    """

    init_eta, init_theta = estimate(100, y, x, load, emit)

    return gibbs(ndraws, (y, x), (load, emit), init_eta, init_theta, sample_eta, sample_theta)


def estimate_eta(data: Data, theta: Theta) -> Eta:
    """Compute the expectation of eta given theta and data.

    :param data:
    :param theta:
    :returns: eta expected values given theta and data
    """

    mu, sig, tau = theta
    hyperN, rho = update_eta(*data, *theta)
    bet, = mlmn.get_ev(*hyperN)
    gam = bet - mu

    return bet, gam, rho


def sample_eta(data: Data, theta: Theta) -> Eta:
    """Sample from eta given theta and data.

    :param data:
    :param theta:
    :returns: eta sample given theta and data
    """

    mu, sig, tau = theta
    hyperN, rho = update_eta(*data, *theta)
    bet, = mlmn.sample_param(1, *hyperN)[0]
    gam = bet - mu

    return bet, gam, rho


def estimate_theta(data: Data, eta: Eta, hyper: Hyper) -> Theta:
    """Compute the mode of theta given eta and data.

    :param data:
    :param eta:
    :param hyper:
    :returns: theta modes given eta and data
    """

    rho = eta[2]
    if rho[0] < rho[1]:
        loadN, emitN = update_theta_cp(*data, *eta, *hyper)
        mu, sig = mnmnw.get_mode(*loadN)
        tau, = nc.get_mode(*emitN)
    else:
        loadN, emitN = update_theta_ncp(*data, *eta, *hyper)
        sig, = mnw.get_mode(*loadN)
        mu, tau = [x[0] for x in lmnc.get_mode(*emitN)]

    return mu, sig, tau


def sample_theta(data: Data, eta: Eta, hyper: Hyper) -> Theta:
    """Sample from theta given eta and data.

    :param data:
    :param eta:
    :param hyper:
    :returns: theta sample given eta and data
    """

    rho = eta[2]
    if rho[0] < rho[1]:
        loadN, emitN = update_theta_cp(*data, *eta, *hyper)
        mu, sig = mnmnw.sample_param(1, *loadN)
        tau, = nc.sample_param(1, *emitN)
    else:
        loadN, emitN = update_theta_ncp(*data, *eta, *hyper)
        sig, = mnw.sample_param(1, *loadN)
        mu, tau = [x[0] for x in lmnc.sample_param(1, *emitN)]

    return mu[0], sig[0], tau[0]


def eval_loglik(data: Data, eta: Eta, theta: Theta) -> np.ndarray:
    """Evaluate the likelihood given both blocks.

    :param data:
    :param eta:
    :param theta:
    :returns: likelihood given both blocks
    """

    return np.sum(nc.eval_loglik(data[0], eta[0] @ data[1], np.array(theta[2])), 1)


def eval_logobserved(data: Data, theta: Theta) -> float:
    """Evaluate the likelihood given theta, integrating out eta variables.

    :param data:
    :param theta:
    :returns: log observed likelihood
    """

    return mlmn.eval_logmargin(*reshape(*data, *theta))


def update_eta(y: np.ndarray, x: np.ndarray, mu: np.ndarray, sig: np.ndarray, tau: float) -> Tuple[mlmn.Hyper, np.ndarray]:
    """Update eta given theta and data.

    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param mu: [nvar]
    :param sig: (PD)[nvar, nvar]
    :param tau: float(>0)
    :returns: eta posterior hyperparameter values given theta and data
    """

    rho = eval_rate_convergence(x, sig, tau)

    return mlmn.update(*reshape(y, x, mu, sig, tau)), rho


def update_theta_cp(y: np.ndarray, x: np.ndarray, bet: np.ndarray, gam: np.ndarray, rho: np.ndarray,
                    load: HyperLoad, emit: HyperEmit) -> Tuple[mnmnw.Hyper, nc.Hyper]:
    """Update theta (centered parameterization) given eta and data.

    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param bet: [nres, nvar]
    :param gam: [nres, nvar]
    :param rho: (>0)[2]
    :param load:
    :param emit:
    :returns: theta hyperparameter values given eta and data
    """

    emit_train = (y - bet @ x).flatten()[:, np.newaxis]
    loadN = mnmnw.update(bet, np.zeros(bet.shape[1]), 0, *load)
    emitN = nc.update(emit_train, np.zeros(1), *emit)

    return loadN, emitN


def update_theta_ncp(y: np.ndarray, x: np.ndarray, bet: np.ndarray, gam: np.ndarray, rho: np.ndarray,
                     load: HyperLoad, emit: HyperEmit) -> Tuple[mnw.Hyper, lmnc.Hyper]:
    """Update theta (non-centered parameterization) given eta and data.

    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param bet: [nres, nvar]
    :param gam: [nres, nvar]
    :param rho: (>0)[2]
    :param load:
    :param emit:
    :returns: theta hyperparameter values given eta and data
    """

    emit_train = (y - gam @ x).flatten()[:, np.newaxis]
    loadN = mnw.update(gam, np.zeros(gam.shape[1]), *load)
    emitN = lmnc.update(emit_train.T, np.tile(x, y.shape[0]), np.zeros([1, x.shape[0]]), np.zeros(2 * [x.shape[0]]), *emit)

    return loadN, emitN


def eval_rate_convergence(x: np.ndarray, sig: np.ndarray, tau: float) -> np.ndarray:
    """Evaluate the rate of geometric convergence of the centered and non-centered parameterizations.

    :param x: [nvar, nobs]
    :param sig: (PD)[nvar, nvar]
    :param tau: (>0)
    :returns: rates of convergence for centered and non-centered parameterizations
    """

    f = sig @ x @ x.T / tau
    feig = np.linalg.eigvals(f)
    rho = (np.max(np.abs(1 / (feig + 1))), np.max(np.abs(1 / (1 / feig + 1))))

    return np.array(rho)


def reshape(y: np.ndarray, x: np.ndarray, mu: np.ndarray, sig: np.ndarray, tau: float) -> 5 * (np.ndarray,):
    """Reshape data and theta to match the theta sampling module.

    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param mu: [nvar]
    :param sig: (PD)[nvar, nvar]
    :param tau: float(>0)
    :returns: reshaped data and theta variables
    """

    return y, x, tau * np.identity(y.shape[1]), mu, sig


def _generate_fixture(nres: int = 3, nobs: int = 2, nvar: int = 1, seed: int = 666) -> (Data, Param, Hyper):
    """Generate a set of input data.

    :param nres: (>0)
    :param nobs: (>0)
    :param nvar: (>0)
    :param seed: random number generator seed
    :returns: generated data, generated hyperparameters
    """

    # ensure deterministic output
    np.random.seed(seed)

    # set eta ground truth
    bet = np.random.standard_normal((nres, nvar))

    # set theta ground truth
    mu = np.zeros(nvar)
    sig = np.identity(nvar)
    tau = 1
    theta = mu, sig, tau

    # set input
    x = np.random.standard_normal((nvar, nobs))
    y = bet @ x + np.random.standard_normal((nres, nobs))
    eta = bet, bet, eval_rate_convergence(x, sig, tau)

    # set loading distribution hyperparameters
    v = nvar + 2
    s = np.diag(np.ones(nvar)) * v
    load = v, s

    # set residual variance hyperparameters
    v = 1
    s = 1
    emit = v, s

    return (y, x), (eta, theta), (load, emit)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
