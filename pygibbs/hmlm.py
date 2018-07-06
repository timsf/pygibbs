"""
Implements Monte Carlo inference for Gaussian hierarchical linear models of the following form:
    Y[i·]|(X, Bet, tau2) ~ Mv-Normal(Bet[i·] * X[i··], tau2 * I) [emission distribution]
    Bet[i·]|(mu, Sig) ~ Mv-Normal(mu, Sig) [loading distribution]
    where I is the identity matrix.

Alternatively, the model may be expressed in its non-centered parameterization:
    Y[i·]|(X, Bet, tau2) ~ Mv-Normal((Gam[i·] + mu) * X[i··], tau2 * I) [emission distribution]
    Gam[i·]|(mu, Sig) ~ Mv-Normal(0, Sig) [loading distribution]
    where I is the identity matrix.

The algorithm will automatically switch to the appropriate parameterization.

Data
----
Y : Matrix[nres, nobs]
X : Tensor[nres, nvar, nobs]

Block 1
-------
Bet : Matrix[nres, nvar]
Gam : Matrix[nres, nvar]
rho : Vector[2]

Block 2
-------
mu : Vector[nvar]
Sig >= 0 (positive semi-definite) : Matrix[nvar, nvar]
tau2 >= 0 : Real

Hyperparameters
---------------
load0 : (Real >= 0, Matrix[nvar, nvar] >= 0)
    Sig hyperparameters (v0, S0)
emit0 : (Real >= 0, Real >= 0)
    tau2 hyperparameters (v0, s0)
"""

import numpy as np

from pygibbs.tools.gibbs import em, gibbs
from pygibbs.conjugate import norm_chisq as nc, mnorm_wishart as mnw, mnorm_mnorm_wishart as mnmnw, lm_mnorm_chisq as rmnc
from pygibbs.semi_conjugate import mlm_mnorm as mrmn
from pygibbs.tools.densities import eval_lm


def estimate(niter: int,
             Y: np.ndarray,
             X: np.ndarray,
             load0: (float, np.ndarray),
             emit0: (float, float)) -> (list, list):
    """Compute the parameter MAP estimator.

    :param niter: maximum number of iterations
    :param Y: response matrix
    :param X: covariate tensor
    :param load0: loading distribution hyperparameters
    :param emit0: emission distribution hyperparameters
    :returns: model parameter modes by block
    """

    init = np.random.standard_normal((Y.shape[0], X.shape[1]))

    return em(
        niter, (Y, X), (load0, emit0), (init, init, np.ones(2)),
        eval_logobserved, estimate_block1, estimate_block2)


def sample(ndraws: int,
           Y: np.ndarray,
           X: np.ndarray,
           load0: (float, np.ndarray),
           emit0: (float, float)) -> (list, list):
    """Draw Gibbs samples from the joint posterior.

    :param ndraws: number of draws
    :param Y: response matrix
    :param X: covariate tensor
    :param load0: loading distribution hyperparameters
    :param emit0: emission distribution hyperparameters
    :returns: model parameter samples by block
    """

    init = estimate(100, Y, X, load0, emit0)

    return gibbs(
        ndraws, (Y, X), (load0, emit0), init,
        sample_block1, sample_block2)


def update_block1(Y: np.ndarray,
                  X: np.ndarray,
                  mu: np.ndarray,
                  Sig: np.ndarray,
                  tau2: float) -> [(np.ndarray, np.ndarray)]:
    """Update block1 given block2 and data.

    :param Y: response matrix
    :param X: covariate tensor
    :param mu: loading distribution mean vector
    :param Sig: loading distribution covariance matrix
    :param tau2: emission distribution variance
    :returns: block1 posterior hyperparameter values given block2 and data
    """

    return [mrmn.update(*reshape(Y_i[np.newaxis], X_i, mu, Sig, tau2)) for Y_i, X_i in zip(Y, X)]


def update_block2_cp(Y: np.ndarray,
                     X: np.ndarray,
                     Bet: np.ndarray,
                     load0: (float, np.ndarray),
                     emit0: (float, float)) -> (tuple, tuple):
    """Update block2 (centered parameterization) given block1 and data.

    :param Y: response matrix
    :param X: covariate tensor
    :param Bet: centered coefficient matrix
    :param load0: loading distribution covariance matrix
    :param emit0: emission distribution variance
    :returns: block2 hyperparameter values given block1 and data
    """

    emit_train = np.reshape(Y - np.einsum('ij,ijk->ik', Bet, X), (np.prod(Y.shape), 1))
    loadN = mnmnw.update(Bet, 0, 0, *load0)
    emitN = nc.update(emit_train, 0, *emit0)

    return (loadN, emitN)


def update_block2_ncp(Y: np.ndarray,
                      X: np.ndarray,
                      Gam: np.ndarray,
                      load0: (float, np.ndarray),
                      emit0: (float, float)) -> (tuple, tuple):
    """Update block2 (non-centered parameterization) given block1 and data.

    :param Y: response matrix
    :param X: covariate tensor
    :param Gam: non-centered coefficient matrix
    :param load0: loading distribution covariance matrix
    :param emit0: emission distribution variance
    :returns: block2 hyperparameter values given block1 and data
    """

    emit_train = np.reshape(Y - np.einsum('ij,ijk->ik', Gam, X), (np.prod(Y.shape), 1))
    loadN = mnw.update(Gam, 0, *load0)
    emitN = rmnc.update(emit_train.T, np.hstack(X), np.zeros((1, X.shape[1])), np.zeros((X.shape[1], X.shape[1])), *emit0)

    return (loadN, emitN)


def estimate_block1(Y: np.ndarray,
                    X: np.ndarray,
                    mu: np.ndarray,
                    Sig: np.ndarray,
                    tau2: float) -> (np.ndarray, np.ndarray, np.ndarray):
    """Compute the expectation of block1 given block2 and data.

    :param Y: response matrix
    :param X: covariate tensor
    :param mu: loading distribution mean vector
    :param Sig: loading distribution covariance matrix
    :param tau2: emission distribution variance
    :returns: block1 expected values given block2 and data
    """

    Bet = np.vstack([mrmn.get_ev(m, S)[0] for m, S in update_block1(Y, X, mu, Sig, tau2)])
    Gam = Bet - mu
    rho = eval_rate_convergence(X, Sig, tau2)

    return (Bet, Gam, rho)


def estimate_block2(Y: np.ndarray,
                    X: np.ndarray,
                    Bet: np.ndarray,
                    Gam: np.ndarray,
                    rho: np.ndarray,
                    load0: (float, np.ndarray),
                    emit0: (float, float)) -> (np.ndarray, np.ndarray, float):
    """Compute the mode of block2 given block1 and data.

    :param Y: response matrix
    :param X: covariate tensor
    :param Bet: centered coefficient matrix
    :param Gam: non-centered coefficient matrix
    :param rho: rates of convergence for centered and non-centered parameterization
    :param load0: loading distribution covariance matrix
    :param emit0: emission distribution variance
    :returns: block2 modes given block1 and data
    """

    if rho[0] < rho[1]:
        loadN, emitN = update_block2_cp(Y, X, Bet, load0, emit0)
        mu, Sig = mnmnw.get_mode(*loadN)
        tau2, = nc.get_mode(*emitN)
    else:
        loadN, emitN = update_block2_ncp(Y, X, Gam, load0, emit0)
        Sig, = mnw.get_mode(*loadN)
        mu, tau2 = [x[0] for x in rmnc.get_mode(*emitN)]

    return (mu, Sig, tau2)


def sample_block1(Y: np.ndarray,
                  X: np.ndarray,
                  mu: np.ndarray,
                  Sig: np.ndarray,
                  tau2: float) -> (np.ndarray, np.ndarray, np.ndarray):
    """Sample from block1 given block2 and data.

    :param Y: response matrix
    :param X: covariate tensor
    :param mu: loading distribution mean vector
    :param Sig: loading distribution covariance matrix
    :param tau2: emission distribution variance
    :returns: block1 sample given block2 and data
    """

    Bet = np.vstack([mrmn.sample_param(1, m, S)[0][0] for m, S in update_block1(Y, X, mu, Sig, tau2)])
    Gam = Bet - mu
    rho = eval_rate_convergence(X, Sig, tau2)

    return (Bet, Gam, rho)


def sample_block2(Y: np.ndarray,
                  X: np.ndarray,
                  Bet: np.ndarray,
                  Gam: np.ndarray,
                  rho: np.ndarray,
                  load0: (float, np.ndarray),
                  emit0: (float, float)) -> (np.ndarray, np.ndarray, float):
    """Sample from block2 given block1 and data.

    :param Y: response matrix
    :param X: covariate tensor
    :param Bet: centered coefficient matrix
    :param Gam: non-centered coefficient matrix
    :param rho: rates of convergence for centered and non-centered parameterization
    :param load0: loading distribution covariance matrix
    :param emit0: emission distribution variance
    :returns: block2 sample given block1 and data
    """

    if rho[0] < rho[1]:
        loadN, emitN = update_block2_cp(Y, X, Bet, load0, emit0)
        mu, Sig = mnmnw.sample_param(1, *loadN)
        tau2, = nc.sample_param(1, *emitN)
    else:
        loadN, emitN = update_block2_ncp(Y, X, Gam, load0, emit0)
        Sig, = mnw.sample_param(1, *loadN)
        mu, tau2 = [x[0] for x in rmnc.sample_param(1, *emitN)]

    return (mu[0], Sig[0], tau2[0])


def eval_loglik(Y: np.ndarray,
                X: np.ndarray,
                Bet: np.ndarray,
                Gam: np.ndarray,
                rho: np.ndarray,
                mu: np.ndarray,
                Sig: np.ndarray,
                tau2: float) -> float:
    """Evaluate the likelihood given both blocks.

    :param Y: response matrix
    :param X: covariate tensor
    :param Bet: centered coefficient matrix
    :param Gam: non-centered coefficient matrix
    :param rho: rates of convergence for centered and non-centered parameterization
    :param mu: loading distribution mean vector
    :param Sig: loading distribution covariance matrix
    :param tau2: emission distribution variance
    :returns: likelihood given both blocks
    """

    return sum([eval_lm(Y_i[np.newaxis], X_i, tau2) for Y_i, X_i in zip(Y, X)])


def eval_logobserved(Y: np.ndarray, X: np.ndarray, mu: np.ndarray, Sig: np.ndarray, tau2: float) -> float:
    """Evaluate the likelihood given block 2, integrating out block 1 variables.

    :param Y: response matrix
    :param X: covariate tensor
    :param mu: loading distribution mean vector
    :param Sig: loading distribution covariance matrix
    :param tau2: emission distribution variance
    :returns: log observed likelihood
    """

    return sum([mrmn.eval_logmargin(*reshape(Y_i[np.newaxis], X_i, mu, Sig, tau2)) for Y_i, X_i in zip(Y, X)])


def eval_rate_convergence(X: np.ndarray, Sig: np.ndarray, tau2: float) -> np.ndarray:
    """Evaluate the rate of geometric convergence of the centered and non-centered parameterizations.

    :param X: covariate tensor
    :param Sig: loading distribution covariance matrix
    :param tau2: emission distribution variance
    :returns: rates of convergence for centered and non-centered parameterizations
    """

    F = Sig @ np.einsum('ijk,ilk->jl', X, X)
    Feig = np.linalg.eigvals(F) / X.shape[0] / tau2
    rho = (np.max(np.abs(1 / (Feig + 1))), np.max(np.abs(1 / (1 / Feig + 1))))

    return np.array(rho)


def reshape(Y: np.ndarray, X: np.ndarray, mu: np.ndarray, Sig: np.ndarray, tau2: float) -> 5 * (np.ndarray,):
    """Reshape data and block2 to match the block2 sampling module.

    :param Y: response matrix
    :param X: covariate matrix
    :param mu: loading distribution mean vector
    :param Sig: loading distribution covariance matrix
    :param tau2: emission distribution variance
    :returns: reshaped data and block2 variables
    """

    return (Y, X, tau2 * np.identity(Y.shape[1]), mu, Sig)