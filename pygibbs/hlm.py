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

Data
----
y : np.ndarray[nres, nobs] # response matrix
x : np.ndarray[nvar, nobs] # covariate matrix

Block 1
-------
bet : np.ndarray[nres, nvar] # centered coefficient matrix
gam : np.ndarray[nres, nvar] # non-centered coefficient matrix
rho : np.ndarray[2] # convergence rates for centered/non-centered parameterization

Block 2
-------
mu : np.ndarray[nvar] # loading distribution location vector
sig : np.ndarray(PD)[nvar, nvar] # loading distribution variance matrix
tau : float(>0) # emission distribution variance

Hyperparameters
---------------
load : tuple # loading distribution prior
    (float(>0) # dof, np.ndarray(PSD)[nvar, nvar] # location)
emit : tuple # emission distribution prior
    (float(>0) # dof, float(>0) # location)
"""

import numpy as np

from pygibbs.tools.gibbs import em, gibbs
from pygibbs.conjugate import norm_chisq as nc, mnorm_wishart as mnw, mnorm_mnorm_wishart as mnmnw, \
    lm_mnorm_chisq as rmnc, mlm_mnorm as mrmn
from pygibbs.tools.densities import eval_lm


def estimate(niter: int,
             y: np.ndarray,
             x: np.ndarray,
             load: (float, np.ndarray),
             emit: 2 * (float,)) -> 2 * (list,):
    """Compute the parameter MAP estimator.

    :param niter: maximum number of iterations
    :param y:
    :param x:
    :param load:
    :param emit:
    :returns: model parameter modes by block

    >>> data, param = _generate_fixture(3, 2, 1, seed=666)
    >>> y, x = data
    >>> load, emit = param
    >>> block1, block2 = estimate(10, y, x, load, emit)
    >>> block1
    (array([[ 0.87926421],
           [-0.22260438],
           [-0.17316597]]), array([[ 0.719135  ],
           [-0.38273359],
           [-0.33329518]]), array([0.47991797, 0.52008203]))
    >>> block2
    (array([0.16012921]), array([[0.47181098]]), array([0.39604008]))
    """

    init = np.random.standard_normal((y.shape[0], x.shape[0]))

    return em(niter, (y, x), (load, emit), (init, init, np.ones(2)), eval_logobserved, estimate_block1, estimate_block2)


def sample(ndraws: int,
           y: np.ndarray,
           x: np.ndarray,
           load: (float, np.ndarray),
           emit: 2 * (float,)) -> 2 * (list,):
    """Draw Gibbs samples from the joint posterior.

    :param ndraws: number of draws
    :param y:
    :param x:
    :param load:
    :param emit:
    :returns: model parameter samples by block

    >>> data, param = _generate_fixture(3, 2, 1, seed=666)
    >>> y, x = data
    >>> load, emit = param
    >>> block1, block2 = sample(10, y, x, load, emit)
    >>> [np.mean(s, 0) for s in block1]
    [array([[ 0.50802877],
           [-0.12771121],
           [-0.16904228]]), array([[ 0.1745868 ],
           [-0.46115319],
           [-0.50248425]]), array([0.58712939, 0.41287061])]
    >>> [np.mean(s, 0) for s in block2]
    [array([0.33344197]), array([[0.54740638]]), array([0.72927159])]
    """

    init = estimate(100, y, x, load, emit)

    return gibbs(
        ndraws, (y, x), (load, emit), init,
        sample_block1, sample_block2)


def update_block1(y: np.ndarray,
                  x: np.ndarray,
                  mu: np.ndarray,
                  sig: np.ndarray,
                  tau: float) -> 2 * (np.ndarray,):
    """Update block1 given block2 and data.

    :param y:
    :param x:
    :param mu:
    :param sig:
    :param tau:
    :returns: block1 posterior hyperparameter values given block2 and data
    """

    return mrmn.update(*reshape(y, x, mu, sig, tau))


def update_block2_cp(y: np.ndarray,
                     x: np.ndarray,
                     bet: np.ndarray,
                     load: (float, np.ndarray),
                     emit: 2 * (float,)) -> ((float, np.ndarray), 2 * (float,)):
    """Update block2 (centered parameterization) given block1 and data.

    :param y:
    :param x:
    :param bet:
    :param load:
    :param emit:
    :returns: block2 hyperparameter values given block1 and data
    """

    emit_train = (y - bet @ x).flatten()[:, np.newaxis]
    loadN = mnmnw.update(bet, 0, 0, *load)
    emitN = nc.update(emit_train, 0, *emit)

    return (loadN, emitN)


def update_block2_ncp(y: np.ndarray,
                      x: np.ndarray,
                      gam: np.ndarray,
                      load: (float, np.ndarray),
                      emit: 2 * (float,)) -> ((float, np.ndarray), 2 * (float,)):
    """Update block2 (non-centered parameterization) given block1 and data.

    :param y:
    :param x:
    :param gam:
    :param load:
    :param emit:
    :returns: block2 hyperparameter values given block1 and data
    """

    emit_train = (y - gam @ x).flatten()[:, np.newaxis]
    loadN = mnw.update(gam, 0, *load)
    emitN = rmnc.update(emit_train.T, np.tile(x, y.shape[0]), np.zeros([1, x.shape[0]]), np.zeros(2 * [x.shape[0]]), *emit)

    return (loadN, emitN)


def estimate_block1(y: np.ndarray,
                    x: np.ndarray,
                    mu: np.ndarray,
                    sig: np.ndarray,
                    tau: float) -> 3 * (np.ndarray,):
    """Compute the expectation of block1 given block2 and data.

    :param y:
    :param x:
    :param mu:
    :param sig:
    :param tau:
    :returns: block1 expected values given block2 and data
    """

    bet, = mrmn.get_ev(*update_block1(y, x, mu, sig, tau))
    gam = bet - mu
    rho = eval_rate_convergence(x, sig, tau)

    return (bet, gam, rho)


def estimate_block2(y: np.ndarray,
                    x: np.ndarray,
                    bet: np.ndarray,
                    gam: np.ndarray,
                    rho: np.ndarray,
                    load: (float, np.ndarray),
                    emit: 2 * (float,)) -> 3 * (np.ndarray,):
    """Compute the mode of block2 given block1 and data.

    :param y:
    :param x:
    :param bet:
    :param gam:
    :param rho:
    :param load:
    :param emit:
    :returns: block2 modes given block1 and data
    """

    if rho[0] < rho[1]:
        loadN, emitN = update_block2_cp(y, x, bet, load, emit)
        mu, sig = mnmnw.get_mode(*loadN)
        tau, = nc.get_mode(*emitN)
    else:
        loadN, emitN = update_block2_ncp(y, x, gam, load, emit)
        sig, = mnw.get_mode(*loadN)
        mu, tau = [x[0] for x in rmnc.get_mode(*emitN)]

    return (mu, sig, tau)


def sample_block1(y: np.ndarray,
                  x: np.ndarray,
                  mu: np.ndarray,
                  sig: np.ndarray,
                  tau: float) -> 3 * (np.ndarray,):
    """Sample from block1 given block2 and data.

    :param y:
    :param x:
    :param mu:
    :param sig:
    :param tau:
    :returns: block1 sample given block2 and data
    """

    bet, = mrmn.sample_param(1, *update_block1(y, x, mu, sig, tau))[0]
    gam = bet - mu
    rho = eval_rate_convergence(x, sig, tau)

    return (bet, gam, rho)


def sample_block2(y: np.ndarray,
                  x: np.ndarray,
                  bet: np.ndarray,
                  gam: np.ndarray,
                  rho: np.ndarray,
                  load: (float, np.ndarray),
                  emit: 2 * (float,)) -> (2 * (np.ndarray,), float):
    """Sample from block2 given block1 and data.

    :param y:
    :param x:
    :param bet:
    :param gam:
    :param rho:
    :param load:
    :param emit:
    :returns: block2 sample given block1 and data
    """

    if rho[0] < rho[1]:
        loadN, emitN = update_block2_cp(y, x, bet, load, emit)
        mu, sig = mnmnw.sample_param(1, *loadN)
        tau, = nc.sample_param(1, *emitN)
    else:
        loadN, emitN = update_block2_ncp(y, x, gam, load, emit)
        sig, = mnw.sample_param(1, *loadN)
        mu, tau = [x[0] for x in rmnc.sample_param(1, *emitN)]

    return (mu[0], sig[0], tau[0])


def eval_loglik(y: np.ndarray,
                x: np.ndarray,
                bet: np.ndarray,
                gam: np.ndarray,
                rho: np.ndarray,
                mu: np.ndarray,
                sig: np.ndarray,
                tau: float) -> float:
    """Evaluate the likelihood given both blocks.

    :param y:
    :param x:
    :param bet:
    :param gam:
    :param rho:
    :param mu:
    :param sig:
    :param tau:
    :returns: likelihood given both blocks
    """

    return eval_lm(y, x, bet, tau)


def eval_logobserved(y: np.ndarray, x: np.ndarray, mu: np.ndarray, sig: np.ndarray, tau: float) -> float:
    """Evaluate the likelihood given block 2, integrating out block 1 variables.

    :param y:
    :param x:
    :param mu:
    :param sig:
    :param tau:
    :returns: log observed likelihood
    """

    return mrmn.eval_logmargin(*reshape(y, x, mu, sig, tau))


def eval_rate_convergence(x: np.ndarray, sig: np.ndarray, tau: float) -> np.ndarray:
    """Evaluate the rate of geometric convergence of the centered and non-centered parameterizations.

    :param x:
    :param sig:
    :param tau:
    :returns: rates of convergence for centered and non-centered parameterizations
    """

    f = sig @ x @ x.T / tau
    feig = np.linalg.eigvals(f)
    rho = (np.max(np.abs(1 / (feig + 1))), np.max(np.abs(1 / (1 / feig + 1))))

    return np.array(rho)


def reshape(y: np.ndarray, x: np.ndarray, mu: np.ndarray, sig: np.ndarray, tau: float) -> 5 * (np.ndarray,):
    """Reshape data and block2 to match the block2 sampling module.

    :param y:
    :param x:
    :param mu:
    :param sig:
    :param tau:
    :returns: reshaped data and block2 variables
    """

    return (y, x, tau * np.identity(y.shape[1]), mu, sig)


def _generate_fixture(nres: int, nobs: int, nvar: int, seed: int=666) -> (2 * (np.ndarray,), ((float, np.ndarray), 2 * (float,))):
    """Generate a set of input data.

    :param nres:
    :param nobs:
    :param nvar:
    :param seed: random number generator seed
    :returns: generated data, generated hyperparameters

    >>> data, param = _generate_fixture(3, 2, 1, seed=666)
    >>> data
    (array([[ 1.17346801,  0.90904807],
           [-0.57172145, -0.10949727],
           [ 0.01902826, -0.94376106]]), array([[0.82418808, 0.479966  ]]))
    >>> param
    ((3, array([[3.]])), (3, 3))
    """

    # ensure deterministic output
    np.random.seed(seed)

    # set input
    x = np.random.standard_normal((nvar, nobs))
    y = np.random.standard_normal((nres, nobs))

    # set loading distribution hyperparameters
    v = nvar + 2
    s = np.diag(np.ones(nvar)) * v
    load = (v, s)

    # set residual variance hyperparameters
    v = 3
    s = 3
    emit = (v, s)

    return ((y, x), (load, emit))