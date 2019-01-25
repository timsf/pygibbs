from typing import Tuple, TypeVar, Callable

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import logsumexp


Param = TypeVar('Param')
Integrand = Callable[[Param], float]
M = TypeVar('M')
L = TypeVar('L')


def fit_approx(init: Param,
               log_obj: Integrand,
               grad_log_obj: Callable[[Param], np.ndarray],
               hess_log_obj: Callable[[Param], np.ndarray]) -> Tuple[M, L]:
    """Find the parameters of the Laplace approximation to log_obj.

    :param init: initial values
    :param log_obj: integrand
    :param grad_log_obj: gradient of integrand
    :param hess_log_obj: hessian of integrand
    :returns: parameters of laplace approximation
    """

    out = minimize(
        lambda param: -log_obj(param),
        init,
        method='BFGS',
        jac=lambda param: -grad_log_obj(param),
        # hess=lambda param: -hess_log_obj(param)
    )

    if not out.success:
        raise Exception('Laplace approximation did not succeed.')

    mN = out.x
    lN = -hess_log_obj(mN)

    return mN, lN


def est_integral(ndraws: int, m: M, l: L, log_obj: Integrand) -> float:
    """Use Laplace importance sampling to estimate the integral of the given objective with respect to parameters.

    :param ndraws: (>0) number of importance samples
    :param m: mean of laplace approximation
    :param l: precision of laplace approximation
    :param log_obj: integrand
    :returns: estimate of integral
    """

    # preliminary computations
    cf_l = np.linalg.cholesky(l)

    # sample from laplace approximation
    z = np.random.standard_normal((ndraws, *m.shape))
    theta = m + solve_triangular(cf_l.T, z.T).T

    # compute importance sampling estimate
    nc = (m.shape[0] * np.log(2 * np.pi)) / 2 - np.log(ndraws) - np.sum(np.log(np.diag(cf_l)))
    log_weights = np.array([log_obj(the_i) for the_i in theta]) + np.sum(z ** 2, 1) / 2

    return nc + logsumexp(log_weights)
