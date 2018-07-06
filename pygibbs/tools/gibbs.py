import numpy as np


def em(niter, data, priors, init, eval_obj, update_block1, update_block2, tol=1e-4):
    """Find the MAP estimator of the model with the EM algorithm.

    parameters
    ----------
    niter : int in N+
        maximum number of iterations
    data : tuple
        observed quantities
    priors : tuple
        prior hyperparameters for block2
    init : tuple
        initial block1 variable values
    eval_obj : func
        objective function
    update_block1 : func
        e-step function
    update_block2 : func
        m-step function
    tol : float in R+, default 1e-10
        minimum objective increment

    Returns
    -------
    tuple
        (block1 estimates, block 2 estimates)
    """

    block1, block2 = init, None
    obj = [-np.inf]
    
    for t in range(niter):

        # M-step
        block2 = update_block2(*data, *block1, *priors)

        # E-step
        block1 = update_block1(*data, *block2)

        # stopping criterion
        obj.append(eval_obj(*data, *block2))
        if obj[t + 1] - obj[t] < tol:
            break

    return (block1, block2)


def gibbs(ndraws, data, priors, init, sample_block1, sample_block2):
    """Sample from the posterior distribution given data and hyperparameters.

    parameters
    ----------
    ndraws : int in N+
        number of draws to be sampled
    data : tuple
        observed quantities
    priors : tuple
        prior hyperparameters for block2
    init : tuple
        initial block1 and block2 values
    sample_block1 : func
        block1 given block2 sampling function
    sample_block2 : func
        block2 given block1 sampling function

    Returns
    -------
    tuple
        (block1 draws, block2 draws)
    """

    # init
    block1_init, block2_init = init
    block1 = [[x] for x in block1_init]
    block2 = [[x] for x in block2_init]

    for t in range(ndraws - 1):

        # sample block2 given glock1
        for x, y in zip(block2, sample_block2(*data, *[z[-1] for z in block1], *priors)):
            x.append(y)
        
        # sample block1 given block2
        for x, y in zip(block1, sample_block1(*data, *[z[-1] for z in block2])):
            x.append(y)
        
    return ([np.array(x) for x in block1], [np.array(x) for x in block2])
