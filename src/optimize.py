"""Implementation of the cross-entropy optimization method."""


import numpy as np
import scipy as sp


def minimize(func, x0):
    """Minimize scalar function using cross-entropy optimization method.

    Args:
        func (ObjectiveFunction): Function to minimize.
        x0 (numpy.ndarray): Starting point (typically, prior parameters).

    Returns:
        x_min (numpy.ndarray): Found minimum.
    """
    params_dimension = len(x0)
    Nel = 10
    N = 750
    alpha = 0.8
    eps = 5.0 * 1e-3
    mu = x0
    sigma = 0.5 * np.ones(params_dimension)

    mu_last = mu
    sigma_last = sigma
    t = 0

    while sigma.max() > eps:
        t = t + 1
        mu = alpha * mu + (1 - alpha) * mu_last
        sigma = alpha * sigma + (1 - alpha) * sigma_last

        X = sp.random.normal(mu, sigma, (N, params_dimension))
        SA = func.compute(X)

        I_sort = np.argsort(SA)
        mu_last = mu
        sigma_last = sigma
        temp_best = I_sort[0:Nel]
        Xel = np.take(X, temp_best, axis=0)
        mu = np.mean(Xel, axis=0)
        sigma = np.nanstd(Xel, axis=0)

        # print('mu = ', mu)
        # print("sigma = ", sigma, '\n')

    x_min = mu
    return x_min

