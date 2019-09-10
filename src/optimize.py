"""Implementation of cross-entropy optimization method."""


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
    beta = 0.7
    q = 5
    eps = 5 * 1e-3
    mu = x0
    sigma = 0.5 * np.ones(params_dimension)

    mu_last = mu
    sigma_last = sigma
    X_best_overall = x0
    S_best_overall = func.compute(X_best_overall)
    # S_target = func.compute_from_array([0.25, 1.0, 1.0, 0.01])
    # print("S_best_0 = ", S_best_overall)
    # print("S_target = ", S_target)

    # S_best_overall = 10000
    t = 0
    # SA = np.zeros(N)

    while sigma.max() > eps:
        t = t + 1
        mu = alpha * mu + (1 - alpha) * mu_last
        B_mod = beta - beta * (1 - 1 / t) ** q
        # sigma = B_mod * sigma - (1 - B_mod) * sigma_last  # dynamic smoothing
        sigma = alpha * sigma + (1 - alpha) * sigma_last
        # X = np.ones((N, 1)) * mu + sp.randn(N, 4) * np.diag(np.repeat(sigma.max(), N))
        X = sp.random.normal(mu, sigma, (N, params_dimension))
        # X = sp.random.uniform(mu, [0.3, 1.5, 1.5, 0.1], (N, 4))

        SA = func.compute(X)

        S_sort = np.sort(SA)
        # print("S_sort = ", S_sort)
        I_sort = np.argsort(SA)
        # print("I_sort = ", I_sort)

        # gam = S_sort[0] -- not used?
        S_best = S_sort[Nel]

        if S_best < S_best_overall:
            S_best_overall = S_best
            X_best_overall = X[I_sort[0]]

        mu_last = mu
        sigma_last = sigma
        # Xel = X[I_sort[0]:I_sort[Nel], ]
        temp_best = I_sort[0:Nel]
        Xel = np.take(X, temp_best, axis=0)
        # print("Xel = ", Xel)
        mu = np.mean(Xel, axis=0)
        sigma = np.nanstd(Xel, axis=0)
        print('mu = ', mu)
        print("sigma = ", sigma)
        print()

    print("mu_last = ", mu)
    print("sigma_last = ", sigma)
    print("X_best = ", X_best_overall)
    print("S_best_overall", S_best_overall)

    x_min = X_best_overall
    return x_min  # what should be returned?

