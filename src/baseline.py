import numpy as np
import scipy as sp

import objective_function
import data
import utils



def perturb_gen_params(true_gen_params):
    """Perturb true generator parameters.

    Gets true values of generator parameters and perturbs them
    by adding a random value to each of them from uniform distribution.
    This uniform distribution is at [-0.5; 0.5] segment.

    Args:
        true_gen_params (class settings.GeneratorParameters):
            true generator parameters specified by user (in GUI)

    Returns:
        perturbed_gen_params (tuple): perturbed parameters of a generator.
            perturbed_gen_params[0] (class OptimizingGeneratorParameters):
                prior mean of generator parameters
            perturbed_gen_params[1] (class OptimizingGeneratorParameters):
                prior standard deviations of generator parameters
    """
    deviation_fraction = 1000.0  # 100000% deviation
    perturbations = np.random.uniform(
        low=-deviation_fraction, high=deviation_fraction, size=4
    )

    gen_params_prior_mean = (
        objective_function.OptimizingGeneratorParameters(
            D_Ya=true_gen_params.d_2 * (1.0 + perturbations[0]),   # check accordance D_Ya <-> d_2
            Ef_a=true_gen_params.e_2 * (1.0 + perturbations[1]),   # check accordance Ef_a <-> e_2
            M_Ya=true_gen_params.m_2 * (1.0 + perturbations[2]),   # check accordance M_Ya <-> m_2
            X_Ya=true_gen_params.x_d2 * (1.0 + perturbations[3]),  # check accordance X_Ya <-> x_d2
        )
    )
    gen_params_prior_std_dev = (
        objective_function.OptimizingGeneratorParameters(
            D_Ya=deviation_fraction,  # std_dev of D_Ya
            Ef_a=deviation_fraction,  # std_dev of Ef_a
            M_Ya=deviation_fraction,  # std_dev of M_Ya
            X_Ya=deviation_fraction   # std_dev of X_Ya
        )
    )

    # Just for testing -- remove in release
    gen_params_prior_mean.D_Ya = 0.206552362540141
    gen_params_prior_mean.Ef_a = 0.837172184078094
    gen_params_prior_mean.M_Ya = 0.665441037484483
    gen_params_prior_mean.X_Ya = 0.00771416811078329

    return gen_params_prior_mean, gen_params_prior_std_dev



def run_all_computations(initial_params):
    """This function is called from GUI (using Run button).

    Args:
        initial_params (class Settings): all configuration parameters
            (should be obtained from GUI)

    Returns:
        None (it is not clear now what should be returned)
    """
    data_holder = data.DataHolder(initial_params)
    stage2_data = data_holder.get_data(remove_fo_band=False)

    # Perturb generator parameters (replace true parameters with prior)
    gen_params_prior_mean, gen_params_prior_std_dev = (
        perturb_gen_params(initial_params.generator_parameters)
    )  # now generator parameters are perturbed and uncertain
    print('PRIOR MEAN =', gen_params_prior_mean.as_array)
    print('PRIOR STD =', gen_params_prior_std_dev.as_array)

    # plot before parameters clarification
    # utils.plot_Im_psd(stage2_data, gen_params_prior_mean.as_array, is_xlabel=False)

    # f denotes the objective function
    f = objective_function.ObjectiveFunction(
        freq_data=stage2_data,
        gen_params_prior_mean=gen_params_prior_mean,
        gen_params_prior_std_dev=gen_params_prior_std_dev
    )

    print(
        'true_gen_params: f(0.25, 1.00, 1.00, 0.01) =',
        f.compute_from_array([0.25, 1.00, 1.00, 0.01])
    )
    print('\n######################################################')
    print('### DEBUG: OPTIMIZATION ROUTINE IS STARTING NOW!!! ###')
    print('######################################################\n')

    posterior_gen_params = minimize_objective_function(
        func=f,
        x0=gen_params_prior_mean.as_array
    )

    # plot after parameters clarification
    # utils.plot_Im_psd(stage2_data, posterior_gen_params, is_xlabel=True)

    # It is not clear now what should be returned
    return None



def minimize_objective_function(func, x0):
    """TODO: write the docstring"""
    Nel = 10
    N = 100 * 4
    alpha = 0.8
    beta = 0.7
    q = 5
    eps = 0.5 * 1e-2
    mu = x0
    sigma = 0.5 * np.ones(4)
    mu_last = mu
    sigma_last = sigma
    X_best_overall = x0
    S_best_overall = func.compute_from_array(X_best_overall)
    S_target = func.compute_from_array([0.25, 1.0, 1.0, 0.01])
    print("S_best_0 = ", S_best_overall)
    print("S_target = ", S_target)

    # S_best_overall = 10000
    t = 0
    SA = np.zeros(N)

    while sigma.max() > eps:
        t = t + 1
        mu = alpha * mu + (1 - alpha) * mu_last
        B_mod = beta - beta * (1 - 1 / t) ** q
        # sigma = B_mod * sigma - (1 - B_mod) * sigma_last  # dynamic smoothing
        sigma = alpha * sigma + (1 - alpha) * sigma_last
        # X = np.ones((N, 1)) * mu + sp.randn(N, 4) * np.diag(np.repeat(sigma.max(), N))
        X = sp.random.normal(mu, sigma, (N, 4))
        # X = sp.random.uniform(mu, [0.3, 1.5, 1.5, 0.1], (N, 4))

        for i in range(N):
            x_batch = X[i]
            # x_batch[0] = x_batch[0] * 2.0  ### upd
            SA[i] = func.compute_from_array(x_batch)

        S_sort = np.sort(SA)
        print("S_sort = ", S_sort)
        I_sort = np.argsort(SA)
        print("I_sort = ", I_sort)

        gam = S_sort[0]
        S_best = S_sort[Nel]

        if (S_best < S_best_overall):
            S_best_overall = S_best
            X_best_overall = X[I_sort[0]]

        mu_last = mu
        sigma_last = sigma
        # Xel = X[I_sort[0]:I_sort[Nel], ]
        temp_best = I_sort[0:Nel]
        Xel = np.take(X, temp_best, axis=0)
        print("Xel = ", Xel)
        mu = np.mean(Xel, axis=0)
        sigma = np.nanstd(Xel, axis=0)
        print("sigma = ", sigma)

    print("mu_last = ", mu)
    print("sigma_last = ", sigma)
    print("X_best = ", X_best_overall)
    print("S_best_overall", S_best_overall)
    return X_best_overall  # what should be returned?

