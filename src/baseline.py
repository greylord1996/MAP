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
    deviation_fraction = 0.5  # 50% deviation
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

    # utils.plot_Im_psd(stage2_data, posterior_gen_params, is_xlabel=True)

    # It is not clear now what should be returned
    return None



def minimize_objective_function(func, x0):
    """"""
    Nel = 10
    N = 100 * 4
    alpha = 0.8
    beta = 0.7
    q = 5
    eps = 0.5 * 1e-2
    mu = x0
    sigma = 0.2 * np.ones(4)
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
    return mu  # mu should be returned?





# ----------------------------------------------------------------
# ----------------------- Testing now ----------------------------
# ----------------------------------------------------------------
#
import sys
import os
import os.path


# WARNING! ONLY FOR TESTING!
PATH_TO_THIS_FILE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PATH_TO_THIS_FILE, '..', 'tests'))
import our_data
import correct_data

TEST_DIR = os.path.join(PATH_TO_THIS_FILE, '..', 'tests', 'Rnd_Amp_0002')
initial_params = our_data.get_initial_params(TEST_DIR)

assert initial_params.generator_parameters.d_2 == 0.25
assert initial_params.generator_parameters.e_2 == 1.00
assert initial_params.generator_parameters.m_2 == 1.00
assert initial_params.generator_parameters.x_d2 == 0.01
assert initial_params.integration_settings.dt_step == 0.05
assert initial_params.freq_data.max_freq == 6.0
assert initial_params.freq_data.lower_fb == 1.988
assert initial_params.freq_data.upper_fb == 2.01
assert initial_params.oscillation_parameters.osc_freq == 2.000
assert initial_params.oscillation_parameters.osc_amp == 0.005


# correct_freq_data = correct_data.get_prepared_freq_data(TEST_DIR)
# print('=================== DATA HAVE BEEN PREPARED ========================')
#
#
# gen_params_prior_mean, gen_params_prior_std_dev = perturb_gen_params(
#     initial_params.generator_parameters
# )  # now generator parameters are perturbed and uncertain

# f = objective_function.ObjectiveFunction(
#     freq_data=correct_freq_data,
#     gen_params_prior_mean=gen_params_prior_mean,
#     gen_params_prior_std_dev=gen_params_prior_std_dev
# )
# print('f(0.25, 1.00, 1.00, 0.01) =', f.compute_from_array([0.25, 1.00, 1.00, 0.01]))


# print()
# print('######################################################')
# print('### DEBUG: OPTIMIZATION ROUTINE IS STARTING NOW!!! ###')
# print('######################################################')
# print()
#
# opt_res = sp.optimize.minimize(
#     fun=f1.compute_from_array,
#     x0=gen_params_prior_mean.as_array,
#     method='BFGS',
#     options={
#         'maxiter': 40,
#         'disp': True
#     }
# )
#
# print('opt_success?', opt_res.success)
# print('opt_message:', opt_res.message)
# print('theta_MAP1 =', opt_res.x)


