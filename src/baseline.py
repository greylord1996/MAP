import numpy as np
import scipy as sp

import dynamic_equations_to_simulate
import objective_function
# import settings
import data

import matplotlib.pyplot as plt
import admittance_matrix
import sympy


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
    # perturbations = np.random.uniform(low=-0.5, high=0.5, size=4)

    # Just for testing -- remove in release
    perturbations = [
        0.206552362540141 - true_gen_params.d_2,
        0.837172184078094 - true_gen_params.e_2,
        0.665441037484483 - true_gen_params.m_2,
        0.00771416811078329 - true_gen_params.x_d2
    ]

    gen_params_prior_mean = (
        objective_function.OptimizingGeneratorParameters(
            D_Ya=true_gen_params.d_2 + perturbations[0],   # check accordance D_Ya <-> d_2
            Ef_a=true_gen_params.e_2 + perturbations[1],   # check accordance Ef_a <-> e_2
            M_Ya=true_gen_params.m_2 + perturbations[2],   # check accordance M_Ya <-> m_2
            X_Ya=true_gen_params.x_d2 + perturbations[3],  # check accordance X_Ya <-> x_d2
        )
    )
    gen_params_prior_std_dev = (
        objective_function.OptimizingGeneratorParameters(
            D_Ya=1000.0,  # std_dev of D_Ya
            Ef_a=1000.0,  # std_dev of Ef_a
            M_Ya=1000.0,  # std_dev of M_Ya
            X_Ya=1000.0   # std_dev of X_Ya
        )
    )
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
    # stage1_data = data_holder.get_data(remove_fo_band=True)
    stage2_data = data_holder.get_data(remove_fo_band=False)

    # Perturb generator parameters (replace true parameters with prior)
    gen_params_prior_mean, gen_params_prior_std_dev = (
        perturb_gen_params(initial_params.generator_parameters)
    )  # now generator parameters are perturbed and uncertain

    # plot_Im_psd(stage2_data, gen_params_prior_mean.as_array, is_xlabel=False)

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
    print()
    print('######################################################')
    print('### DEBUG: OPTIMIZATION ROUTINE IS STARTING NOW!!! ###')
    print('######################################################')
    print()

    Nel = 10
    N = 100 * 4
    alpha = 0.8
    beta = 0.7
    q = 5
    eps = 0.5 * 1e-2
    mu = gen_params_prior_mean.as_array
    sigma = 0.2 * np.ones(4)
    mu_last = mu
    sigma_last = sigma
    X_best_overall = gen_params_prior_mean.as_array
    S_best_overall = f.compute_from_array(X_best_overall)
    S_target = f.compute_from_array([0.25, 1.0, 1.0, 0.01])
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
            #x_batch[0] = x_batch[0] * 2.0  ### upd
            SA[i] = f.compute_from_array(x_batch)

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


    # opt_res = sp.optimize.minimize(
    #     fun=f.compute_from_array,
    #     x0=gen_params_prior_mean.as_array,
    #     method='BFGS',
    #     options={
    #         'maxiter': 30,
    #         'disp': True
    #     }
    # )
    #
    # print('opt_success?', opt_res.success)
    # print('opt_message:', opt_res.message)
    # print('theta_MAP1 =', opt_res.x)

    # plot_Im_psd(stage2_data, opt_res.x, is_xlabel=True)

    # It is not clear now what should be returned
    return None



def plot_Im_psd(freq_data, gen_params, is_xlabel):
    D_Ya, Ef_a, M_Ya, X_Ya, Omega_a = sympy.symbols(
        'D_Ya Ef_a M_Ya X_Ya Omega_a',
        real=True
    )

    matrix_Y = admittance_matrix.AdmittanceMatrix().Ys
    Y11 = sympy.lambdify(args=[D_Ya, Ef_a, M_Ya, X_Ya, Omega_a], expr=matrix_Y[0, 0], modules='numpy')
    Y12 = sympy.lambdify(args=[D_Ya, Ef_a, M_Ya, X_Ya, Omega_a], expr=matrix_Y[0, 1], modules='numpy')
    # Y21 = sympy.lambdify(args=[D_Ya, Ef_a, M_Ya, X_Ya, Omega_a], expr=matrix_Y[1, 0], modules='numpy')
    # Y22 = sympy.lambdify(args=[D_Ya, Ef_a, M_Ya, X_Ya, Omega_a], expr=matrix_Y[1, 1], modules='numpy')

    predicted_Im = np.zeros(len(freq_data.freqs), dtype=np.complex64)
    # predicted_Ia = np.zeros(len(stage2_data.freqs), dtype=np.complex64)
    for i in range(len(freq_data.freqs)):
        curr_Y11 = Y11(
            D_Ya=gen_params[0],
            Ef_a=gen_params[1],
            M_Ya=gen_params[2],
            X_Ya=gen_params[3],
            Omega_a=2.0 * np.pi * freq_data.freqs[i]
        )
        curr_Y12 = Y12(
            D_Ya=gen_params[0],
            Ef_a=gen_params[1],
            M_Ya=gen_params[2],
            X_Ya=gen_params[3],
            Omega_a=2.0 * np.pi * freq_data.freqs[i]
        )
        # curr_Y21 = Y21(
        #     D_Ya=gen_params[0],
        #     Ef_a=gen_params[1],
        #     M_Ya=gen_params[2],
        #     X_Ya=gen_params[3],
        #     Omega_a=2.0 * np.pi * freq_data.freqs[i]
        # )
        # curr_Y22 = Y22(
        #     D_Ya=gen_params[0],
        #     Ef_a=gen_params[1],
        #     M_Ya=gen_params[2],
        #     X_Ya=gen_params[3],
        #     Omega_a=2.0 * np.pi * freq_data.freqs[i]
        # )

        predicted_Im[i] = curr_Y11 * freq_data.Vm[i] + curr_Y12 * freq_data.Va[i]
        # predicted_Ia[i] = curr_Y21*stage2_data.Vm[i] + curr_Y22*stage2_data.Va[i]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(24, 8))

    plt.plot(
        freq_data.freqs,
        (np.abs(freq_data.Im))**2,
        color='black'
    )
    plt.plot(
        freq_data.freqs,
        (np.abs(predicted_Im))**2
    )

    plot_name = (
        'gen_params=' +
        str(gen_params[0]) + '_' + str(gen_params[1]) + '_' +
        str(gen_params[2]) + '_' + str(gen_params[3])
    )
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize=50, direction='in', length=12, width=3, pad=12)
    plt.yticks([0.001, 0.000001])
    if is_xlabel:
        plt.xlabel('Frequency (Hz)', fontsize=50)
    plt.ylabel(r'$\tilde{\mathrm{I}}$ PSD', fontsize=50)
    plt.legend(['Measured', 'Predicted'], loc='upper left', prop={'size': 50}, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join('samples', plot_name + '.pdf'), dpi=180, format='pdf')





# ----------------------------------------------------------------
# ----------------------- Testing now ----------------------------
# ----------------------------------------------------------------
#
import time
import sys
import os
import os.path
import matplotlib.pyplot as plt

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

# start_time = time.time()
# f1 = objective_function.ObjectiveFunction(
#     freq_data=correct_freq_data,
#     gen_params_prior_mean=gen_params_prior_mean,
#     gen_params_prior_std_dev=gen_params_prior_std_dev,
#     stage=1
# )
# print("constructing objective function : %s seconds" % (time.time() - start_time))
# print('f(0.25, 1.00, 1.00, 0.01) =', f1.compute_from_array([0.25, 1.00, 1.00, 0.01]))


# from numdifftools import Hessian
#
# def fun_hess(x):
#     return Hessian(lambda x: f1.compute_from_array(x))(x)
#
#
# H_true_params = fun_hess([0.25, 1.00, 1.00, 0.01])
# H_middle1 = fun_hess([0.50, 2.00, 2.00, 0.02])
# H_middle2 = fun_hess([1.00, 4.00, 4.00, 0.04])
# H_far = fun_hess([5.0, 20.00, 20.00, 0.4])

# print('H(0.25, 1.00, 1.00, 0.01) =', np.linalg.cond(fun_hess([0.25, 1.00, 1.00, 0.01])))
# print('H(11.0, 11.0, 11.0, 11.0) =', np.linalg.cond(fun_hess([11.0, 11.0, 11.0, 11.0])))
#
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


