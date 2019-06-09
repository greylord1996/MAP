import numpy as np
import scipy as sp

import dynamic_equations_to_simulate
import objective_function
# import settings
import data



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
    stage1_data = data_holder.get_data(stage=1)

    # Perturb generator parameters (replace true parameters with prior)
    gen_params_prior_mean, gen_params_prior_std_dev = (
        perturb_gen_params(initial_params.generator_parameters)
    )  # now generator parameters are perturbed and uncertain

    # f1 denotes the objective function which has prepared for stage1
    f1 = objective_function.ObjectiveFunction(
        freq_data=stage1_data,
        gen_params_prior_mean=gen_params_prior_mean,
        gen_params_prior_std_dev=gen_params_prior_std_dev,
        stage=1
    )

    print(
        'true_gen_params: f(0.25, 1.00, 1.00, 0.01) =',
        f1.compute_from_array([0.25, 1.00, 1.00, 0.01])
    )
    print()
    print('######################################################')
    print('### DEBUG: OPTIMIZATION ROUTINE IS STARTING NOW!!! ###')
    print('######################################################')
    print()

    opt_res = sp.optimize.minimize(
        fun=f1.compute_from_array,
        x0=gen_params_prior_mean.as_array,
        method='BFGS',
        options={
            'maxiter': 40,
            'disp': True
        }
    )

    print('opt_success?', opt_res.success)
    print('opt_message:', opt_res.message)
    print('theta_MAP1 =', opt_res.x)

    # It is not clear now what should be returned
    return None









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


