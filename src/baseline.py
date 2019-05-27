import numpy as np
import scipy as sp

import dynamic_equations_to_simulate
import objective_function
# import settings
import data



def prepare_freq_data(initial_params):
    """Prepares data for our optimization routine.

    Simulates data in frequency domain, applies white noise,
    performs FFT, trims data. If you want to use these data
    for stage1, you should exclude forced oscillation band too
    (by calling exclude_fo_band from the result of this function).

    Args:
        initial_params (class Settings): all configuration parameters
            (should be obtained from GUI)

    Returns:
        freq_data (class FreqData): data in frequency domain
            which are prepared for running stage2 but not stage1
    """
    ode_solver_object = dynamic_equations_to_simulate.OdeSolver(
        white_noise=initial_params.white_noise,
        gen_param=initial_params.generator_parameters,
        osc_param=initial_params.oscillation_parameters,
        integr_param=initial_params.integration_settings
    )
    # ode_solver_object.solve() -- shouldn't be called???

    # Simulate data in time domain
    ode_solver_object.simulate_time_data()
    time_data = data.TimeData(
        Vm_time_data=ode_solver_object.Vc1_abs,
        Va_time_data=ode_solver_object.Vc1_angle,
        Im_time_data=ode_solver_object.Ig_abs,
        Ia_time_data=ode_solver_object.Ig_angle,
        dt=ode_solver_object.dt
    )

    # Apply white noise to simulated data in time domain
    # TODO: Move snr and d_coi to GUI
    time_data.apply_white_noise(snr=45.0, d_coi=0.0)

    # Moving from time domain to frequency domain
    freq_data = data.FreqData(time_data)

    # Trim data
    freq_data.remove_zero_frequency()
    freq_data.trim(
        min_freq=0.0,
        max_freq=initial_params.freq_data.max_freq
    )

    # Return data in frequency domain which are prepared for stage 1
    return freq_data



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
    # 4 - number of generator parameters - it is hardcoded
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
        None (It is not clear now what should be returned)
    """
    freq_data = prepare_freq_data(initial_params)

    # Remove data from forced oscillation band
    # (it is necessary only for running stage1, not stage2!)
    freq_data.remove_data_from_fo_band(
        min_fo_freq=initial_params.freq_data.lower_fb,
        max_fo_freq=initial_params.freq_data.upper_fb
    )

    # Perturb generator parameters (replace true parameters with prior)
    gen_params_prior_mean, gen_params_prior_std_dev = (
        perturb_gen_params(initial_params.generator_parameters)
    )

    # f denotes the objective function
    f = objective_function.ObjectiveFunction(
        freq_data=freq_data,
        gen_params_prior_mean=gen_params_prior_mean,
        gen_params_prior_std_dev=gen_params_prior_std_dev
    )

    print('\n######################################################')
    print('### DEBUG: OPTIMIZATION ROUTINE IS STARTING NOW!!! ###')
    print('######################################################\n')

    opt_res = sp.optimize.minimize(
        fun=f.compute_from_array,
        x0=gen_params_prior_mean.as_array,
        # method='CG'
        # jac=f.compute_gradient_from_array,
        # tol=100.0,  What does this argument mean?
        options={
            'maxiter': 50,
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
correct_freq_data = correct_data.get_prepared_freq_data(TEST_DIR)


gen_params_prior_mean, gen_params_prior_std_dev = perturb_gen_params(
    initial_params.generator_parameters
)  # now generator parameters are perturbed and uncertain

start_time = time.time()
f = objective_function.ObjectiveFunction(
    freq_data=correct_freq_data,
    gen_params_prior_mean=gen_params_prior_mean,
    gen_params_prior_std_dev=gen_params_prior_std_dev
)
print("constructing objective function : %s seconds" % (time.time() - start_time))

# print()
# print('######################################################')
# print('### DEBUG: OPTIMIZATION ROUTINE IS STARTING NOW!!! ###')
# print('######################################################')
# print()
#
# opt_res = sp.optimize.minimize(
#     fun=f.compute_from_array,
#     x0=gen_params_prior_mean.as_array,
#     # method='CG'
#     # jac=f.compute_gradient_from_array,
#     # tol=100.0,  What does this argument mean?
#     options={
#         'maxiter': 20,
#         'disp': True
#     }
# )
#
# print('opt_success?', opt_res.success)
# print('opt_message:', opt_res.message)
# print('theta_MAP1 =', opt_res.x)


