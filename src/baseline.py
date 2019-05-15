import numpy as np
import scipy as sp

import dynamic_equations_to_simulate
import objective_function
import settings
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
                prior means of generator parameters
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

    gen_params_prior_means = (
        objective_function.OptimizingGeneratorParameters(
            D_Ya=true_gen_params.d_2 + perturbations[0],   # check accordance D_Ya <-> d_2
            Ef_a=true_gen_params.e_2 + perturbations[1],   # check accordance Ef_a <-> e_2
            M_Ya=true_gen_params.m_2 + perturbations[2],   # check accordance M_Ya <-> m_2
            X_Ya=true_gen_params.x_d2 + perturbations[3],  # check accordance X_Ya <-> x_d2
        )
    )
    gen_params_prior_std_devs = (
        objective_function.OptimizingGeneratorParameters(
            D_Ya=1000.0,  # std dev of D_Ya
            Ef_a=1000.0,  # std dev of Ef_a
            M_Ya=1000.0,  # std dev of M_Ya
            X_Ya=1000.0   # std dev of X_Ya
        )
    )

    return gen_params_prior_means, gen_params_prior_std_devs



def run_all_computations(all_params):
    ode_solver_object = dynamic_equations_to_simulate.OdeSolver(
        white_noise=all_params['WhiteNoise'],
        gen_param=all_params['GeneratorParameters'],
        osc_param=all_params['OscillationParameters'],
        integr_param=all_params['IntegrationSettings']
    )
    # ode_solver_object.solve()

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
    time_data.apply_white_noise(snr=45.0, d_coi=0.0)

    # Moving from time domain to frequency domain
    freq_data = data.FreqData(time_data)

    # Perturb generator parameters (replace true parameters with prior)
    gen_params_prior_means, gen_params_prior_std_devs = (
        perturb_gen_params(all_params['GeneratorParameters'])
    )

    # f denotes the objective function
    f = objective_function.ObjectiveFunction(
        freq_data=freq_data,
        gen_params_prior_means=gen_params_prior_means,
        gen_params_prior_std_devs=gen_params_prior_std_devs
    )

    # Here we should minimize the objective function

    # It is not clear now what should be returned
    return ode_solver_object.get_appropr_data_to_gui()











# ----------------------------------------------------------------
# ----------------------- Testing now ----------------------------
# ----------------------------------------------------------------

import time
import sys
import os
import os.path

# ONLY FOR TESTING!
PATH_TO_THIS_FILE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PATH_TO_THIS_FILE, '..', 'tests'))
import our_data
import correct_data

TEST_DIR = os.path.join(PATH_TO_THIS_FILE, '..', 'tests', 'Rnd_Amp_0002')

initial_params = our_data.get_initial_params(TEST_DIR)
correct_freq_data = correct_data.get_prepared_freq_data(TEST_DIR)

print('=================== DATA HAVE BEEN PREPARED ========================')


gen_params_prior_means, gen_params_prior_std_devs = perturb_gen_params(
    initial_params.generator_parameters
)  # now params are perturbed and uncertain

start_time = time.time()
f = objective_function.ObjectiveFunction(
    freq_data=correct_freq_data,
    gen_params_prior_means=gen_params_prior_means,
    gen_params_prior_std_devs=gen_params_prior_std_devs
)

print("constructing objective function : %s seconds" % (time.time() - start_time))

#
# import matplotlib.pyplot as plt
#
# true_gen_params = objective_function.OptimizingGeneratorParameters(
#     D_Ya=initial_params.generator_parameters.d_2,
#     Ef_a=initial_params.generator_parameters.e_2,
#     M_Ya=initial_params.generator_parameters.m_2,
#     X_Ya=initial_params.generator_parameters.x_d2
# )
# assert true_gen_params.D_Ya == 0.25
# assert true_gen_params.Ef_a == 1.00
# assert true_gen_params.M_Ya == 1.00
# assert true_gen_params.X_Ya == 0.01
#
# thetas4 = np.arange(start=0.001, stop=0.050, step=0.001)
# repeated_true_gen_params_arrays = objective_function._construct_gen_params_arrays(
#     true_gen_params,
#     len(thetas4)
# )
#
# f_values = np.array([
#     f.compute_from_array(np.array([
#         repeated_true_gen_params_arrays['D_Ya'][i],
#         repeated_true_gen_params_arrays['Ef_a'][i],
#         repeated_true_gen_params_arrays['M_Ya'][i],
#         thetas4[i],
#     ]))
#     for i in range(len(thetas4))
# ])
#
# plt.plot(thetas4, f_values)
# plt.savefig(os.path.join(PATH_TO_THIS_FILE, '..', 'samples', 'theta4.pdf'), dpi=180, format='pdf')

#
# print('f(0.2500, 1.0000, 1.0000, 0.0100) =', f.compute_from_array([0.2500, 1.0000, 1.0000, 0.0100]))
# print('f(0.3015, 1.2460, 1.1922, 0.0484) =', f.compute_from_array([0.3015, 1.2460, 1.1922, 0.0484]))
# print('f(0.2200, 1.1200, 1.3700, 0.0500) =', f.compute_from_array([0.2200, 1.1200, 1.3700, 0.0500]))
# print('f(0.2066, 0.8372, 0.6654, 0.0077) =', f.compute_from_array([0.2066, 0.8372, 0.6654, 0.0077]))
# print('f(0.2462, 1.2772, 1.1675, 0.0420) =', f.compute_from_array([0.2462, 1.2772, 1.1675, 0.0420]))
# print('f(0.1373, 1.0525, 1.4881, 0.0479) =', f.compute_from_array([0.1373, 1.0525, 1.4881, 0.0479]))


# print()
# print('######################################################')
# print('### DEBUG: OPTIMIZATION ROUTINE IS STARTING NOW!!! ###')
# print('######################################################')
# print()
#
#
# opt_res = sp.optimize.minimize(
#     fun=f.compute_from_array,
#     x0=[0.25, 1.0, 1.0, 0.01],
#     # method='',
#     jac=f.compute_gradient_from_array
#     # tol=15.5,
#     # options={
#     #     'maxiter': 5,
#     #     'disp': True
#     # }
# )
#
# print('opt_success?', opt_res.success)
# print('opt_message:', opt_res.message)
# print('theta_MAP1 =', opt_res.x)

