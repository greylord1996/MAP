import numpy as np
import scipy as sp

import dynamic_equations_to_simulate
import objective_function
import settings
import data



def perturb_gen_params(true_gen_params):
    """Perturb true generator parameters.

    It gets true values of generator parameters and perturbs them
    by adding a random value to each of them from uniform distribution.
    This uniform distribution is at [-0.5; 0.5] segment.

    Args:
        true_gen_params (class settings.GeneratorParameters): true generator
            parameters specified by user (in GUI)
    """
    # 4 - number of generator parameters - it is hardcoded
    perturbations = np.random.uniform(low=-0.5, high=0.5, size=4)

    # Just for testing
    # perturbations = [0.0, 0.0, 0.0, 0.0]

    return objective_function.UncertainGeneratorParameters(
        D_Ya=true_gen_params.d_2 + perturbations[0],  # check accordance D_Ya <-> d_2
        Ef_a=true_gen_params.e_2 + perturbations[1],  # check accordance Ef_a <-> e_2
        M_Ya=true_gen_params.m_2 + perturbations[2],  # check accordance M_Ya <-> m_2
        X_Ya=true_gen_params.x_d2 + perturbations[3],  # check accordance X_Ya <-> x_d2
        std_dev_D_Ya=1000.0,
        std_dev_Ef_a=1000.0,
        std_dev_M_Ya=1000.0,
        std_dev_X_Ya=1000.0
    )



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
    prior_gen_params = perturb_gen_params(all_params['GeneratorParameters'])

    # f denotes the objective function
    f = objective_function.ObjectiveFunction(
        freq_data=freq_data,
        prior_gen_params=prior_gen_params
    )

    # Here we should minimize the objective function
    # opt_res = sp.optimize.minimize(
    #     fun=f.compute_by_array,
    #     x0=prior_gen_params.get_as_array(),
    #     method='BFGS',
    #     # tol=...?
    #     options={
    #         'maxiter': 1,
    #         'disp': True
    #     }
    # )

    # It is not clear now what should be returned
    return ode_solver_object.get_appropr_data_to_gui()











# ----------------------------------------------------------------
# ----------------------- Testing now ----------------------------
# ----------------------------------------------------------------

import time
import sys


FD = settings.FreqData(
    lower_fb=1.99,
    upper_fb=2.01,
    max_freq=6.00,
    dt=0.05
)

WN = settings.WhiteNoise(
    rnd_amp=0.000
)

GP = settings.GeneratorParameters(  # true generator parameters
    d_2=0.25,
    e_2=1.0,
    m_2=1.0,
    x_d2=0.01,
    ic_d2=1.0
)

IS = settings.IntegrationSettings(
    dt_step=0.05,
    df_length=100.0
)

OP = settings.OscillationParameters(
    osc_amp=2.00,
    osc_freq=0.005
)

solver = dynamic_equations_to_simulate.OdeSolver(
    white_noise=WN,
    gen_param=GP,
    osc_param=OP,
    integr_param=IS
)
# solver.solve()

solver.simulate_time_data()
time_data = data.TimeData(
    Vm_time_data=solver.Vc1_abs,
    Va_time_data=solver.Vc1_angle,
    Im_time_data=solver.Ig_abs,
    Ia_time_data=solver.Ig_angle,
    dt=solver.dt
)

print('Vm_time_data =', time_data.Vm)
print('Im_time_data =', time_data.Im)
print('Va_time_data =', time_data.Va)
print('Ia_time_data =', time_data.Ia)



time_data.apply_white_noise(snr=45.0, d_coi=0.0)


freq_data = data.FreqData(time_data)
# freq_data.remove_zero_frequency()
# freq_data.trim(min_freq=0.0, max_freq=FD.max_freq)
# freq_data.remove_data_from_FO_band()

print('Vm_freq_data =', freq_data.Vm)
print('Im_freq_data =', freq_data.Im)
print('Va_freq_data =', freq_data.Va)
print('Ia_freq_data =', freq_data.Ia)


print('===========================================')


# prior_gen_params = perturb_gen_params(GP)  # now params are perturbed and uncertain
#
# start_time = time.time()
# f = objective_function.ObjectiveFunction(
#     freq_data=freq_data,
#     prior_gen_params=prior_gen_params
# )
# print("constructing objective function : %s seconds" % (time.time() - start_time))
#
# start_time = time.time()
# f0 = f.compute(prior_gen_params)
# print('f0 =', f0, type(f0))
# print("calculating objective function : %s seconds" % (time.time() - start_time))
#
#
# opt_res = sp.optimize.minimize(
#     fun=f.compute_by_array,
#     x0=prior_gen_params.as_array,
#     method='BFGS',
#     # tol=15.5,
#     options={
#         'maxiter': 5,
#         'disp': True
#     }
# )
#
# print('opt_success?', opt_res.success)
# print('opt_message:', opt_res.message)
# print('theta_MAP1 =', opt_res.x)
#
