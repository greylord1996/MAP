import time

import numpy as np

# from generator import admittance_matrix
# from generator import dynamic_equations_to_simulate
from motor import admittance_matrix
from motor import dynamic_equations_to_simulate

import bayesian_framework as bf
import data
import utils


def get_generator_time_data():
    """Simulate generator's data in time domain."""
    ode_solver_object = dynamic_equations_to_simulate.OdeSolver(
        noise={
            'rnd_amp': 0.002,
            'snr': 45.0
        },
        gen_param={
            'd_2': 0.25,
            'e_2': 1.0,
            'm_2': 1.0,
            'x_d2': 0.01,
            'ic_d2': 1.0
        },
        osc_param={
            'osc_amp': 0.000,
            'osc_freq': 0.0
        },
        integr_param={
            'df_length': 100.0,
            'dt_step': 0.05
        }
    )
    ode_solver_object.simulate_time_data()
    inputs = np.array([
        ode_solver_object.Vc1_abs,
        ode_solver_object.Vc1_angle
    ])

    outputs = np.array([
        ode_solver_object.Ig_abs,
        ode_solver_object.Ig_angle
    ])
    return data.TimeData(inputs, outputs, dt=ode_solver_object.dt)


def get_motor_time_data():
    """Simulate motor's data in time domain."""
    ode_solver_object = dynamic_equations_to_simulate.OdeSolver(
        noise={
            'rnd_amp': 0.002,
            'snr': 45.0
        },
        osc_param={
            'osc_amp': 0.000,
            'osc_freq': 0.0
        },
        integr_param={
            'df_length': 200.0,
            'dt_step': 0.001
        }
    )
    ode_solver_object.simulate()
    # ode_solver_object.show_results_in_test_mode()
    # print("ode_solver_object.V1t = ", ode_solver_object.vt)
    inputs = np.array([
        ode_solver_object.vc1_real,
        ode_solver_object.vc1_imag
    ])
    outputs = np.array([
        ode_solver_object.id,
        ode_solver_object.iq
    ])
    print('!!! inputs.shape =', inputs.shape)
    print('!!! outputs.shape =', outputs.shape)

    assert ode_solver_object.dt == 0.001
    return data.TimeData(inputs, outputs, dt=ode_solver_object.dt)



def main():
    time_data = get_motor_time_data()

    # true_params = np.array([0.25, 1.00, 1.00, 0.01])
    true_params = np.array([0.08, 0.2, 0.5])
    n_params = len(true_params)
    prior_params_std = np.array([0.5 for _ in range(n_params)])

    snrs = 1.0 * np.arange(1, 26, 1)
    optimization_time = np.zeros(len(snrs))
    priors = np.zeros((len(snrs), n_params))
    posteriors = np.zeros((len(snrs), n_params))
    for snr_idx in range(len(snrs)):
        print('\n######################################################')
        print('SNR =', snrs[snr_idx])
        freq_data = bf.prepare_freq_data(
            time_data=time_data, snr=snrs[snr_idx],
            remove_zero_freq=True, min_freq=0.0, max_freq=6.0
        )

        prior_params = bf.perturb_params(true_params, prior_params_std)
        start_time = time.time()
        posterior_params = bf.compute_posterior_params(
            freq_data=freq_data,
            admittance_matrix=admittance_matrix.AdmittanceMatrix(),
            prior_params=prior_params,
            prior_params_std=prior_params_std
        )
        end_time = time.time()
        optimization_time[snr_idx] = end_time - start_time
        priors[snr_idx] = prior_params
        posteriors[snr_idx] = posterior_params
        print('optimization time =', optimization_time[snr_idx], 'seconds')
        print('######################################################\n')

    print('optimization time mean =', optimization_time.mean(), '(seconds)')
    print('optimization time std  =', optimization_time.std(), '(seconds)')
    utils.plot_params_convergences(
        snrs=snrs,
        prior_values=priors,
        posterior_values=posteriors,
        true_values=true_params,
        # params_names=["D", "E^{'}", "M", r"X_{\! d}^{'}"],
        params_names=["R", "X", "H"],
        ylims=np.array([[0.0, 2.0 * true_params[i]] for i in range(n_params)])
    )


if __name__ == '__main__':
    main()

