import numpy as np
from generator import admittance_matrix
from generator import dynamic_equations_to_simulate

import bayesian_framework as bf
import data
# import utils


def get_time_data():
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


def main():
    time_data = get_time_data()
    freq_data = bf.prepare_data(
        time_data=time_data, snr=45.0,
        remove_zero_freq=True, min_freq=0.0, max_freq=6.0
    )

    true_params = np.array([0.25, 1.00, 1.00, 0.01])
    prior_params_std = np.array([0.5, 0.5, 0.5, 0.5])
    prior_params = bf.perturb_params(true_params, prior_params_std)
    print('PRIOR PARAMS =', prior_params)
    print('PRIOR PARAMS STD =', prior_params_std)

    # plot before parameters clarification
    # utils.plot_Im_psd(stage2_data, gen_params_prior_mean.as_array, is_xlabel=False)

    posterior_params = bf.compute_posterior_params(
        freq_data=freq_data,
        admittance_matrix=admittance_matrix.AdmittanceMatrix(),
        prior_params=prior_params,
        prior_params_std=prior_params_std
    )
    print('POSTERIOR PARAMS:', posterior_params)

    # plot after parameters clarification
    # utils.plot_Im_psd(stage2_data, posterior_gen_params, is_xlabel=True)
    # utils.plot_all_params_convergences(
    #     param_names=["D", "E^{'}", "M", r"X_{\! d}^{'}"],
    #     snrs=snrs,
    #     priors=priors,
    #     posteriors=posteriors,
    #     true_values=[0.25, 1.00, 1.00, 0.01]
    # )


if __name__ == '__main__':
    main()

