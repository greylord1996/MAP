import time

import numpy as np

from generator import admittance_matrix
from generator import dynamic_equations_to_simulate
import bayesian_framework as bf
import data
import utils


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

    true_params = np.array([0.25, 1.00, 1.00, 0.01])
    n_params = len(true_params)
    prior_params_std = np.array([0.5 for _ in range(n_params)])

    snrs = 1.0 * np.arange(1, 26, 1)
    priors = np.zeros((len(snrs), n_params))
    posteriors = np.zeros((len(snrs), n_params))
    for snr_idx in range(len(snrs)):
        freq_data = bf.prepare_data(
            time_data=time_data, snr=snrs[snr_idx],
            remove_zero_freq=True, min_freq=0.0, max_freq=6.0
        )

        prior_params = bf.perturb_params(true_params, prior_params_std)

        prior_predictions = utils.predict_outputs(
            admittance_matrix.AdmittanceMatrix(),
            prior_params,
            freq_data.freqs,
            freq_data.inputs
        )
        utils.plot_measurements_and_predictions(
            freq_data.freqs[:250],
            (np.abs(freq_data.outputs[0][:250]))**2,
            (np.abs(prior_predictions[0][:250]))**2,
            'prior_SNR=' + str(snrs[snr_idx]),
            yscale='log', yticks=[0.001, 0.000001],
            xlabel=None, ylabel=r'$\tilde{\mathrm{I}}$ PSD'
        )

        start_time = time.time()
        posterior_params = bf.compute_posterior_params(
            freq_data=freq_data,
            admittance_matrix=admittance_matrix.AdmittanceMatrix(),
            prior_params=prior_params,
            prior_params_std=prior_params_std
        )
        end_time = time.time()
        optimization_time = end_time - start_time
        print('optimization time =', round(optimization_time, 1), 'seconds')

        posterior_predictions = utils.predict_outputs(
            admittance_matrix.AdmittanceMatrix(),
            posterior_params,
            freq_data.freqs,
            freq_data.inputs
        )
        assert len(freq_data.freqs) == 600
        utils.plot_measurements_and_predictions(
            freq_data.freqs[:250],
            (np.abs(freq_data.outputs[0][:250]))**2,
            (np.abs(posterior_predictions[0][:250]))**2,
            'posterior_SNR=' + str(snrs[snr_idx]),
            yscale='log', yticks=[0.001, 0.000001],
            xlabel='Frequency (Hz)', ylabel=r'$\tilde{\mathrm{I}}$ PSD'
        )

        priors[snr_idx] = prior_params
        posteriors[snr_idx] = posterior_params

    utils.plot_params_convergences(
        snrs=snrs,
        prior_values=priors,
        posterior_values=posteriors,
        true_values=true_params,
        params_names=["D", "E^{'}", "M", r"X_{\! d}^{'}"],
        ylims=np.array([[0.0, 2.0 * true_params[i]] for i in range(n_params)])
    )


if __name__ == '__main__':
    main()

