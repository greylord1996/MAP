import os
import os.path

import numpy as np
import sympy
import matplotlib.pyplot as plt
import seaborn

import data
import admittance_matrix
import objective_function
import baseline



def plot_objective_function(initial_params):
    """Plots an objective for different SNR."""
    # Probably, you will need to comment or remove @utils.singleton
    # before classes ResidualVector, CovarianceMatrix and ObjectiveFunction
    initial_snr = initial_params.noise.snr
    SNRS = (1, 3,)
    for snr in SNRS:
        # WARNING! It affects the original initial_params
        initial_params.noise.snr = snr

        data_holder = data.DataHolder(initial_params)
        stage2_data = data_holder.get_data(remove_fo_band=False)

        # Perturb generator parameters (replace true parameters with prior)
        gen_params_prior_mean, gen_params_prior_std_dev = (
            baseline.perturb_gen_params(initial_params.generator_parameters)
        )  # now generator parameters are perturbed and uncertain

        f = objective_function.ObjectiveFunction(
            freq_data=stage2_data,
            gen_params_prior_mean=gen_params_prior_mean,
            gen_params_prior_std_dev=gen_params_prior_std_dev
        )

        thetas1 = 0.01 * np.arange(start=-175, stop=225, step=2)
        f_values = np.zeros(len(thetas1))

        for i in range(len(f_values)):
            f_values[i] = f.compute_from_array([thetas1[i], 1.00, 1.00, 0.01])

        plt.plot(thetas1, f_values, label='SNR=' + str(snr) + ' (original f)')

        # for i in range(len(f_values)):
        #     f_values[i] = f.compute_from_array([4*thetas1[i], 1.00, 1.00, 0.01])
        #
        # plt.plot(thetas1, f_values, label='SNR=' + str(snr) + ' (scaled f)')

    plt.xlabel('theta_g1')
    plt.ylabel('objective function (f)')
    plt.xticks([-2.0, -1.0, 0.25, 1.0, 2.0, 3.0])
    plt.gca().get_xticklabels()[2].set_color("red")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'samples', 'vary_theta_g4_1.pdf'
        ),
        dpi=180,
        format='pdf'
    )
    initial_params.noise.snr = initial_snr



def plot_Im_psd(freq_data, gen_params, is_xlabel):
    """Plots measured and predicted Im PSD."""
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
    plt.legend(['Measured', 'Predicted'], loc='upper right', prop={'size': 50}, frameon=False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'samples', plot_name + '.pdf'
        ),
        dpi=180, format='pdf'
    )

