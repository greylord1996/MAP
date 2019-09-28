"""Auxiliary functions to make some plots."""


import os
import os.path

import numpy as np
import sympy
import matplotlib.pyplot as plt


def predict_outputs(admittance_matrix, sys_params, freqs, inputs):
    """Calculate outputs based on admittance matrix and inputs."""
    n_inputs = admittance_matrix.data.shape[1]
    n_outputs = admittance_matrix.data.shape[0]
    n_freqs = len(freqs)
    n_params = len(sys_params)
    assert n_inputs == inputs.shape[0]
    assert n_freqs == inputs.shape[1]

    Y = admittance_matrix.data
    computed_Y = np.zeros((n_outputs, n_inputs, n_freqs), dtype=np.complex64)
    for row_idx in range(Y.shape[0]):
        for column_idx in range(Y.shape[1]):
            element_expr = Y[row_idx, :][column_idx]
            for param_idx in range(n_params):
                element_expr = element_expr.subs(
                    admittance_matrix.params[param_idx],
                    sys_params[param_idx]
                )
            computed_Y[row_idx][column_idx] = sympy.lambdify(
                args=admittance_matrix.omega,
                expr=element_expr,
                modules='numexpr'
            )(2.0 * np.pi * freqs)

    predictions = np.zeros((n_outputs, n_freqs), dtype=np.complex64)
    for freq_idx in range(n_freqs):
        predictions[:, freq_idx] = (
            computed_Y[:, :, freq_idx] @ inputs[:, freq_idx]
        )
    return predictions


def plot_measurements_and_predictions(freqs, measurements, predictions,
                                      out_file_name, yscale=None, yticks=None,
                                      xlabel=None, ylabel=None):
    """Plot measured and predicted data."""
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(24, 8))

    plt.plot(freqs, measurements, color='black')
    plt.plot(freqs, predictions, color='blue')

    if yscale is not None:
        plt.yscale(yscale)
    plt.tick_params(
        axis='both', labelsize=60, direction='in', length=12, width=3, pad=12
    )
    plt.yticks(yticks)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=60)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=60)
    plt.legend(
        ['Measured', 'Predicted'],
        loc='upper right', prop={'size': 60}, frameon=True, ncol=2
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', 'predictions_output_data', out_file_name + '.pdf'
        ),
        dpi=180, format='pdf'
    )


def plot_params_convergences(snrs, prior_values, posterior_values,
                             true_values, params_names, ylims):
    """Plot convergence of one parameter for different SNR."""
    assert len(snrs) == len(posterior_values) == len(prior_values)
    assert (prior_values.shape[1] == posterior_values.shape[1]
            == len(true_values) == len(params_names) == len(ylims))
    assert ylims.shape[1] == 2

    n_points = len(snrs)
    n_params = len(true_values)

    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    fig, axes = plt.subplots(n_params, 1, figsize=(24, 12 * n_params))

    for param_idx in range(n_params):
        ax = axes[param_idx]
        ax.plot(
            snrs, prior_values[:, param_idx],
            label='prior', linewidth=4, marker='o', color='b'
        )
        ax.plot(
            snrs, posterior_values[:, param_idx],
            label='posterior', linewidth=4, marker='o', color='g'
        )
        ax.plot(
            snrs, [true_values[param_idx] for _ in range(n_points)],
            label='true', linewidth=4, linestyle='dashed', color='r'
        )

        ax.grid(alpha=0.75)
        ax.tick_params(
            axis='both', labelsize=60, direction='in',
            length=12, width=3, pad=12
        )
        n_ticks = 5
        y_min = ylims[param_idx][0]
        y_max = ylims[param_idx][1]
        step = (y_max - y_min) / (n_ticks - 1)
        ax.set_yticks(np.arange(y_min, y_max + step, step))
        ax.set_ylim(ylims[param_idx])
        ax.set_xticks(range(0, n_points + 1, 5))

        ax.set_xlabel('SNR', fontsize=60)
        param_name = params_names[param_idx]
        ax.set_ylabel('$' + param_name + '$', labelpad=20, fontsize=60)
        ax.legend(loc='upper right', prop={'size': 60}, frameon=True, ncol=3)

    fig.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', 'convergences_different_SNR',
            'params_convergences.pdf'
        ),
        dpi=180, format='pdf'
    )


# def plot_objective_function(obj_func, true_params):
#     """Plot the objective function."""
#     thetas1 = 0.01 * np.arange(start=1, stop=50, step=1)
#     f_values = np.zeros(len(thetas1))
#     for i in range(len(f_values)):
#         f_values[i] = f.compute(np.array([
#             thetas1[i], 1.00, 1.00, 0.01
#         ]))
#     plt.plot(thetas1, f_values, label='SNR=' + str(snr) + ' (original f)')
#
#     plt.xlabel('theta_g1')
#     plt.ylabel('objective function (f)')
#     # plt.xticks([0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5])
#     # plt.gca().get_xticklabels()[3].set_color("red")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(
#             os.path.abspath(os.path.dirname(__file__)),
#             '..', 'graphics', 'vary_theta_g1.pdf'
#         ),
#         dpi=180,
#         format='pdf'
#     )

