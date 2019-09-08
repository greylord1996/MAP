"""Auxiliary functions to make some plots."""


import os
import os.path

import numpy as np
import sympy
import matplotlib.pyplot as plt
import seaborn


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
        axis='both', labelsize=50, direction='in', length=12, width=3, pad=12
    )
    plt.yticks(yticks)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=50)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=50)
    plt.legend(
        ['Measured', 'Predicted'],
        loc='upper right', prop={'size': 50}, frameon=False
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'samples', 'predictions', out_file_name + '.pdf'
        ),
        dpi=180, format='pdf'
    )


def plot_param_convergence(snrs, prior_values, posterior_values, true_value,
                           param_name, ylim):
    """Plot convergence of one parameter for different SNR."""
    assert len(snrs) == len(posterior_values)
    n_points = len(snrs)

    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.figure(figsize=(24, 12))

    plt.plot(
        snrs, prior_values,
        label='prior', linewidth=4, marker='o', color='b'
    )
    plt.plot(
        snrs, posterior_values,
        label='posterior', linewidth=4, marker='o', color='g'
    )
    plt.plot(
        snrs, [true_value for _ in range(n_points)],
        label='true', linewidth=4, linestyle='dashed', color='r'
    )

    plt.tick_params(
        axis='both', labelsize=50, direction='in', length=12, width=3, pad=12
    )
    n_ticks = 5
    step = (ylim[1] - ylim[0]) / (n_ticks - 1)
    plt.yticks(np.arange(ylim[0], ylim[1], step))
    plt.ylim(ylim)

    plt.xlabel('SNR', fontsize=50)
    plt.ylabel('$' + param_name + '$', fontsize=50)
    plt.legend(loc='upper right', prop={'size': 50}, frameon=True, ncol=3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'samples', 'convergences_different_SNR', param_name + '.pdf'
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
#             '..', 'samples', 'vary_theta_g1.pdf'
#         ),
#         dpi=180,
#         format='pdf'
#     )

