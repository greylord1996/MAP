"""Bayesian framework for dynamical system parameters identification.

This module contains all functions which should be used by a user of
this framework. It is not necessary to import other modules
(except for utils.py).

The typical baseline structures as follows:
a user collects data in time domain, transforms it to frequency domain
by employing 'prepare_data' function and obtains posterior parameters
of an abstract dynamical system via 'compute_posterior_params'.
To use the latter function it is necessary to define your own class
representing the admittance matrix of your system (see the paper).
The format of the admittance matrix is fixed. See examples
for more details.

"""

import copy

import numpy as np

import objective_function
import data
import optimize


def perturb_params(true_params, dev_fractions):
    """Perturb true parameters of a dynamical system.

    Given true values of parameters, this function perturbs them
    by adding a random value sampled from uniform distribution.
    This uniform distribution is at [-dev_fraction; dev_fraction].

    Args:
        true_params (numpy.ndarray): True parameters of a system.
        dev_fractions (numpy.ndarray): Uncertainties in parameters
            (for example, 0.5 means 50% uncertainty).

    Returns:
        perturbed_params (numpy.ndarray): Perturbed parameters
            of a dynamical system.
    """
    if len(true_params.shape) != 1 or len(dev_fractions.shape) != 1:
        raise ValueError('Arguments must be one-dimensional numpy arrays.')
    if true_params.shape != dev_fractions.shape:
        raise ValueError('Number of system parameters is not equal'
                         'to number of deviation fractions.')

    perturbations = np.random.uniform(low=-dev_fractions, high=dev_fractions)
    perturbed_params = (1.0 + perturbations) * true_params

    # Just for testing -- remove in release
    # assert len(true_params) == 4
    # perturbed_params[0] = 0.206552362540141
    # perturbed_params[1] = 0.837172184078094
    # perturbed_params[2] = 0.665441037484483
    # perturbed_params[3] = 0.00771416811078329

    return perturbed_params


def prepare_data(time_data, snr=None, remove_zero_freq=True,
                 min_freq=None, max_freq=None):
    """Transform data from time domain to frequency domain.

    Args:
        time_data (TimeData): The object containing data in time domain.
        snr (double, optional): The value of SNR specifying noise
            which will be applied to data in time domain.
            If None, there will be no applying of any noise.
        remove_zero_freq (bool, optional): Whether to remove
            the constant components from the given data
            (corresponding to zero frequency).
        min_freq (double, optional): The left border of analyzing data
            in frequency domain. Defaults to None that is equivalent to 0.
        max_freq (double, optional): The right border of analyzing data
            in frequency domain. Defaults to None which means that
            all frequencies will be used.

    Returns:
        freq_data (FreqData): Data after transformation from time domain
            to frequency domain.
    """
    if snr < 0.0:
        raise ValueError('SNR can not be negative.')
    if min_freq < 0.0:
        raise ValueError('min_freq can not be negative.')
    if min_freq > max_freq:
        raise ValueError('min_freq must be less than max_freq.')

    time_data_copy = copy.deepcopy(time_data)

    if snr is not None:
        time_data_copy.apply_white_noise(snr)
    elif time_data.input_std_devs is None or time_data.output_std_devs is None:
        raise ValueError('Noise is not specified.')

    freq_data = data.FreqData(time_data_copy, remove_zero_freq)
    freq_data.trim(min_freq, max_freq)
    return freq_data


def compute_posterior_params(freq_data, admittance_matrix,
                             prior_params, prior_params_std):
    """Calculate posterior parameters employing Bayesian approach.

    Args:
        freq_data (FreqData): Data after transformation from time domain
            to frequency domain produced by the 'prepare_data' function.
        admittance_matrix (AdmittanceMatrix): User-defined class
            representing an admittance matrix of a dynamical system.
        prior_params (numpy.ndarray): Prior parameters of a system.
        prior_params_std (numpy.ndarray): Prior uncertainties in
            system parameters (see the 'perturb_params' function).

    Returns:
        posterior_params (numpy.ndarray): Posterior parameters
            of a dynamical system calculated by employing Bayesian
            approach and special optimization routine.
    """
    if (len(freq_data.outputs) != admittance_matrix.data.shape[0] or
            len(freq_data.inputs) != admittance_matrix.data.shape[1]):
        raise ValueError('Inconsistent shapes of data and admittance matrix.')
    if len(prior_params.shape) != 1 or len(prior_params_std.shape) != 1:
        raise ValueError('Prior parameters and deviations'
                         'must be one-dimensional numpy arrays.')
    if prior_params.shape != prior_params_std.shape:
        raise ValueError('Number of system parameters is not equal'
                         'to number of deviation fractions.')

    obj_func = objective_function.ObjectiveFunction(
        freq_data=freq_data,
        admittance_matrix=admittance_matrix,
        prior_params=prior_params,
        prior_params_std=prior_params_std
    )

    print('\n######################################################')
    print('### DEBUG: OPTIMIZATION ROUTINE IS STARTING NOW!!! ###')
    print('######################################################\n')
    posterior_params = optimize.minimize(func=obj_func, x0=prior_params)
    print('f(true_params) =', obj_func.compute(np.array([0.25, 1.0, 1.0, 0.01])))
    print('\n######################################################')
    return posterior_params

