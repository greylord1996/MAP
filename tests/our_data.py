import sys
import os
import os.path
import json

import numpy as np

# directory with source code
PATH_TO_THIS_FILE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PATH_TO_THIS_FILE, '..', 'src'))
from generator import dynamic_equations_to_simulate
import data
import correct_data


def get_initial_params(test_dir):
    """Load initial parameters from a json file.

    Construct an absolute path to the file
    'test_dir/initial_parameters.json' and return a bunch of initial
    parameters.

    Args:
        test_dir (str): Name of a directory specifying a test set.

    Returns:
        initial_params (dict): Initial parameters of the given test set
            (which is specified by the 'test_dir' argument).
    """
    path_to_params_file = os.path.join(
        PATH_TO_THIS_FILE, test_dir, 'initial_parameters.json'
    )
    json_initial_params = None
    with open(path_to_params_file, 'r') as params_file:
        json_initial_params = json.load(params_file)
    return json_initial_params



def get_initial_time_data(test_dir):
    """Return our data in time domain after simulation.

    Args:
        test_dir (str): Name of a directory specifying a test set.

    Returns:
        initial_time_data (class TimeData): Our initial data.
    """
    initial_params = get_initial_params(test_dir)
    ode_solver_object = dynamic_equations_to_simulate.OdeSolver(
        noise=initial_params['Noise'],
        gen_param=initial_params['GeneratorParameters'],
        osc_param=initial_params['OscillationParameters'],
        integr_param=initial_params['IntegrationSettings']
    )
    ode_solver_object.simulate_time_data()
    inputs = np.array([ode_solver_object.Vc1_abs, ode_solver_object.Vc1_angle])
    outputs = np.array([ode_solver_object.Ig_abs, ode_solver_object.Ig_angle])
    return data.TimeData(inputs, outputs, dt=ode_solver_object.dt)



def get_time_data_after_snr(test_dir):
    """Return our data after applying SNR to simulated time data.

    Args:
        test_dir (str): Name of a directory specifying a test set.

    Returns:
        our_time_data (class TimeData): Data in time domain
            after applying AWGN to correct initial data.
    """
    correct_initial_time_data_dict = (
        correct_data.get_initial_time_data(test_dir)
    )
    inputs = np.array([
        correct_initial_time_data_dict['Vm'],
        correct_initial_time_data_dict['Va']
    ])
    outputs = np.array([
        correct_initial_time_data_dict['Im'],
        correct_initial_time_data_dict['Ia']
    ])
    our_time_data = data.TimeData(
        inputs, outputs,
        dt=get_initial_params(test_dir)['IntegrationSettings']['dt_step']
    )

    # note: applying white noise includes generation random numbers
    our_time_data.apply_white_noise(snr=45.0)
    return our_time_data


def get_freq_data_after_fft_and_trimming(test_dir):
    """Return our data in frequency domain after applying FFT.

    Args:
        test_dir (str): Name of a directory specifying a test set.

    Returns:
        freq_data_after_fft (class FreqData): Data in frequency domain
            after applying FFT to correct data in time domain.
    """
    our_time_data = get_time_data_after_snr(test_dir)  # init std_deviations
    correct_time_data_dict = correct_data.get_time_data_after_snr(test_dir)

    our_time_data.inputs[0] = correct_time_data_dict['Vm']
    our_time_data.inputs[1] = correct_time_data_dict['Va']
    our_time_data.outputs[0] = correct_time_data_dict['Im']
    our_time_data.outputs[1] = correct_time_data_dict['Ia']

    our_freq_data = data.FreqData(our_time_data, remove_zero_freq=True)
    our_freq_data.trim(min_freq=0.0, max_freq=6.0)
    return our_freq_data


def get_freq_data_after_remove_fo_band(test_dir):
    """Return data in frequency domain after removing forced oscillation band.

    Return correct data which should be just before running stage 1
    (stage 1 is optimizing uncertain generator parameters).

    Args:
        test_dir (str): Name of a directory specifying a test set.

    Returns:
        our_freq_data (class FreqData): Data in frequency domain
            just before running stage 1 of optimization.
    """
    our_freq_data = get_freq_data_after_fft_and_trimming(test_dir)
    correct_freq_data_dict = (
        correct_data.get_freq_data_after_fft_and_trimming(test_dir)
    )

    our_freq_data.freqs = correct_freq_data_dict['freqs']
    our_freq_data.inputs[0] = correct_freq_data_dict['Vm']
    our_freq_data.inputs[1] = correct_freq_data_dict['Va']
    our_freq_data.outputs[0] = correct_freq_data_dict['Im']
    our_freq_data.outputs[1] = correct_freq_data_dict['Ia']

    all_correct_values = correct_data.get_correct_values(test_dir)
    correct_std_deviations = all_correct_values['freq_data_std_dev_eps']
    our_freq_data.input_std_devs[0] = correct_std_deviations['std_dev_eps_Vm']
    our_freq_data.input_std_devs[1] = correct_std_deviations['std_dev_eps_Va']
    our_freq_data.output_std_devs[0] = correct_std_deviations['std_dev_eps_Im']
    our_freq_data.output_std_devs[1] = correct_std_deviations['std_dev_eps_Ia']

    initial_params = get_initial_params(test_dir)
    our_freq_data.remove_band(
        min_freq=initial_params['FreqData']['lower_fb'],
        max_freq=initial_params['FreqData']['upper_fb']
    )
    return our_freq_data

