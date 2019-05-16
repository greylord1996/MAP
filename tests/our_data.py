import sys
import os
import os.path
import json

# directory with source code
PATH_TO_THIS_FILE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PATH_TO_THIS_FILE, '..', 'src'))

import dynamic_equations_to_simulate
import data
import settings
import correct_data



def get_initial_params(test_dir):
    """Loads initial parameters from a json file.

    Constructs an absolute path to the file
    'test_dir/initial_parameters.json' and returns a bunch of initial
    parameters as an instance of the class settings.Settings.

    Args:
        test_dir (str): name of a directory specifying a test set

    Returns:
        initial_params (class settings.Settings):
            initial parameters of the given test set
            (which is specified by the 'test_dir' argument)
    """
    path_to_params_file = os.path.join(
        PATH_TO_THIS_FILE, test_dir, 'initial_parameters.json'
    )
    json_initial_params = None
    with open(path_to_params_file, 'r') as params_file:
        json_initial_params = json.load(params_file)

    initial_params = settings.Settings(json_initial_params)
    return initial_params



def get_initial_time_data(test_dir):
    """Returns our data in time domain after simulation.

    Args:
        test_dir (str): name of a directory specifying a test set

    Returns:
        initial_time_data (class TimeData): initial data
            in time domain generated by ourselves
    """
    initial_params = get_initial_params(test_dir)
    solver = dynamic_equations_to_simulate.OdeSolver(
        white_noise=initial_params.white_noise,
        gen_param=initial_params.generator_parameters,
        osc_param=initial_params.oscillation_parameters,
        integr_param=initial_params.integration_settings
    )
    solver.simulate_time_data()
    return data.TimeData(
        Vm_time_data=solver.Vc1_abs,
        Va_time_data=solver.Vc1_angle,
        Im_time_data=solver.Ig_abs,
        Ia_time_data=solver.Ig_angle,
        dt=initial_params.integration_settings.dt_step
    )



def get_time_data_after_snr(test_dir):
    """Returns our data after applying SNR to simulated time data.

    Args:
        test_dir (str): name of a directory specifying a test set

    Returns:
        our_time_data (class TimeData): data in time domain
            after applying AWGN to correct initial data
    """
    correct_initial_time_data_dict = (
        correct_data.get_initial_time_data(test_dir)
    )
    our_time_data = data.TimeData(
        Vm_time_data=correct_initial_time_data_dict['Vm'],
        Va_time_data=correct_initial_time_data_dict['Va'],
        Im_time_data=correct_initial_time_data_dict['Im'],
        Ia_time_data=correct_initial_time_data_dict['Ia'],
        dt=get_initial_params(test_dir).integration_settings.dt_step
    )
    # note: applying white noise includes generation random numbers
    our_time_data.apply_white_noise(snr=45.0, d_coi=0.0)
    return our_time_data



def get_freq_data_after_fft(test_dir):
    """Returns our data in frequency domain after applying FFT.

    Args:
        test_dir (str): name of a directory specifying a test set

    Returns:
        freq_data_after_fft (class FreqData): data in frequency domain
            after applying FFT to correct data in time domain
    """
    our_time_data = get_time_data_after_snr(test_dir)  # init std_deviations
    correct_time_data_dict = correct_data.get_time_data_after_snr(test_dir)

    our_time_data.Vm = correct_time_data_dict['Vm']
    our_time_data.Va = correct_time_data_dict['Va']
    our_time_data.Im = correct_time_data_dict['Im']
    our_time_data.Ia = correct_time_data_dict['Ia']

    our_freq_data = data.FreqData(our_time_data)
    return our_freq_data



def get_freq_data_after_remove_dc_and_trim(test_dir):
    """Returns data in frequency domain after removing DC and trimming.

    Args:
        test_dir (str): name of a directory specifying a test set

    Returns:
        our_freq_data (class FreqData): data in frequency domain
            after removing DC and trimming
            (based on correct initial data in frequency domain)
    """
    our_freq_data = get_freq_data_after_fft(test_dir)
    correct_freq_data_dict = (
        correct_data.get_freq_data_after_remove_dc_and_trim(test_dir)
    )

    our_freq_data.freqs = correct_freq_data_dict['freqs']
    our_freq_data.Vm = correct_freq_data_dict['Vm']
    our_freq_data.Va = correct_freq_data_dict['Va']
    our_freq_data.Im = correct_freq_data_dict['Im']
    our_freq_data.Ia = correct_freq_data_dict['Ia']

    all_correct_values = correct_data.get_correct_values(test_dir)
    correct_std_deviations = all_correct_values['freq_data_std_dev_eps']
    our_freq_data.std_w_Vm = correct_std_deviations['std_dev_eps_Vm']
    our_freq_data.std_w_Va = correct_std_deviations['std_dev_eps_Va']
    our_freq_data.std_w_Im = correct_std_deviations['std_dev_eps_Im']
    our_freq_data.std_w_Ia = correct_std_deviations['std_dev_eps_Ia']

    return our_freq_data



def get_freq_data_after_remove_fo_band(test_dir):
    """Returns data in frequency domain after removing forced oscillation band.

    Returns correct data which should be just before running stage 1
    (stage 1 is optimizing uncertain generator parameters).

    Args:
        test_dir (str): name of a directory specifying a test set

    Returns:
        our_freq_data (class FreqData): data in frequency domain
            just before running stage 1 of optimization
    """
    our_freq_data = get_freq_data_after_remove_dc_and_trim(test_dir)
    correct_freq_data_dict = (
        correct_data.get_freq_data_after_remove_fo_band(test_dir)
    )

    our_freq_data.freqs = correct_freq_data_dict['freqs']
    our_freq_data.Vm = correct_freq_data_dict['Vm']
    our_freq_data.Va = correct_freq_data_dict['Va']
    our_freq_data.Im = correct_freq_data_dict['Im']
    our_freq_data.Ia = correct_freq_data_dict['Ia']

    all_correct_values = correct_data.get_correct_values(test_dir)
    correct_std_deviations = all_correct_values['freq_data_std_dev_eps']
    assert our_freq_data.std_w_Vm == correct_std_deviations['std_dev_eps_Vm']
    assert our_freq_data.std_w_Va == correct_std_deviations['std_dev_eps_Va']
    assert our_freq_data.std_w_Im == correct_std_deviations['std_dev_eps_Im']
    assert our_freq_data.std_w_Ia == correct_std_deviations['std_dev_eps_Ia']

    return our_freq_data
