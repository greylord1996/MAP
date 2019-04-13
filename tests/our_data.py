import sys
import os
import os.path
import json

import numpy as np

# directory with source code
sys.path.append(os.path.join(
    os.path.abspath(os.path.dirname(__file__)), '..', 'src')
)

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
    path_to_this_file = os.path.abspath(os.path.dirname(__file__))
    path_to_params_file = os.path.join(
        path_to_this_file, test_dir, 'initial_parameters.json'
    )
    json_initial_params = None
    with open(path_to_params_file, 'r') as params_file:
        json_initial_params = json.load(params_file)

    initial_params = settings.Settings(json_initial_params)
    return initial_params



def get_initial_freq_data(test_dir):
    """Returns data in frequency domain after applying FFT.

    Some unit tests require data in frequency domain. But such data can be
    constructed only from an instance of data.TimeData.
    This function gets correct data in time domain after applying SNR,
    constructs an instance of data.FreqData from these data and returns
    obtained data in frequency domain (as an instance of data.FreqData).

    Args:
        test_dir (str): name of a directory specifying a test set

    Returns:
         freq_data (data.FreqData): not-modified data in frequency domain
    """
    initial_params = get_initial_params(test_dir)
    time_data_after_snr_as_dict = (
        correct_data.get_time_data_after_snr(test_dir)
    )
    time_data_after_snr = data.TimeData(
        Vm_time_data=time_data_after_snr_as_dict['Vm'],
        Va_time_data=time_data_after_snr_as_dict['Va'],
        Im_time_data=time_data_after_snr_as_dict['Im'],
        Ia_time_data=time_data_after_snr_as_dict['Ia'],
        dt=initial_params.integration_settings.dt_step
    )

    snr = 45.0  # should be in initial_params.json?
    d_coi = 0.0  # should be in initial_params.json?
    time_data_after_snr.std_dev_Vm = (
        np.std(time_data_after_snr.Vm, ddof=1) / (10.0**(snr/20.0))
    )
    time_data_after_snr.std_dev_Im = (
        np.std(time_data_after_snr.Im, ddof=1) / (10.0**(snr/20.0))
    )
    time_data_after_snr.std_dev_Va = (
        np.std(time_data_after_snr.Va - d_coi, ddof=1) / (10.0**(snr/20.0))
    )
    time_data_after_snr.std_dev_Ia = (
        np.std(time_data_after_snr.Ia - d_coi, ddof=1) / (10.0**(snr/20.0))
    )

    freq_data = data.FreqData(time_data_after_snr)
    return freq_data



def get_prepared_freq_data(test_dir):
    """Returns data in frequency domain which are ready for stage 1.

    Some unit tests require prepared data in frequency domain
    which can be immediately passed to an optimization routine.

    Args:
        test_dir (str): name of a directory specifying a test set

    Returns:
         freq_data (data.FreqData): prepared data for an optimization routine
    """
    initial_params = get_initial_params(test_dir)
    freq_data = get_initial_freq_data(test_dir)
    freq_data.remove_zero_frequency()
    freq_data.trim(
        min_freq=0.0,
        max_freq=initial_params.freq_data.max_freq
    )
    freq_data.remove_data_from_fo_band(
        min_fo_freq=initial_params.freq_data.lower_fb,
        max_fo_freq=initial_params.freq_data.upper_fb
    )
    return freq_data

