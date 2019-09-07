import os
import os.path
import sys
import json

import numpy as np

# directory with source code
PATH_TO_THIS_FILE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PATH_TO_THIS_FILE, '..', 'src'))
import data


def _get_data_from_file(data_file, is_complex):
    # Extract data from given file and return it as np.array
    data_array = None
    with open(data_file, 'r') as input_file:
        lines = input_file.readlines()
        data_array = np.zeros(
            len(lines),
            dtype=(np.complex if is_complex else np.float)
        )

        for i in range(len(lines)):
            number = lines[i].rstrip().replace(',', '.')
            if is_complex:
                number = number.replace('i', 'j')
                number = number.replace(' ', '')
                data_array[i] = np.complex(number)
            else:
                data_array[i] = np.float(number)

    return data_array


def _get_data_from_files(Vm_file, Va_file, Im_file, Ia_file, is_complex):
    # Just call 4 times _get_data_from_file to get Vm, Va, Im, Ia
    Vm_data = _get_data_from_file(
        data_file=os.path.join(PATH_TO_THIS_FILE, Vm_file),
        is_complex=is_complex
    )
    Va_data = _get_data_from_file(
        data_file=os.path.join(PATH_TO_THIS_FILE, Va_file),
        is_complex=is_complex
    )
    Im_data = _get_data_from_file(
        data_file=os.path.join(PATH_TO_THIS_FILE, Im_file),
        is_complex=is_complex
    )
    Ia_data = _get_data_from_file(
        data_file=os.path.join(PATH_TO_THIS_FILE, Ia_file),
        is_complex=is_complex
    )

    return {
        'Vm': Vm_data,
        'Va': Va_data,
        'Im': Im_data,
        'Ia': Ia_data
    }


def _get_freqs_from_file(test_dir, preparation_stage):
    # Extract frequency points from file and return it as np.array
    path_to_freqs_file = os.path.join(
        test_dir, 'freq_data', preparation_stage, 'freqs.txt'
    )
    return _get_data_from_file(
        data_file=path_to_freqs_file,
        is_complex=False
    )


def get_initial_time_data(test_dir):
    """Return correct data in time domain after simulation.

    Args:
        test_dir (str): Name of a directory specifying a test set.

    Returns:
        initial_time_data (dict): contains 4 key-value pairs:
            'Vm': np.array of voltage magnitudes in time domain;
            'Va': np.array of voltage phases in time domain;
            'Im': np.array of current magnitudes in time domain;
            'Ia': np.array of current phases in time domain.
    """
    initial_time_data = _get_data_from_files(
        Vm_file=os.path.join(test_dir, 'time_data', 'initial', 'Vm.txt'),
        Va_file=os.path.join(test_dir, 'time_data', 'initial', 'Va.txt'),
        Im_file=os.path.join(test_dir, 'time_data', 'initial', 'Im.txt'),
        Ia_file=os.path.join(test_dir, 'time_data', 'initial', 'Ia.txt'),
        is_complex=False
    )

    # These asserts have to be removed after extension of test sets
    assert len(initial_time_data['Vm']) == 2001
    assert len(initial_time_data['Va']) == 2001
    assert len(initial_time_data['Im']) == 2001
    assert len(initial_time_data['Ia']) == 2001

    return initial_time_data


def get_time_data_after_snr(test_dir):
    """Return data in time domain after applying SNR to simulated time data.

    Args:
        test_dir (str): Name of a directory specifying a test set.

    Returns:
        time_data_after_snr (dict): contains 4 key-value pairs:
            'Vm': np.array of voltage magnitudes in time domain;
            'Va': np.array of voltage phases in time domain;
            'Im': np.array of current magnitudes in time domain;
            'Ia': np.array of current phases in time domain.
    """
    time_data_after_snr = _get_data_from_files(
        Vm_file=os.path.join(
            test_dir, 'time_data', 'after_snr', 'Vm_PMUn.txt'
        ),
        Va_file=os.path.join(
            test_dir, 'time_data', 'after_snr', 'Va_PMUn.txt'
        ),
        Im_file=os.path.join(
            test_dir, 'time_data', 'after_snr', 'Im_PMUn.txt'
        ),
        Ia_file=os.path.join(
            test_dir, 'time_data', 'after_snr', 'Ia_PMUn.txt'
        ),
        is_complex=False
    )

    # These asserts have to be removed after extension of test sets
    assert len(time_data_after_snr['Vm']) == 2001
    assert len(time_data_after_snr['Va']) == 2001
    assert len(time_data_after_snr['Im']) == 2001
    assert len(time_data_after_snr['Ia']) == 2001

    return time_data_after_snr


def get_freq_data_after_fft_and_trimming(test_dir):
    """Return data after applying FFT to time data and trimming it.

    Args:
        test_dir (str): Name of a directory specifying a test set.

    Returns:
        freq_data_after_fft (dict): contains 5 key-value pairs:
            'freqs': np.array of frequency points;
            'Vm': np.array of voltage magnitudes in frequency domain;
            'Va': np.array of voltage phases in frequency domain;
            'Im': np.array of current magnitudes in frequency domain;
            'Ia': np.array of current phases in frequency domain.
    """
    freq_data_after_fft = _get_data_from_files(
        Vm_file=os.path.join(test_dir, 'freq_data', 'after_fft_and_trimming', 'y_Vm.txt'),
        Va_file=os.path.join(test_dir, 'freq_data', 'after_fft_and_trimming', 'y_Va.txt'),
        Im_file=os.path.join(test_dir, 'freq_data', 'after_fft_and_trimming', 'y_Im.txt'),
        Ia_file=os.path.join(test_dir, 'freq_data', 'after_fft_and_trimming', 'y_Ia.txt'),
        is_complex=True
    )
    freq_data_after_fft['freqs'] = _get_freqs_from_file(test_dir, 'after_fft_and_trimming')

    # These asserts have to be removed after extension of test sets
    assert len(freq_data_after_fft['freqs']) == 600
    assert len(freq_data_after_fft['Vm']) == 600
    assert len(freq_data_after_fft['Va']) == 600
    assert len(freq_data_after_fft['Im']) == 600
    assert len(freq_data_after_fft['Ia']) == 600

    return freq_data_after_fft


def get_freq_data_after_remove_fo_band(test_dir):
    """Return data in frequency domain after removing forced oscillation band.

    Args:
        test_dir (str): Name of a directory specifying a test set.

    Returns:
        freq_data_after_remove_fo_band (dict): contains 5 key-value pairs:
            'freqs': np.array of frequency points;
            'Vm': np.array of voltage magnitudes in frequency domain;
            'Va': np.array of voltage phases in frequency domain;
            'Im': np.array of current magnitudes in frequency domain;
            'Ia': np.array of current phases in frequency domain.
    """
    freq_data_after_remove_fo_band = _get_data_from_files(
        Vm_file=os.path.join(
            test_dir, 'freq_data', 'after_remove_fo_band', 'y_VmS1.txt'
        ),
        Va_file=os.path.join(
            test_dir, 'freq_data', 'after_remove_fo_band', 'y_VaS1.txt'
        ),
        Im_file=os.path.join(
            test_dir, 'freq_data', 'after_remove_fo_band', 'y_ImS1.txt'
        ),
        Ia_file=os.path.join(
            test_dir, 'freq_data', 'after_remove_fo_band', 'y_IaS1.txt'
        ),
        is_complex=True
    )
    freq_data_after_remove_fo_band['freqs'] = (
        _get_freqs_from_file(test_dir, 'after_remove_fo_band')
    )

    # These asserts have to be removed after extension of test sets
    assert len(freq_data_after_remove_fo_band['freqs']) == 597
    assert len(freq_data_after_remove_fo_band['Vm']) == 597
    assert len(freq_data_after_remove_fo_band['Va']) == 597
    assert len(freq_data_after_remove_fo_band['Im']) == 597
    assert len(freq_data_after_remove_fo_band['Ia']) == 597

    return freq_data_after_remove_fo_band


def get_correct_values(test_dir):
    """Return some correct values from file correct_values.json as dict.

    Some correct values represent values which are not big arrays.
    For example, standard deviations of Vm after applying FFT is just
    a single number. This and other correct values are stored
    in the test_dir/correct_values.json file.

    Args:
        test_dir (str): Name of a directory specifying a test set.

    Returns:
        json_correct_values (dict): dict with nested dicts containing
            correct values extracted from correct_values.json.
    """
    path_to_this_file = os.path.abspath(os.path.dirname(__file__))
    path_to_correct_values_file = os.path.join(
        path_to_this_file, test_dir, 'correct_values.json'
    )
    json_correct_values = None
    with open(path_to_correct_values_file, 'r') as correct_values_file:
        json_correct_values = json.load(correct_values_file)
    return json_correct_values


def get_prepared_freq_data(test_dir):
    """Return correct data which should be just before running stage 1.

    Args:
        test_dir (str): Name of a directory specifying a test set.

    Returns:
        prepared_freq_data (class FreqData): Data in frequency domain
            which can be used immediately for minimization
            of the objective function.
    """
    all_correct_values = get_correct_values(test_dir)
    correct_std_deviations = all_correct_values['freq_data_std_dev_eps']
    prepared_freq_data_dict = get_freq_data_after_remove_fo_band(test_dir)

    # It doesn't matter initial values of aux_time_data and prepared_freq_data.
    # aux_time_data is used only for creating an instance of data.FreqData.
    # We will overwrite all attributes of prepared_freq_data below
    # in this function.
    aux_time_data = data.TimeData(
        inputs=np.array([[0.0 for _ in range(567)], [0.0 for _ in range(567)]]),
        outputs=np.array([[0.0 for _ in range(567)], [0.0 for _ in range(567)]]),
        dt=1.23456789
    )
    aux_time_data.apply_white_noise(snr=0.0)
    prepared_freq_data = data.FreqData(aux_time_data)

    prepared_freq_data.freqs = prepared_freq_data_dict['freqs']
    prepared_freq_data.inputs = np.array([
        prepared_freq_data_dict['Vm'],
        prepared_freq_data_dict['Va']
    ])
    prepared_freq_data.outputs = np.array([
        prepared_freq_data_dict['Im'],
        prepared_freq_data_dict['Ia']
    ])

    prepared_freq_data.input_std_devs[0] = correct_std_deviations['std_dev_eps_Vm']
    prepared_freq_data.input_std_devs[1] = correct_std_deviations['std_dev_eps_Va']
    prepared_freq_data.output_std_devs[0] = correct_std_deviations['std_dev_eps_Im']
    prepared_freq_data.output_std_devs[1] = correct_std_deviations['std_dev_eps_Ia']

    return prepared_freq_data

