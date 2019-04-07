import os
import os.path
import numpy as np
import data


PATH_TO_THIS_FILE = os.path.abspath(os.path.dirname(__file__))



def _get_data_from_file(data_file):
    # Extracts data from given file and returns it as np.array
    data = None
    with open(data_file) as input_file:
        lines = input_file.readlines()
        data = np.zeros(len(lines))
        for i in range(len(lines)):
            data[i] = np.float(lines[i].rstrip().replace(',', '.'))
    return data



def _get_data_from_files(Vm_file, Va_file, Im_file, Ia_file, dtype):
    return dtype(
        Vm_time_data=_get_data_from_file(os.path.join(
            PATH_TO_THIS_FILE, '..', 'tests', Vm_file
        )),
        Va_time_data=_get_data_from_file(os.path.join(
            PATH_TO_THIS_FILE, '..', 'tests', Va_file
        )),
        Im_time_data=_get_data_from_file(os.path.join(
            PATH_TO_THIS_FILE, '..', 'tests', Im_file
        )),
        Ia_time_data=_get_data_from_file(os.path.join(
            PATH_TO_THIS_FILE, '..', 'tests', Ia_file
        )),
        dt=0.05
    )



def get_initial_time_data():
    initial_time_data = _get_data_from_files(
        Vm_file=os.path.join('initial_time_data', 'Vm_time_data.txt'),
        Va_file=os.path.join('initial_time_data', 'Va_time_data.txt'),
        Im_file=os.path.join('initial_time_data', 'Im_time_data.txt'),
        Ia_file=os.path.join('initial_time_data', 'Ia_time_data.txt'),
        dtype=data.TimeData
    )

    assert len(initial_time_data.Vm) == 2001
    assert len(initial_time_data.Va) == 2001
    assert len(initial_time_data.Im) == 2001
    assert len(initial_time_data.Ia) == 2001

    return initial_time_data



def get_time_data_after_snr():
    time_data_after_snr = _get_data_from_files(
        Vm_file=os.path.join('time_data_after_snr', 'Vm_snr_time_data.txt'),
        Va_file=os.path.join('time_data_after_snr', 'Va_snr_time_data.txt'),
        Im_file=os.path.join('time_data_after_snr', 'Im_snr_time_data.txt'),
        Ia_file=os.path.join('time_data_after_snr', 'Ia_snr_time_data.txt'),
        dtype=data.TimeData
    )

    assert len(time_data_after_snr.Vm) == 2001
    assert len(time_data_after_snr.Va) == 2001
    assert len(time_data_after_snr.Im) == 2001
    assert len(time_data_after_snr.Ia) == 2001

    return time_data_after_snr



def get_freq_data_after_fft():
    freq_data_after_fft = _get_data_from_files(
        Vm_file=os.path.join('freq_data_after_fft', 'Vm_freq_data.txt'),
        Va_file=os.path.join('freq_data_after_fft', 'Va_freq_data.txt'),
        Im_file=os.path.join('freq_data_after_fft', 'Im_freq_data.txt'),
        Ia_file=os.path.join('freq_data_after_fft', 'Ia_freq_data.txt'),
        dtype=data.FreqData
    )

    assert len(freq_data_after_fft.Vm) == 1001
    assert len(freq_data_after_fft.Va) == 1001
    assert len(freq_data_after_fft.Im) == 1001
    assert len(freq_data_after_fft.Ia) == 1001

    return freq_data_after_fft



def get_freq_data_after_trim():
    freq_data_after_trim = _get_data_from_files(
        Vm_file=os.path.join('freq_data_after_trim', 'y_Vm.txt'),
        Va_file=os.path.join('freq_data_after_trim', 'y_Va.txt'),
        Im_file=os.path.join('freq_data_after_trim', 'y_Im.txt'),
        Ia_file=os.path.join('freq_data_after_trim', 'y_Ia.txt'),
        dtype=data.FreqData
    )

    assert len(freq_data_after_trim.Vm) == 600
    assert len(freq_data_after_trim.Va) == 600
    assert len(freq_data_after_trim.Im) == 600
    assert len(freq_data_after_trim.Ia) == 600

    return freq_data_after_trim



def get_freq_data_before_stage1():
    freq_data_before_stage1 = _get_data_from_files(
        Vm_file=os.path.join('S1_data_vectors', 'y_VmS1.txt'),
        Va_file=os.path.join('S1_data_vectors', 'y_VaS1.txt'),
        Im_file=os.path.join('S1_data_vectors', 'y_ImS1.txt'),
        Ia_file=os.path.join('S1_data_vectors', 'y_IaS1.txt'),
        dtype=data.FreqData
    )

    assert len(freq_data_before_stage1.Vm) == 597
    assert len(freq_data_before_stage1.Va) == 597
    assert len(freq_data_before_stage1.Im) == 597
    assert len(freq_data_before_stage1.Ia) == 597

    return freq_data_before_stage1
















# import numpy as np
# from settings import GeneratorParameters, PriorData
#
# gp = GeneratorParameters()
# param_dist = 1   # TO SETTINGS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# GP_vec = np.array(gp.get_list_of_values())
#
# print('GP_vec: ', GP_vec)
# print('GP_vec.size: ', GP_vec.size)
# print('type; ', type(GP_vec))
#
# rnd_vec = param_dist * np.random.rand(GP_vec.size, 1) - param_dist/2
# print('rnd_vec: ', rnd_vec)
# rnd_vec = rnd_vec.reshape(-1)
#
# print('rnd_vec: ', rnd_vec)
# print('rnd_vec.size: ', rnd_vec.size)
# print('type_rnd: ', type(rnd_vec))
#
# prior_mean = GP_vec + np.multiply(rnd_vec, GP_vec)
# print('prior_mean: ', prior_mean)
# prior_std = 1000 * np.ones(4)
# print('prior_std: ', prior_std)
#
#
# SNR = 45
# d_coi = 0
#
