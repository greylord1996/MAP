import os
import os.path
import numpy as np
import settings
import data



def _get_data_from_file(data_file, is_complex):
    # Extracts data from given file and returns it as np.array
    data = None
    with open(data_file) as input_file:
        lines = input_file.readlines()
        data = np.zeros(
            len(lines),
            dtype=(np.complex_ if is_complex else np.float)
        )

        for i in range(len(lines)):
            number = lines[i].rstrip().replace(',', '.')
            if is_complex:
                number = number.replace('i', 'j')
                number = number.replace(' ', '')
                data[i] = np.complex(number)
            else:
                data[i] = np.float(number)

    return data



def _get_data_from_files(Vm_file, Va_file, Im_file, Ia_file, is_complex):
    path_to_this_file = os.path.abspath(os.path.dirname(__file__))

    Vm_data = _get_data_from_file(
        data_file=os.path.join(path_to_this_file, '..', 'tests', Vm_file),
        is_complex=is_complex
    )
    Va_data = _get_data_from_file(
        data_file=os.path.join(path_to_this_file, '..', 'tests', Va_file),
        is_complex=is_complex
    )
    Im_data = _get_data_from_file(
        data_file=os.path.join(path_to_this_file, '..', 'tests', Im_file),
        is_complex=is_complex
    )
    Ia_data = _get_data_from_file(
        data_file=os.path.join(path_to_this_file, '..', 'tests', Ia_file),
        is_complex=is_complex
    )

    return {
        'Vm': Vm_data,
        'Va': Va_data,
        'Im': Im_data,
        'Ia': Ia_data
    }



def get_initial_time_data():
    initial_time_data = _get_data_from_files(
        Vm_file=os.path.join('initial_time_data', 'Vm_time_data.txt'),
        Va_file=os.path.join('initial_time_data', 'Va_time_data.txt'),
        Im_file=os.path.join('initial_time_data', 'Im_time_data.txt'),
        Ia_file=os.path.join('initial_time_data', 'Ia_time_data.txt'),
        is_complex=False
    )

    assert len(initial_time_data['Vm']) == 2001
    assert len(initial_time_data['Va']) == 2001
    assert len(initial_time_data['Im']) == 2001
    assert len(initial_time_data['Ia']) == 2001

    return initial_time_data



def get_time_data_after_snr():
    time_data_after_snr = _get_data_from_files(
        Vm_file=os.path.join('time_data_after_snr', 'Vm_snr_time_data.txt'),
        Va_file=os.path.join('time_data_after_snr', 'Va_snr_time_data.txt'),
        Im_file=os.path.join('time_data_after_snr', 'Im_snr_time_data.txt'),
        Ia_file=os.path.join('time_data_after_snr', 'Ia_snr_time_data.txt'),
        is_complex=False
    )

    assert len(time_data_after_snr['Vm']) == 2001
    assert len(time_data_after_snr['Va']) == 2001
    assert len(time_data_after_snr['Im']) == 2001
    assert len(time_data_after_snr['Ia']) == 2001

    return time_data_after_snr



def get_freq_data_after_fft():
    freq_data_after_fft = _get_data_from_files(
        Vm_file=os.path.join('freq_data_after_fft', 'Vm_freq_data.txt'),
        Va_file=os.path.join('freq_data_after_fft', 'Va_freq_data.txt'),
        Im_file=os.path.join('freq_data_after_fft', 'Im_freq_data.txt'),
        Ia_file=os.path.join('freq_data_after_fft', 'Ia_freq_data.txt'),
        is_complex=True
    )

    assert len(freq_data_after_fft['Vm']) == 1001
    assert len(freq_data_after_fft['Va']) == 1001
    assert len(freq_data_after_fft['Im']) == 1001
    assert len(freq_data_after_fft['Ia']) == 1001

    return freq_data_after_fft



def get_freq_data_after_trim():
    freq_data_after_trim = _get_data_from_files(
        Vm_file=os.path.join('freq_data_after_trim', 'y_Vm.txt'),
        Va_file=os.path.join('freq_data_after_trim', 'y_Va.txt'),
        Im_file=os.path.join('freq_data_after_trim', 'y_Im.txt'),
        Ia_file=os.path.join('freq_data_after_trim', 'y_Ia.txt'),
        is_complex=True
    )

    assert len(freq_data_after_trim['Vm']) == 600
    assert len(freq_data_after_trim['Va']) == 600
    assert len(freq_data_after_trim['Im']) == 600
    assert len(freq_data_after_trim['Ia']) == 600

    return freq_data_after_trim



def get_freq_data_before_stage1():
    freq_data_before_stage1 = _get_data_from_files(
        Vm_file=os.path.join('S1_data_vectors', 'y_VmS1.txt'),
        Va_file=os.path.join('S1_data_vectors', 'y_VaS1.txt'),
        Im_file=os.path.join('S1_data_vectors', 'y_ImS1.txt'),
        Ia_file=os.path.join('S1_data_vectors', 'y_IaS1.txt'),
        is_complex=True
    )

    assert len(freq_data_before_stage1['Vm']) == 597
    assert len(freq_data_before_stage1['Va']) == 597
    assert len(freq_data_before_stage1['Im']) == 597
    assert len(freq_data_before_stage1['Ia']) == 597

    return freq_data_before_stage1





GP = settings.GeneratorParameters(  # true generator parameters
    d_2=0.25,
    e_2=1.0,
    m_2=1.0,
    x_d2=0.01,
    ic_d2=1.0
)



print('======================================')
initial_time_data = get_initial_time_data()
print('Vm_initial_time_data =', initial_time_data['Vm'])
print('Va_initial_time_data =', initial_time_data['Va'])
print('Im_initial_time_data =', initial_time_data['Im'])
print('Ia_initial_time_data =', initial_time_data['Ia'])
print('======================================')


print('======================================')
time_data_after_snr = get_time_data_after_snr()
print('Vm_time_data_after_snr =', time_data_after_snr['Vm'])
print('Va_time_data_after_snr =', time_data_after_snr['Im'])
print('Im_time_data_after_snr =', time_data_after_snr['Va'])
print('Ia_time_data_after_snr =', time_data_after_snr['Ia'])
print('======================================')


print('======================================')
freq_data_after_fft = get_freq_data_after_fft()
print('Vm_freq_data_after_fft =', freq_data_after_fft['Vm'])
print('Va_freq_data_after_fft =', freq_data_after_fft['Im'])
print('Im_freq_data_after_fft =', freq_data_after_fft['Va'])
print('Ia_freq_data_after_fft =', freq_data_after_fft['Ia'])
print('======================================')


print('======================================')
freq_data_after_trim = get_freq_data_after_trim()
print('Vm_freq_data_after_trim =', freq_data_after_trim['Vm'])
print('Va_freq_data_after_trim =', freq_data_after_trim['Im'])
print('Im_freq_data_after_trim =', freq_data_after_trim['Va'])
print('Ia_freq_data_after_trim =', freq_data_after_trim['Ia'])
print('======================================')


print('======================================')
freq_data_before_stage1 = get_freq_data_before_stage1()
print('Vm_freq_data_before_stage1 =', freq_data_before_stage1['Vm'])
print('Va_freq_data_before_stage1 =', freq_data_before_stage1['Im'])
print('Im_freq_data_before_stage1 =', freq_data_before_stage1['Va'])
print('Ia_freq_data_before_stage1 =', freq_data_before_stage1['Ia'])
print('======================================')

