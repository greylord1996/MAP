import sys
import os
import os.path
import unittest
import numpy as np

sys.path.append(os.path.join(
    os.path.abspath(os.path.dirname(__file__)), '..', 'src')
)
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
        data_file=os.path.join(path_to_this_file, Vm_file),
        is_complex=is_complex
    )
    Va_data = _get_data_from_file(
        data_file=os.path.join(path_to_this_file, Va_file),
        is_complex=is_complex
    )
    Im_data = _get_data_from_file(
        data_file=os.path.join(path_to_this_file, Im_file),
        is_complex=is_complex
    )
    Ia_data = _get_data_from_file(
        data_file=os.path.join(path_to_this_file, Ia_file),
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



# =================================================================
# ============= Preparations for testing finished =================
# =================================================================



class AdmittanceMatrixTests(unittest.TestCase):
    def test_admittance_matrix_values(self):
        pass



class TimeDataTests(unittest.TestCase):
    def test_snr(self):
        initial_time_data_as_dict = get_initial_time_data()
        time_data = data.TimeData(
            Vm_time_data=initial_time_data_as_dict['Vm'],
            Va_time_data=initial_time_data_as_dict['Va'],
            Im_time_data=initial_time_data_as_dict['Im'],
            Ia_time_data=initial_time_data_as_dict['Ia'],
            dt=0.05
        )

        time_data.apply_white_noise(snr=45.0, d_coi=0.0)
        correct_data_after_snr = get_time_data_after_snr()

        self.assertEqual(len(time_data.Vm), len(time_data.Va))
        self.assertEqual(len(time_data.Va), len(time_data.Im))
        self.assertEqual(len(time_data.Im), len(time_data.Ia))
        self.assertEqual(len(time_data.Vm), len(correct_data_after_snr['Vm']))
        self.assertEqual(len(time_data.Va), len(correct_data_after_snr['Va']))
        self.assertEqual(len(time_data.Im), len(correct_data_after_snr['Im']))
        self.assertEqual(len(time_data.Ia), len(correct_data_after_snr['Ia']))
        time_data_points_n = len(time_data.Vm)

        # Applying SNR includes generation random numbers
        # That is why we compare a few digits using self.assertAlmostEqual
        for i in range(time_data_points_n):
            self.assertAlmostEqual(
                time_data.Vm[i], correct_data_after_snr['Vm'][i],
                places=3
            )
            self.assertAlmostEqual(
                time_data.Va[i], correct_data_after_snr['Va'][i],
                places=3
            )
            self.assertAlmostEqual(
                time_data.Im[i], correct_data_after_snr['Im'][i],
                places=1  # a bit strange that only 1 digit matches
            )
            self.assertAlmostEqual(
                time_data.Ia[i], correct_data_after_snr['Ia'][i],
                places=3
            )



class FreqDataTests(unittest.TestCase):
    def test_fft(self):
        time_data_after_snr_as_dict = get_time_data_after_snr()
        time_data_after_snr = data.TimeData(
            Vm_time_data=time_data_after_snr_as_dict['Vm'],
            Va_time_data=time_data_after_snr_as_dict['Va'],
            Im_time_data=time_data_after_snr_as_dict['Im'],
            Ia_time_data=time_data_after_snr_as_dict['Ia'],
            dt=0.05
        )

        snr = 45.0
        d_coi = 0.0
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
        correct_freq_data = get_freq_data_after_fft()

        # TODO: check freq_data.freqs
        freq_data_points_n = len(freq_data.freqs)
        self.assertEqual(freq_data_points_n, 1001)

        self.assertEqual(freq_data_points_n, len(correct_freq_data['Vm']))
        self.assertEqual(freq_data_points_n, len(correct_freq_data['Va']))
        self.assertEqual(freq_data_points_n, len(correct_freq_data['Im']))
        self.assertEqual(freq_data_points_n, len(correct_freq_data['Ia']))

        # places -- number of digits after decimal point to compare
        self.assertAlmostEqual(freq_data.std_w_Vm * 10**7, 3.4196, places=3)
        self.assertAlmostEqual(freq_data.std_w_Va * 10**7, 7.2294, places=2)
        self.assertAlmostEqual(freq_data.std_w_Im * 10**4, 1.4343, places=4)
        self.assertAlmostEqual(freq_data.std_w_Ia * 10**7, 9.9982, places=3)

        # DC has been excluded
        self.assertEqual(freq_data.Vm[0], 0.0)
        self.assertEqual(freq_data.Va[0], 0.0)
        self.assertEqual(freq_data.Im[0], 0.0)
        self.assertEqual(freq_data.Ia[0], 0.0)

        for i in range(1, freq_data_points_n):
            self.assertAlmostEqual(
                freq_data.Vm[i], correct_freq_data['Vm'][i],
                places=13
            )
            self.assertAlmostEqual(
                freq_data.Va[i], correct_freq_data['Va'][i],
                places=13
            )
            self.assertAlmostEqual(
                freq_data.Im[i], correct_freq_data['Im'][i],
                places=13
            )
            self.assertAlmostEqual(
                freq_data.Ia[i], correct_freq_data['Ia'][i],
                places=13
            )




    def test_trim_and_exclude_dc(self):
        pass



    def test_define_stage1_data_vectors(self):
        pass



if __name__ == '__main__':
    unittest.main()


# GP = settings.GeneratorParameters(  # true generator parameters
#     d_2=0.25,
#     e_2=1.0,
#     m_2=1.0,
#     x_d2=0.01,
#     ic_d2=1.0
# )
#
# FD = settings.FreqData(
#     lower_fb=1.99,
#     upper_fb=2.01,
#     max_freq=6.00,
#     dt=0.05
# )


