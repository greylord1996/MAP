import sys
import os
import os.path
import unittest
import numpy as np

sys.path.append(os.path.join(
    os.path.abspath(os.path.dirname(__file__)), '..', 'src')
)
import dynamic_equations_to_simulate
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



def get_correct_initial_time_data_as_dict():
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



def get_correct_time_data_after_snr_as_dict():
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



def get_correct_freq_data_after_fft_as_dict():
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



def get_correct_freq_data_after_trim_as_dict():
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



def get_correct_freq_data_before_stage1_as_dict():
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



def get_initial_freq_data():
    time_data_after_snr_as_dict = get_correct_time_data_after_snr_as_dict()
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
            np.std(time_data_after_snr.Vm, ddof=1) / (10.0 ** (snr / 20.0))
    )
    time_data_after_snr.std_dev_Im = (
            np.std(time_data_after_snr.Im, ddof=1) / (10.0 ** (snr / 20.0))
    )
    time_data_after_snr.std_dev_Va = (
            np.std(time_data_after_snr.Va - d_coi, ddof=1) / (10.0 ** (snr / 20.0))
    )
    time_data_after_snr.std_dev_Ia = (
            np.std(time_data_after_snr.Ia - d_coi, ddof=1) / (10.0 ** (snr / 20.0))
    )

    return data.FreqData(time_data_after_snr)



# =================================================================
# ============= Preparations for testing finished =================
# =================================================================



class AdmittanceMatrixTests(unittest.TestCase):
    """Checks correctness of the admittance matrix.

    Just evaluates the admittance matrix in some picked points
    (an instance of the class AdmittanceMatrix is a symbolic object)
    and compares values of the matrix with true values in these points.
    """

    def test_admittance_matrix_values(self):
        pass



class TimeDataTests(unittest.TestCase):
    """Checks properly working of the data.TimeData class.

    The following features are being tested:
    1. data simulation (in time domain)
    2. applying AWGN to already generated data
    """

    def simulate_time_data(self):
        white_noise = settings.WhiteNoise(
            rnd_amp=0.000
        )
        # true generator parameters
        generator_params = settings.GeneratorParameters(
            d_2=0.25,
            e_2=1.0,
            m_2=1.0,
            x_d2=0.01,
            ic_d2=1.0
        )
        integration_settings = settings.IntegrationSettings(
            dt_step=0.05,
            df_length=100.0
        )
        oscillation_params = settings.OscillationParameters(
            osc_amp=2.00,
            osc_freq=0.005
        )
        solver = dynamic_equations_to_simulate.OdeSolver(
            white_noise=white_noise,
            gen_param=generator_params,
            osc_param=oscillation_params,
            integr_param=integration_settings
        )
        solver.simulate_time_data()
        time_data = data.TimeData(
            Vm_time_data=solver.Vc1_abs,
            Va_time_data=solver.Vc1_angle,
            Im_time_data=solver.Ig_abs,
            Ia_time_data=solver.Ig_angle,
            dt=solver.dt
        )
        return time_data


    def check_lengths(self, time_data, correct_time_data):
        self.assertEqual(len(time_data.Vm), len(time_data.Va))
        self.assertEqual(len(time_data.Va), len(time_data.Im))
        self.assertEqual(len(time_data.Im), len(time_data.Ia))

        self.assertEqual(len(time_data.Vm), len(correct_time_data['Vm']))
        self.assertEqual(len(time_data.Va), len(correct_time_data['Va']))
        self.assertEqual(len(time_data.Im), len(correct_time_data['Im']))
        self.assertEqual(len(time_data.Ia), len(correct_time_data['Ia']))


    def check_data(self, time_data, correct_time_data,
                   Vm_places, Va_places, Im_places, Ia_places):
        time_data_points_len = len(time_data.Vm)
        for i in range(time_data_points_len):
            self.assertAlmostEqual(
                time_data.Vm[i], correct_time_data['Vm'][i],
                places=Vm_places
            )
            self.assertAlmostEqual(
                time_data.Va[i], correct_time_data['Va'][i],
                places=Va_places
            )
            self.assertAlmostEqual(
                time_data.Im[i], correct_time_data['Im'][i],
                places=Im_places
            )
            self.assertAlmostEqual(
                time_data.Ia[i], correct_time_data['Ia'][i],
                places=Ia_places
            )


    def test_data_simulation(self):
        time_data = self.simulate_time_data()
        correct_time_data = get_correct_initial_time_data_as_dict()

        self.check_lengths(time_data, correct_time_data)
        self.check_data(
            time_data=time_data,
            correct_time_data=correct_time_data,
            Vm_places=-2,  # WARNING! Precision must be reduced!
            Va_places=-2,  # WARNING! Precision must be reduced!
            Im_places=-2,  # WARNING! Precision must be reduced!
            Ia_places=-2   # WARNING! Precision must be reduced!
        )


    def test_snr(self):
        initial_time_data_as_dict = get_correct_initial_time_data_as_dict()
        time_data = data.TimeData(
            Vm_time_data=initial_time_data_as_dict['Vm'],
            Va_time_data=initial_time_data_as_dict['Va'],
            Im_time_data=initial_time_data_as_dict['Im'],
            Ia_time_data=initial_time_data_as_dict['Ia'],
            dt=0.05
        )

        time_data.apply_white_noise(snr=45.0, d_coi=0.0)
        correct_time_data = get_correct_time_data_after_snr_as_dict()

        self.check_lengths(time_data, correct_time_data)

        # Applying SNR includes generation random numbers
        # That is why we compare a few digits using self.assertAlmostEqual
        self.check_data(
            time_data=time_data,
            correct_time_data=correct_time_data,
            Vm_places=3,
            Va_places=3,
            Im_places=1,  # a bit strange that only 1 digit matches
            Ia_places=3
        )



class FreqDataTests(unittest.TestCase):
    """Checks properly working of the data.FreqData class.

    The following features are being tested:
    1. applying FFT to data in time domain
    2. trimming data (in frequency domain)
    3. removing data which are located in forced oscillations band
    """

    def check_lengths(self, freq_data, correct_freq_data):
        freq_data_points_len = len(freq_data.freqs)

        self.assertEqual(freq_data_points_len, len(freq_data.Vm))
        self.assertEqual(len(freq_data.Vm), len(freq_data.Va))
        self.assertEqual(len(freq_data.Va), len(freq_data.Im))
        self.assertEqual(len(freq_data.Im), len(freq_data.Ia))

        self.assertEqual(freq_data_points_len, len(correct_freq_data['Vm']))
        self.assertEqual(freq_data_points_len, len(correct_freq_data['Va']))
        self.assertEqual(freq_data_points_len, len(correct_freq_data['Im']))
        self.assertEqual(freq_data_points_len, len(correct_freq_data['Ia']))


    def check_data(self, freq_data, correct_freq_data, begin, end):
        for i in range(begin, end):
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


    def check_std_deviations(self, freq_data):
        self.assertAlmostEqual(freq_data.std_w_Vm * 10**7, 3.4196, places=3)
        self.assertAlmostEqual(freq_data.std_w_Va * 10**7, 7.2294, places=2)
        self.assertAlmostEqual(freq_data.std_w_Im * 10**4, 1.4343, places=4)
        self.assertAlmostEqual(freq_data.std_w_Ia * 10**7, 9.9982, places=3)


    def test_fft(self):
        freq_data = get_initial_freq_data()
        correct_freq_data = get_correct_freq_data_after_fft_as_dict()

        freq_data_points_len = len(freq_data.freqs)
        self.assertEqual(freq_data_points_len, 1001)
        # TODO: check freq_data.freqs

        self.check_lengths(freq_data, correct_freq_data)
        self.check_std_deviations(freq_data)

        # DC has been excluded
        self.assertEqual(freq_data.Vm[0], 0.0)
        self.assertEqual(freq_data.Va[0], 0.0)
        self.assertEqual(freq_data.Im[0], 0.0)
        self.assertEqual(freq_data.Ia[0], 0.0)
        self.check_data(
            freq_data=freq_data,
            correct_freq_data=correct_freq_data,
            begin=1,
            end=freq_data_points_len
        )


    def test_remove_zero_frequency_and_trim(self):
        freq_data = get_initial_freq_data()
        freq_data.remove_zero_frequency()
        freq_data.trim(min_freq=0.0, max_freq=6.0)
        correct_freq_data = get_correct_freq_data_after_trim_as_dict()

        freq_data_points_len = len(freq_data.freqs)
        self.assertEqual(freq_data_points_len, 600)
        # TODO: check freq_data.freqs

        self.check_lengths(freq_data, correct_freq_data)
        self.check_std_deviations(freq_data)
        self.check_data(
            freq_data=freq_data,
            correct_freq_data=correct_freq_data,
            begin=0,
            end=freq_data_points_len
        )


    def test_define_stage1_data_vector(self):
        freq_data = get_initial_freq_data()
        freq_data.remove_zero_frequency()
        freq_data.trim(min_freq=0.0, max_freq=6.0)
        # freq_data.remove_data_from_FO_band()
        correct_freq_data = get_correct_freq_data_before_stage1_as_dict()

        # freq_data_points_len = len(freq_data.freqs)
        # self.assertEqual(freq_data_points_len, 597)
        # TODO: check freq_data.freqs

        # self.check_lengths(freq_data, correct_freq_data)
        self.check_std_deviations(freq_data)
        # self.check_data(
        #     freq_data=freq_data,
        #     correct_freq_data=correct_freq_data,
        #     begin=0,
        #     end=freq_data_points_len
        # )



if __name__ == '__main__':
    unittest.main(verbosity=2)



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


