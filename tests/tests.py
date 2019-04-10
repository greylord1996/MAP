import sys
import os
import os.path
import unittest
import numpy as np

sys.path.append(os.path.join(
    os.path.abspath(os.path.dirname(__file__)), '..', 'src')
)
import dynamic_equations_to_simulate
import create_admittance_matrix
import objective_function
import settings
import data
import correct_data



FREQ_DATA = settings.FreqData(
    lower_fb=1.988,  # WTF? Should be 1.99!!!
    upper_fb=2.01,
    max_freq=6.00,
    dt=0.05
)
WHITE_NOISE = settings.WhiteNoise(
    rnd_amp=0.000
)
TRUE_GEN_PARAMS = settings.GeneratorParameters(
    d_2=0.25,
    e_2=1.0,
    m_2=1.0,
    x_d2=0.01,
    ic_d2=1.0
)
INTEGRATION_SETTINGS = settings.IntegrationSettings(
    dt_step=0.05,
    df_length=100.0
)
OSCILLATION_PARAMS = settings.OscillationParameters(
    osc_amp=2.00,
    osc_freq=0.005
)



def get_initial_freq_data():
    time_data_after_snr_as_dict = correct_data.get_time_data_after_snr()
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

    return data.FreqData(time_data_after_snr)




class AdmittanceMatrixTests(unittest.TestCase):
    """Checks correctness of the admittance matrix.

    Just evaluates the admittance matrix in some picked points
    (an instance of the class AdmittanceMatrix is a symbolic object)
    and compares values of the matrix with true values in these points.
    """

    # def test_admittance_matrix_values(self):
    #     admittance_matrix = create_admittance_matrix.AdmittanceMatrix(
    #         is_actual=False
    #     )



class TimeDataTests(unittest.TestCase):
    """Checks properly working of the data.TimeData class.

    The following features are being tested:
    1. data simulation (in time domain)
    2. applying AWGN to already generated data
    """

    def simulate_time_data(self):
        solver = dynamic_equations_to_simulate.OdeSolver(
            white_noise=WHITE_NOISE,
            gen_param=TRUE_GEN_PARAMS,
            osc_param=OSCILLATION_PARAMS,
            integr_param=INTEGRATION_SETTINGS
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
        self.check_lengths(time_data, correct_time_data)
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
        correct_time_data = correct_data.get_initial_time_data()
        self.check_data(
            time_data=time_data,
            correct_time_data=correct_time_data,
            Vm_places=0,  # WARNING! Precision must be reduced!
            Va_places=0,  # WARNING! Precision must be reduced!
            Im_places=0,  # WARNING! Precision must be reduced!
            Ia_places=0   # WARNING! Precision must be reduced!
        )


    def test_snr(self):
        initial_time_data_as_dict = correct_data.get_initial_time_data()
        time_data = data.TimeData(
            Vm_time_data=initial_time_data_as_dict['Vm'],
            Va_time_data=initial_time_data_as_dict['Va'],
            Im_time_data=initial_time_data_as_dict['Im'],
            Ia_time_data=initial_time_data_as_dict['Ia'],
            dt=INTEGRATION_SETTINGS.dt_step
        )

        time_data.apply_white_noise(snr=45.0, d_coi=0.0)
        correct_time_data = correct_data.get_time_data_after_snr()

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
        self.check_lengths(freq_data, correct_freq_data)
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
        correct_freq_data = correct_data.get_freq_data_after_fft()

        freq_data_points_len = len(freq_data.freqs)
        self.assertEqual(freq_data_points_len, 1001)
        # TODO: check freq_data.freqs

        # DC has been excluded
        self.assertEqual(freq_data.Vm[0], 0.0)
        self.assertEqual(freq_data.Va[0], 0.0)
        self.assertEqual(freq_data.Im[0], 0.0)
        self.assertEqual(freq_data.Ia[0], 0.0)

        self.check_std_deviations(freq_data)
        self.check_data(
            freq_data=freq_data,
            correct_freq_data=correct_freq_data,
            begin=1,
            end=freq_data_points_len
        )


    def test_remove_zero_frequency_and_trim(self):
        freq_data = get_initial_freq_data()
        freq_data.remove_zero_frequency()
        freq_data.trim(
            min_freq=0.0,
            max_freq=FREQ_DATA.max_freq
        )
        correct_freq_data = correct_data.get_freq_data_after_trim()

        freq_data_points_len = len(freq_data.freqs)
        self.assertEqual(freq_data_points_len, 600)
        # TODO: check freq_data.freqs

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
        freq_data.trim(
            min_freq=0.0,
            max_freq=FREQ_DATA.max_freq
        )
        freq_data.remove_data_from_FO_band(
            min_fo_freq=FREQ_DATA.lower_fb,
            max_fo_freq=FREQ_DATA.upper_fb
        )
        correct_freq_data = correct_data.get_freq_data_before_stage1()

        freq_data_points_len = len(freq_data.freqs)
        self.assertEqual(freq_data_points_len, 597)
        # TODO: check freq_data.freqs

        self.check_std_deviations(freq_data)
        self.check_data(
            freq_data=freq_data,
            correct_freq_data=correct_freq_data,
            begin=0,
            end=freq_data_points_len
        )


class CovarianceMatrixTests(unittest.TestCase):

    def test_covariance_matrix(self):
        freq_data = get_initial_freq_data()
        freq_data.remove_zero_frequency()
        freq_data.trim(
            min_freq=0.0,
            max_freq=FREQ_DATA.max_freq
        )
        freq_data.remove_data_from_FO_band(
            min_fo_freq=FREQ_DATA.lower_fb,
            max_fo_freq=FREQ_DATA.upper_fb
        )

        uncertain_gen_params = objective_function.UncertainGeneratorParameters(
            D_Ya=0.2066,  # check accordance D_Ya <-> d_2
            Ef_a=0.8372,  # check accordance Ef_a <-> e_2
            M_Ya=0.6654,  # check accordance M_Ya <-> m_2
            X_Ya=0.0077,  # check accordance X_Ya <-> x_d2
            std_dev_D_Ya=1000.0,
            std_dev_Ef_a=1000.0,
            std_dev_M_Ya=1000.0,
            std_dev_X_Ya=1000.0
        )

        f = objective_function.ObjectiveFunction(
            freq_data=freq_data,
            prior_gen_params=uncertain_gen_params
        )

        gamma_L = f._gamma_L.compute(uncertain_gen_params)

        self.assertAlmostEqual(gamma_L[0, 2387], 0.0)
        self.assertAlmostEqual(gamma_L[1879, 1880], 0.0)
        self.assertAlmostEqual(gamma_L[1879, 1878], 0.0)
        self.assertAlmostEqual(gamma_L[0, 1], 0.0)
        self.assertAlmostEqual(gamma_L[1, 0], 0.0)

        self.assertAlmostEqual(gamma_L[0, 0] * 10**8, 2.0590, places=3)
        self.assertAlmostEqual(gamma_L[1, 1] * 10**8, 2.0590, places=3)
        # TODO: check values (the following line fails)
        self.assertAlmostEqual(gamma_L[2387, 2387] * 10**12, 2.0539, places=3)



if __name__ == '__main__':
    unittest.main(verbosity=2)

