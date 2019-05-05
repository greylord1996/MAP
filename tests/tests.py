import sys
import os
import os.path
import unittest

import numpy as np


# directory with source code
PATH_TO_THIS_FILE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PATH_TO_THIS_FILE, '..', 'src'))

import dynamic_equations_to_simulate
import admittance_matrix
import objective_function
import data

import our_data
import correct_data


# Test sets
TEST_DIRS = (
    os.path.join(PATH_TO_THIS_FILE, 'Rnd_Amp_0000'),
    # os.path.join(PATH_TO_THIS_FILE, 'Rnd_Amp_0002')
)



# class AdmittanceMatrixTests(unittest.TestCase):
#     """Checks correctness of the admittance matrix.
#
#     Just evaluates the admittance matrix in some picked points
#     (an instance of the class AdmittanceMatrix is a symbolic object)
#     and compares values of the matrix with true values in these points.
#     """
#
#     def test_admittance_matrix_values(self):
#         matrix_Y = admittance_matrix.AdmittanceMatrix(is_actual=False).Ys
#         pass



class TimeDataTests(unittest.TestCase):
    """Checks proper working of the data.TimeData class.

    The following features are being tested:
    1. data simulation (in time domain)
    2. applying AWGN to already generated data
    """

    def _simulate_time_data(self, test_dir):
        initial_params = our_data.get_initial_params(test_dir)
        solver = dynamic_equations_to_simulate.OdeSolver(
            white_noise=initial_params.white_noise,
            gen_param=initial_params.generator_parameters,
            osc_param=initial_params.oscillation_parameters,
            integr_param=initial_params.integration_settings
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


    def _check_lengths(self, time_data, correct_time_data):
        self.assertEqual(len(time_data.Vm), len(time_data.Va))
        self.assertEqual(len(time_data.Va), len(time_data.Im))
        self.assertEqual(len(time_data.Im), len(time_data.Ia))

        self.assertEqual(len(time_data.Vm), len(correct_time_data['Vm']))
        self.assertEqual(len(time_data.Va), len(correct_time_data['Va']))
        self.assertEqual(len(time_data.Im), len(correct_time_data['Im']))
        self.assertEqual(len(time_data.Ia), len(correct_time_data['Ia']))


    def _check_data(self, time_data, correct_time_data):
        self._check_lengths(time_data, correct_time_data)
        time_data_points_len = len(time_data.Vm)
        relative_precision = 3  # WARNING! Low precision!

        print('Ia[0] =', time_data.Ia[0])

        for i in range(time_data_points_len):
            # self.assertAlmostEqual(
            #     time_data.Vm[i] / correct_time_data['Vm'][i], 1.0,
            #     places=relative_precision
            # )
            self.assertAlmostEqual(
                time_data.Va[i] / correct_time_data['Va'][i], 1.0,
                places=relative_precision
            )
            self.assertAlmostEqual(
                time_data.Im[i] / correct_time_data['Im'][i], 1.0,
                places=relative_precision
            )
            self.assertAlmostEqual(
                time_data.Ia[i] / correct_time_data['Ia'][i], 1.0,
                places=relative_precision
            )


    def test_data_simulation(self):
        for test_dir in TEST_DIRS:
            time_data = self._simulate_time_data(test_dir)
            correct_time_data = correct_data.get_initial_time_data(test_dir)

            self._check_data(
                time_data=time_data,
                correct_time_data=correct_time_data
            )


    def test_snr(self):
        for test_dir in TEST_DIRS:
            initial_params = our_data.get_initial_params(test_dir)
            initial_time_data_as_dict = (
                correct_data.get_initial_time_data(test_dir)
            )
            time_data = data.TimeData(
                Vm_time_data=initial_time_data_as_dict['Vm'],
                Va_time_data=initial_time_data_as_dict['Va'],
                Im_time_data=initial_time_data_as_dict['Im'],
                Ia_time_data=initial_time_data_as_dict['Ia'],
                dt=initial_params.integration_settings.dt_step
            )

            snr = 45.0
            d_coi = 0.0
            time_data.apply_white_noise(snr, d_coi)
            correct_time_data = correct_data.get_time_data_after_snr(test_dir)

            # Applying SNR includes generation random numbers
            self._check_data(
                time_data=time_data,
                correct_time_data=correct_time_data
            )



class FreqDataTests(unittest.TestCase):
    """Checks proper working of the data.FreqData class.

    The following features are being tested:
    1. applying FFT to data in time domain
    2. trimming data (in frequency domain)
    3. removing data which are located in forced oscillations band
    """

    def _check_lengths(self, freq_data, correct_freq_data):
        freq_data_points_len = len(freq_data.freqs)

        self.assertEqual(freq_data_points_len, len(freq_data.Vm))
        self.assertEqual(len(freq_data.Vm), len(freq_data.Va))
        self.assertEqual(len(freq_data.Va), len(freq_data.Im))
        self.assertEqual(len(freq_data.Im), len(freq_data.Ia))

        self.assertEqual(freq_data_points_len, len(correct_freq_data['Vm']))
        self.assertEqual(freq_data_points_len, len(correct_freq_data['Va']))
        self.assertEqual(freq_data_points_len, len(correct_freq_data['Im']))
        self.assertEqual(freq_data_points_len, len(correct_freq_data['Ia']))


    def _check_data(self, freq_data, correct_freq_data, begin, end):
        self._check_lengths(freq_data, correct_freq_data)
        for i in range(begin, end):
            self.assertAlmostEqual(
                freq_data.freqs[i], correct_freq_data['freqs'][i],
                places=13
            )

        relative_precision = 10
        for i in range(begin, end):
            # self.assertAlmostEqual(
            #     freq_data.Vm[i] / correct_freq_data['Vm'][i], np.complex(1.0),
            #     places=relative_precision
            # )
            self.assertAlmostEqual(
                freq_data.Va[i] / correct_freq_data['Va'][i], np.complex(1.0),
                places=relative_precision
            )
            self.assertAlmostEqual(
                freq_data.Im[i] / correct_freq_data['Im'][i], np.complex(1.0),
                places=relative_precision
            )
            self.assertAlmostEqual(
                freq_data.Ia[i] / correct_freq_data['Ia'][i], np.complex(1.0),
                places=relative_precision
            )


    def _check_std_deviations(self, test_dir):
        freq_data = our_data.get_initial_freq_data(test_dir)
        all_correct_values = correct_data.get_correct_values(test_dir)
        correct_std_deviations = all_correct_values['freq_data_std_dev_eps']
        relative_precision = 3

        # self.assertAlmostEqual(
        #     freq_data.std_w_Vm / correct_std_deviations['std_dev_eps_Vm'], 1.0,
        #     places=relative_precision
        # )
        self.assertAlmostEqual(
            freq_data.std_w_Va / correct_std_deviations['std_dev_eps_Va'], 1.0,
            places=relative_precision
        )
        self.assertAlmostEqual(
            freq_data.std_w_Im / correct_std_deviations['std_dev_eps_Im'], 1.0,
            places=relative_precision
        )
        self.assertAlmostEqual(
            freq_data.std_w_Ia / correct_std_deviations['std_dev_eps_Ia'], 1.0,
            places=relative_precision
        )


    def test_fft(self):
        for test_dir in TEST_DIRS:
            freq_data = our_data.get_initial_freq_data(test_dir)
            correct_freq_data = correct_data.get_freq_data_after_fft(test_dir)

            # This assert have to be removed after extension of test sets
            assert len(freq_data.freqs) == 1001

            # DC has been excluded
            self.assertEqual(freq_data.freqs[0], 0.0)
            self.assertEqual(freq_data.Vm[0], 0.0)
            self.assertEqual(freq_data.Va[0], 0.0)
            self.assertEqual(freq_data.Im[0], 0.0)
            self.assertEqual(freq_data.Ia[0], 0.0)

            self._check_std_deviations(test_dir)
            self._check_data(
                freq_data=freq_data,
                correct_freq_data=correct_freq_data,
                begin=1,  # we will exclude DC in remove_zero_frequency
                end=len(freq_data.freqs)
            )


    def test_remove_zero_frequency_and_trim(self):
        for test_dir in TEST_DIRS:
            initial_params = our_data.get_initial_params(test_dir)
            freq_data = our_data.get_initial_freq_data(test_dir)
            freq_data.remove_zero_frequency()
            freq_data.trim(
                min_freq=0.0,
                max_freq=initial_params.freq_data.max_freq
            )
            correct_freq_data = (
                correct_data.get_freq_data_after_remove_dc_and_trim(test_dir)
            )

            # This assert have to be removed after extension of test sets
            assert len(freq_data.freqs) == 600

            self._check_std_deviations(test_dir)
            self._check_data(
                freq_data=freq_data,
                correct_freq_data=correct_freq_data,
                begin=0,
                end=len(freq_data.freqs)
            )


    def test_remove_data_from_fo_band(self):
        for test_dir in TEST_DIRS:
            freq_data = our_data.get_prepared_freq_data(test_dir)
            correct_freq_data = (
                correct_data.get_freq_data_after_remove_fo_band(test_dir)
            )

            # for i in range(597):
            #     # if round(freq_data.freqs[i] - correct_freq_data['freqs'][i], 5):
            #     print('@@@@@@@@@@@@@', i, freq_data.freqs[i], correct_freq_data['freqs'][i])

            # This assert have to be removed after extension of test sets
            assert len(freq_data.freqs) == 597

            self._check_std_deviations(test_dir)
            self._check_data(
                freq_data=freq_data,
                correct_freq_data=correct_freq_data,
                begin=0,
                end=len(freq_data.freqs)
            )



class ObjectiveFunctionTests(unittest.TestCase):
    """Checks correctness of the objective function (which will be minimized).

    Computes the objective function at some picked points and compares
    obtained values with true values (see test_dir/correct_values.json).
    """

    def _ckeck_covariance_matrix(self, our_gamma_L, test_dir):
        correct_values = correct_data.get_correct_values(test_dir)
        correct_gamma_L = correct_values['CovarianceMatrix']

        self.assertEqual(our_gamma_L.shape[0], correct_gamma_L['size_y'])
        self.assertEqual(our_gamma_L.shape[1], correct_gamma_L['size_x'])

        for coords, matrix_element in correct_gamma_L['values'].items():
            y_coord = int(coords.split(',')[0]) - 1
            x_coord = int(coords.split(',')[1]) - 1
            self.assertAlmostEqual(
                our_gamma_L[y_coord][x_coord] / matrix_element, 1.0,
                places=3
            )


    def _check_objective_function(self, func, test_dir):
        correct_values = correct_data.get_correct_values(test_dir)
        correct_func = correct_values['ObjectiveFunction']
        # print('f =', f.compute_by_array(np.array([100.0, 100.0, 100.0, 100.0])))

        for args, correct_func_value in correct_func['values'].items():
            args_array = np.array([np.float(arg) for arg in args.split(',')])
            print('+++++++++++++++++++++++++++++++++++')
            print(
                '$$$ TESTING:', 'TEST_OBJECTIVE_FUNCTION, OUR/CORRECT RATIO: ',
                func.compute_by_array(args_array) / correct_func_value
            )
            computed_gamma_L = func._gamma_L.compute(
                objective_function.OptimizingGeneratorParameters(
                    D_Ya=args_array[0],
                    Ef_a=args_array[1],
                    M_Ya=args_array[2],
                    X_Ya=args_array[3]
                )
            )
            print('*** COND NUMBER OF GAMMA_L =', np.linalg.cond(computed_gamma_L))
            print('+++++++++++++++++++++++++++++++++++')
            # self.assertAlmostEqual(
            #     func.compute_by_array(args_array) / correct_func_value, 1.0,
            #     places=3
            # )


    def test_objective_function(self):
        for test_dir in TEST_DIRS:
            # initial_params = our_data.get_initial_params(test_dir)
            freq_data = our_data.get_prepared_freq_data(test_dir)

            # perturbed generator's parameters
            gen_params_prior_means = (
                objective_function.OptimizingGeneratorParameters(
                    D_Ya=0.206552362540141,   # prior value of D_Ya
                    Ef_a=0.837172184078094,   # prior value of Ef_a
                    M_Ya=0.665441037484483,   # prior value of M_Ya
                    X_Ya=0.00771416811078329  # prior value of X_Ya
                )
            )
            gen_params_prior_std_devs = (
                objective_function.OptimizingGeneratorParameters(
                    D_Ya=1000.0,  # std dev of D_Ya
                    Ef_a=1000.0,  # std dev of Ef_a
                    M_Ya=1000.0,  # std dev of M_Ya
                    X_Ya=1000.0   # std dev of X_Ya
                )
            )

            # f -- objective function to minimize
            f = objective_function.ObjectiveFunction(
                freq_data=freq_data,
                gen_params_prior_means=gen_params_prior_means,
                gen_params_prior_std_devs=gen_params_prior_std_devs
            )

            self._ckeck_covariance_matrix(
                our_gamma_L=f._gamma_L.compute(gen_params_prior_means),
                test_dir=test_dir
            )
            self._check_objective_function(
                func=f,
                test_dir=test_dir
            )

            initial_point_gradients = (
                f._gamma_L.compute_gradients(gen_params_prior_means)
            )



if __name__ == '__main__':
    unittest.main(verbosity=2)

