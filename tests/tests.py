import sys
import os
import os.path
import json
import unittest

import numpy as np

# directory with source code
PATH_TO_THIS_FILE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PATH_TO_THIS_FILE, '..', 'src'))

# import admittance_matrix
import objective_function

import our_data
import correct_data


# Test sets
TEST_DIRS = (
    os.path.join(PATH_TO_THIS_FILE, 'Rnd_Amp_0002'),
    # os.path.join(PATH_TO_THIS_FILE, 'Rnd_Amp_0005')
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
    2. applying AWGN to already simulated data
    """

    def _check_lengths(self, our_time_data, correct_time_data):
        assert len(correct_time_data['Vm']) == len(correct_time_data['Va'])
        assert len(correct_time_data['Va']) == len(correct_time_data['Im'])
        assert len(correct_time_data['Im']) == len(correct_time_data['Ia'])

        self.assertEqual(len(our_time_data.Vm), len(our_time_data.Va))
        self.assertEqual(len(our_time_data.Va), len(our_time_data.Im))
        self.assertEqual(len(our_time_data.Im), len(our_time_data.Ia))

        self.assertEqual(len(our_time_data.Vm), len(correct_time_data['Vm']))
        self.assertEqual(len(our_time_data.Va), len(correct_time_data['Va']))
        self.assertEqual(len(our_time_data.Im), len(correct_time_data['Im']))
        self.assertEqual(len(our_time_data.Ia), len(correct_time_data['Ia']))


    def test_data_simulation(self):
        for test_dir in TEST_DIRS:
            our_time_data = our_data.get_initial_time_data(test_dir)
            correct_time_data = correct_data.get_initial_time_data(test_dir)
            self._check_lengths(our_time_data, correct_time_data)

            relative_precision = 0  # WARNING! Low precision!
            time_data_points_len = len(our_time_data.Vm)
            for i in range(time_data_points_len):
                self.assertAlmostEqual(
                    our_time_data.Vm[i] / correct_time_data['Vm'][i], 1.0,
                    places=relative_precision
                )
                self.assertAlmostEqual(
                    our_time_data.Va[i] / correct_time_data['Va'][i], 1.0,
                    places=relative_precision
                )
                self.assertAlmostEqual(
                    our_time_data.Im[i] / correct_time_data['Im'][i], 1.0,
                    places=relative_precision
                )
                self.assertAlmostEqual(
                    our_time_data.Ia[i] / correct_time_data['Ia'][i], 1.0,
                    places=relative_precision
                )


    def test_white_noise(self):
        for test_dir in TEST_DIRS:
            our_time_data = our_data.get_time_data_after_snr(test_dir)
            correct_time_data = correct_data.get_time_data_after_snr(test_dir)
            self._check_lengths(our_time_data, correct_time_data)

            # there is a three-sigma rule
            # we have two variables sampled from normal distribution to compare
            # how far can they be from each other?
            sigma_factor = 6.0  # less far than 6.0*sigma

            time_data_points_len = len(our_time_data.Vm)
            for i in range(time_data_points_len):
                self.assertLessEqual(
                    abs(our_time_data.Vm[i] - correct_time_data['Vm'][i]),
                    sigma_factor * our_time_data.std_dev_Vm
                )
                self.assertLessEqual(
                    abs(our_time_data.Va[i] - correct_time_data['Va'][i]),
                    sigma_factor * our_time_data.std_dev_Va
                )
                self.assertLessEqual(
                    abs(our_time_data.Im[i] - correct_time_data['Im'][i]),
                    sigma_factor * our_time_data.std_dev_Im
                )
                self.assertLessEqual(
                    abs(our_time_data.Ia[i] - correct_time_data['Ia'][i]),
                    sigma_factor * our_time_data.std_dev_Ia
                )



class FreqDataTests(unittest.TestCase):
    """Checks proper working of the data.FreqData class.

    The following features are being tested:
    1. applying FFT to data in time domain
    2. trimming data (in frequency domain)
    3. removing data which are located in forced oscillations band
    """

    def _check_lengths(self, our_freq_data, correct_freq_data):
        assert len(correct_freq_data['freqs']) == len(correct_freq_data['Vm'])
        assert len(correct_freq_data['Vm']) == len(correct_freq_data['Va'])
        assert len(correct_freq_data['Va']) == len(correct_freq_data['Im'])
        assert len(correct_freq_data['Im']) == len(correct_freq_data['Ia'])

        freq_data_points_len = len(our_freq_data.freqs)
        self.assertEqual(freq_data_points_len, len(our_freq_data.Vm))
        self.assertEqual(len(our_freq_data.Vm), len(our_freq_data.Va))
        self.assertEqual(len(our_freq_data.Va), len(our_freq_data.Im))
        self.assertEqual(len(our_freq_data.Im), len(our_freq_data.Ia))

        self.assertEqual(freq_data_points_len, len(correct_freq_data['freqs']))
        self.assertEqual(freq_data_points_len, len(correct_freq_data['Vm']))
        self.assertEqual(freq_data_points_len, len(correct_freq_data['Va']))
        self.assertEqual(freq_data_points_len, len(correct_freq_data['Im']))
        self.assertEqual(freq_data_points_len, len(correct_freq_data['Ia']))


    def _check_data(self, our_freq_data, correct_freq_data, begin, end):
        self._check_lengths(our_freq_data, correct_freq_data)
        for i in range(begin, end):
            self.assertAlmostEqual(
                our_freq_data.freqs[i], correct_freq_data['freqs'][i],
                places=13
            )

        relative_precision = 7
        for i in range(begin, end):
            self.assertAlmostEqual(
                our_freq_data.Vm[i].real / correct_freq_data['Vm'][i].real,
                1.0,
                places=relative_precision
            )
            self.assertAlmostEqual(
                our_freq_data.Vm[i].imag / correct_freq_data['Vm'][i].imag,
                1.0,
                places=relative_precision
            )

            self.assertAlmostEqual(
                our_freq_data.Va[i].real / correct_freq_data['Va'][i].real,
                1.0,
                places=relative_precision
            )
            self.assertAlmostEqual(
                our_freq_data.Va[i].imag / correct_freq_data['Va'][i].imag,
                1.0,
                places=relative_precision
            )

            self.assertAlmostEqual(
                our_freq_data.Im[i].real / correct_freq_data['Im'][i].real,
                1.0,
                places=relative_precision
            )
            self.assertAlmostEqual(
                our_freq_data.Im[i].imag / correct_freq_data['Im'][i].imag,
                1.0,
                places=relative_precision
            )

            self.assertAlmostEqual(
                our_freq_data.Ia[i].real / correct_freq_data['Ia'][i].real,
                1.0,
                places=relative_precision
            )
            self.assertAlmostEqual(
                our_freq_data.Ia[i].imag / correct_freq_data['Ia'][i].imag,
                1.0,
                places=relative_precision
            )


    def _check_std_deviations(self, our_freq_data, test_dir):
        all_correct_values = correct_data.get_correct_values(test_dir)
        correct_std_deviations = all_correct_values['freq_data_std_dev_eps']
        relative_precision = 13

        self.assertAlmostEqual(
            our_freq_data.std_w_Vm / correct_std_deviations['std_dev_eps_Vm'],
            1.0,
            places=relative_precision
        )
        self.assertAlmostEqual(
            our_freq_data.std_w_Va / correct_std_deviations['std_dev_eps_Va'],
            1.0,
            places=relative_precision
        )
        self.assertAlmostEqual(
            our_freq_data.std_w_Im / correct_std_deviations['std_dev_eps_Im'],
            1.0,
            places=relative_precision
        )
        self.assertAlmostEqual(
            our_freq_data.std_w_Ia / correct_std_deviations['std_dev_eps_Ia'],
            1.0,
            places=relative_precision
        )


    def test_fft(self):
        for test_dir in TEST_DIRS:
            our_freq_data = our_data.get_freq_data_after_fft(test_dir)
            correct_freq_data = correct_data.get_freq_data_after_fft(test_dir)

            # This assert have to be removed after extension of test sets
            assert len(our_freq_data.freqs) == 1001

            # DC has been excluded
            self.assertEqual(our_freq_data.freqs[0], 0.0)
            self.assertEqual(our_freq_data.Vm[0], 0.0)
            self.assertEqual(our_freq_data.Va[0], 0.0)
            self.assertEqual(our_freq_data.Im[0], 0.0)
            self.assertEqual(our_freq_data.Ia[0], 0.0)

            self._check_std_deviations(our_freq_data, test_dir)
            self._check_data(
                our_freq_data=our_freq_data,
                correct_freq_data=correct_freq_data,
                begin=1,  # we will exclude DC in remove_zero_frequency
                end=len(our_freq_data.freqs)
            )


    def test_remove_zero_frequency_and_trim(self):
        for test_dir in TEST_DIRS:
            our_freq_data = (
                our_data.get_freq_data_after_remove_dc_and_trim(test_dir)
            )
            correct_freq_data = (
                correct_data.get_freq_data_after_remove_dc_and_trim(test_dir)
            )

            # This assert have to be removed after extension of test sets
            assert len(our_freq_data.freqs) == 600

            self._check_std_deviations(our_freq_data, test_dir)
            self._check_data(
                our_freq_data=our_freq_data,
                correct_freq_data=correct_freq_data,
                begin=0,
                end=len(our_freq_data.freqs)
            )


    def test_remove_data_from_fo_band(self):
        for test_dir in TEST_DIRS:
            our_freq_data = (
                our_data.get_freq_data_after_remove_fo_band(test_dir)
            )
            correct_freq_data = (
                correct_data.get_freq_data_after_remove_fo_band(test_dir)
            )

            # This assert have to be removed after extension of test sets
            assert len(our_freq_data.freqs) == 597

            self._check_std_deviations(our_freq_data, test_dir)
            self._check_data(
                our_freq_data=our_freq_data,
                correct_freq_data=correct_freq_data,
                begin=0,
                end=len(our_freq_data.freqs)
            )



class ObjectiveFunctionTests(unittest.TestCase):
    """Checks correctness of the objective function (which will be minimized).

    Computes the objective function at some picked points and compares
    obtained values with true values (see test_dir/correct_values.json).
    """

    def _check_covariance_matrix(self, our_gamma_L, test_dir):
        # gamma_L is calculated at prior_mean point (see prior.json)
        correct_values = correct_data.get_correct_values(test_dir)
        correct_gamma_L = correct_values['CovarianceMatrix']

        self.assertEqual(our_gamma_L.shape[0], correct_gamma_L['size_y'])
        self.assertEqual(our_gamma_L.shape[1], correct_gamma_L['size_x'])

        for coords, matrix_element in correct_gamma_L['values'].items():
            y_coord = int(coords.split(',')[0]) - 1
            x_coord = int(coords.split(',')[1]) - 1
            self.assertAlmostEqual(
                our_gamma_L[y_coord][x_coord] / matrix_element, 1.0,
                places=10
            )


    def _check_objective_function_values(self, func, test_dir):
        correct_values = correct_data.get_correct_values(test_dir)
        correct_func_values = correct_values['ObjectiveFunction']['values']

        for args, correct_func_value in correct_func_values.items():
            args_array = np.array([np.float(arg) for arg in args.split(',')])
            self.assertAlmostEqual(
                func.compute_from_array(args_array) / correct_func_value, 1.0,
                places=13
            )


    def _check_objective_function_gradients(self, func, test_dir):
        correct_values = correct_data.get_correct_values(test_dir)
        correct_gradients = correct_values['ObjectiveFunction']['gradients']

        for args, correct_gradient in correct_gradients.items():
            args_array = np.array([np.float(arg) for arg in args.split(',')])
            correct_gradient_array = np.array([
                np.float(correct_gradient_component)
                for correct_gradient_component in correct_gradient.split(',')
            ])

            computed_func_gradient = (
                func.compute_gradient_from_array(args_array)
            )
            for i in range(len(args_array)):
                self.assertAlmostEqual(
                    computed_func_gradient[i] / correct_gradient_array[i], 1.0,
                    places=9
                )


    def test_objective_function(self):
        for test_dir in TEST_DIRS:
            correct_prepared_freq_data = (
                correct_data.get_prepared_freq_data(test_dir)
            )

            # perturbed generator's parameters
            json_prior = None
            with open(os.path.join(test_dir, 'prior.json'), 'r') as prior_file:
                json_prior = json.load(prior_file)
            gen_params_prior_mean = (
                objective_function.OptimizingGeneratorParameters(
                    D_Ya=json_prior['prior_mean']['D_Ya'],
                    Ef_a=json_prior['prior_mean']['Ef_a'],
                    M_Ya=json_prior['prior_mean']['M_Ya'],
                    X_Ya=json_prior['prior_mean']['X_Ya']
                )
            )
            gen_params_prior_std_dev = (
                objective_function.OptimizingGeneratorParameters(
                    D_Ya=json_prior['prior_std_dev']['D_Ya'],
                    Ef_a=json_prior['prior_std_dev']['Ef_a'],
                    M_Ya=json_prior['prior_std_dev']['M_Ya'],
                    X_Ya=json_prior['prior_std_dev']['X_Ya']
                )
            )

            # f -- objective function to minimize
            f = objective_function.ObjectiveFunction(
                freq_data=correct_prepared_freq_data,
                gen_params_prior_mean=gen_params_prior_mean,
                gen_params_prior_std_dev=gen_params_prior_std_dev
            )

            self._check_covariance_matrix(
                our_gamma_L=f._gamma_L.compute(gen_params_prior_mean),
                test_dir=test_dir
            )
            self._check_objective_function_values(
                func=f,
                test_dir=test_dir
            )
            self._check_objective_function_gradients(
                func=f,
                test_dir=test_dir
            )

            # just check that the following objects can be successfully computed
            starting_point_R = f._R.compute(gen_params_prior_mean)
            starting_point_gamma_L = f._gamma_L.compute(gen_params_prior_mean)

            starting_point_R_partial_derivatives = (
                f._R.compute_partial_derivatives(gen_params_prior_mean)
            )
            starting_point_gamma_L_partial_derivatives = (
                f._gamma_L.compute_partial_derivatives(gen_params_prior_mean)
            )
            # starting_point_inverted_gamma_L_partial_derivatives = (
            #     f._gamma_L.compute_inverted_matrix_partial_derivatives(
            #         gen_params_prior_mean
            #     )
            # )
            starting_point_f_gradient = f.compute_gradient(gen_params_prior_mean)
            starting_point_f_gradient = (
                f.compute_gradient_from_array(gen_params_prior_mean.as_array)
            )



if __name__ == '__main__':
    unittest.main(verbosity=2)

