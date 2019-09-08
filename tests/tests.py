import sys
import os
import os.path
import time
import json
import unittest

import numpy as np

import our_data
import correct_data

# directory with source code
PATH_TO_THIS_FILE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PATH_TO_THIS_FILE, '..', 'src'))
from generator import admittance_matrix
import objective_function


# Test sets
TEST_DIRS = (
    os.path.join(PATH_TO_THIS_FILE, 'Rnd_Amp_0002'),
    # os.path.join(PATH_TO_THIS_FILE, 'Rnd_Amp_0005')
)


class TimeDataTests(unittest.TestCase):
    """Checks proper working of the data.TimeData class.

    The following features are tested:
    1. data simulation (in time domain);
    2. applying additive white Gaussian noise to already simulated data.
    """

    def _check_lengths(self, our_time_data, correct_time_data):
        assert len(correct_time_data['Vm']) == len(correct_time_data['Va'])
        assert len(correct_time_data['Va']) == len(correct_time_data['Im'])
        assert len(correct_time_data['Im']) == len(correct_time_data['Ia'])

        self.assertEqual(len(our_time_data.inputs[0]), len(our_time_data.inputs[1]))
        self.assertEqual(len(our_time_data.inputs[1]), len(our_time_data.outputs[0]))
        self.assertEqual(len(our_time_data.outputs[0]), len(our_time_data.outputs[1]))

        self.assertEqual(len(our_time_data.inputs[0]), len(correct_time_data['Vm']))
        self.assertEqual(len(our_time_data.inputs[1]), len(correct_time_data['Va']))
        self.assertEqual(len(our_time_data.outputs[0]), len(correct_time_data['Im']))
        self.assertEqual(len(our_time_data.outputs[1]), len(correct_time_data['Ia']))

    def test_data_simulation(self):
        for test_dir in TEST_DIRS:
            our_time_data = our_data.get_initial_time_data(test_dir)
            correct_time_data = correct_data.get_initial_time_data(test_dir)
            self._check_lengths(our_time_data, correct_time_data)

            relative_precision = 0  # WARNING! Low precision!
            n_time_points = len(our_time_data.inputs[0])
            for i in range(n_time_points):
                self.assertAlmostEqual(
                    our_time_data.inputs[0][i] / correct_time_data['Vm'][i], 1.0,
                    places=relative_precision
                )
                self.assertAlmostEqual(
                    our_time_data.inputs[1][i] / correct_time_data['Va'][i], 1.0,
                    places=relative_precision
                )
                self.assertAlmostEqual(
                    our_time_data.outputs[0][i] / correct_time_data['Im'][i], 1.0,
                    places=relative_precision
                )
                self.assertAlmostEqual(
                    our_time_data.outputs[1][i] / correct_time_data['Ia'][i], 1.0,
                    places=relative_precision
                )

            # Vm_anomalies_number = 0
            # for i in range(time_data_points_len):
            #     if (our_time_data.Vm[i] / correct_time_data['Vm'][i] < 0.98
            #             or our_time_data.Vm[i] / correct_time_data['Vm'][i] > 1.02):
            #         Vm_anomalies_number += 1
            #         print(
            #             'i =', i,
            #             'our.Vm[i] =', our_time_data.Vm[i],
            #             'correct.Vm[i] =', correct_time_data['Vm'][i]
            #         )
            #
            # Va_anomalies_number = 0
            # for i in range(time_data_points_len):
            #     if (our_time_data.Va[i] / correct_time_data['Va'][i] < 0.98
            #             or our_time_data.Va[i] / correct_time_data['Va'][i] > 1.02):
            #         Va_anomalies_number += 1
            #         print(
            #             'i =', i,
            #             'our.Va[i] =', our_time_data.Va[i],
            #             'correct.Va[i] =', correct_time_data['Va'][i]
            #         )
            #
            # Im_anomalies_number = 0
            # for i in range(time_data_points_len):
            #     if (our_time_data.Im[i] / correct_time_data['Im'][i] < 0.98
            #             or our_time_data.Im[i] / correct_time_data['Im'][i] > 1.02):
            #         Im_anomalies_number += 1
            #         print(
            #             'i =', i,
            #             'our.Im[i] =', our_time_data.Im[i],
            #             'correct.Im[i] =', correct_time_data['Im'][i]
            #         )
            #
            # Ia_anomalies_number = 0
            # for i in range(time_data_points_len):
            #     if (our_time_data.Ia[i] / correct_time_data['Ia'][i] < 0.98
            #             or our_time_data.Ia[i] / correct_time_data['Ia'][i] > 1.02):
            #         Ia_anomalies_number += 1
            #         print(
            #             'i =', i,
            #             'our.Ia[i] =', our_time_data.Ia[i],
            #             'correct.Ia[i] =', correct_time_data['Ia'][i]
            #         )
            #
            # self.assertLessEqual(Vm_anomalies_number, 200)
            # self.assertLessEqual(Va_anomalies_number, 200)
            # self.assertLessEqual(Im_anomalies_number, 1000)  # WARNING! Check Im!
            # self.assertLessEqual(Ia_anomalies_number, 200)


    def test_white_noise(self):
        for test_dir in TEST_DIRS:
            our_time_data = our_data.get_time_data_after_snr(test_dir)
            correct_time_data = correct_data.get_time_data_after_snr(test_dir)
            self._check_lengths(our_time_data, correct_time_data)

            # there is a three-sigma rule
            # we have two variables sampled from normal distribution to compare
            # how far can they be from each other?
            sigma_factor = 6.0  # less far than 6.0*sigma

            n_time_points = len(our_time_data.inputs[0])
            for i in range(n_time_points):
                self.assertLessEqual(
                    abs(our_time_data.inputs[0][i] - correct_time_data['Vm'][i]),
                    sigma_factor * our_time_data.input_std_devs[0]
                )
                self.assertLessEqual(
                    abs(our_time_data.inputs[1][i] - correct_time_data['Va'][i]),
                    sigma_factor * our_time_data.input_std_devs[1]
                )
                self.assertLessEqual(
                    abs(our_time_data.outputs[0][i] - correct_time_data['Im'][i]),
                    sigma_factor * our_time_data.output_std_devs[0]
                )
                self.assertLessEqual(
                    abs(our_time_data.outputs[1][i] - correct_time_data['Ia'][i]),
                    sigma_factor * our_time_data.output_std_devs[1]
                )



class FreqDataTests(unittest.TestCase):
    """Check proper working of the data.FreqData class.

    The following features are tested:
    1. applying DFT to data in time domain;
    2. trimming data (in frequency domain);
    3. removing data which are located in forced oscillations band.
    """

    def _check_lengths(self, our_freq_data, correct_freq_data):
        assert len(correct_freq_data['freqs']) == len(correct_freq_data['Vm'])
        assert len(correct_freq_data['Vm']) == len(correct_freq_data['Va'])
        assert len(correct_freq_data['Va']) == len(correct_freq_data['Im'])
        assert len(correct_freq_data['Im']) == len(correct_freq_data['Ia'])

        n_freq_points = len(our_freq_data.freqs)
        self.assertEqual(n_freq_points, len(our_freq_data.inputs[0]))
        self.assertEqual(len(our_freq_data.inputs[0]), len(our_freq_data.inputs[1]))
        self.assertEqual(len(our_freq_data.inputs[1]), len(our_freq_data.outputs[0]))
        self.assertEqual(len(our_freq_data.outputs[0]), len(our_freq_data.outputs[1]))

        self.assertEqual(n_freq_points, len(correct_freq_data['freqs']))
        self.assertEqual(n_freq_points, len(correct_freq_data['Vm']))
        self.assertEqual(n_freq_points, len(correct_freq_data['Va']))
        self.assertEqual(n_freq_points, len(correct_freq_data['Im']))
        self.assertEqual(n_freq_points, len(correct_freq_data['Ia']))

    def _check_data(self, our_freq_data, correct_freq_data):
        self._check_lengths(our_freq_data, correct_freq_data)
        n_freqs = len(our_freq_data.freqs)
        for i in range(n_freqs):
            self.assertAlmostEqual(
                our_freq_data.freqs[i], correct_freq_data['freqs'][i],
                places=13
            )

        relative_precision = 8
        for i in range(n_freqs):
            self.assertAlmostEqual(
                our_freq_data.inputs[0][i].real / correct_freq_data['Vm'][i].real,
                1.0,
                places=relative_precision
            )
            self.assertAlmostEqual(
                our_freq_data.inputs[0][i].imag / correct_freq_data['Vm'][i].imag,
                1.0,
                places=relative_precision
            )

            self.assertAlmostEqual(
                our_freq_data.inputs[1][i].real / correct_freq_data['Va'][i].real,
                1.0,
                places=relative_precision
            )
            self.assertAlmostEqual(
                our_freq_data.inputs[1][i].imag / correct_freq_data['Va'][i].imag,
                1.0,
                places=relative_precision
            )

            self.assertAlmostEqual(
                our_freq_data.outputs[0][i].real / correct_freq_data['Im'][i].real,
                1.0,
                places=relative_precision
            )
            self.assertAlmostEqual(
                our_freq_data.outputs[0][i].imag / correct_freq_data['Im'][i].imag,
                1.0,
                places=relative_precision
            )

            self.assertAlmostEqual(
                our_freq_data.outputs[1][i].real / correct_freq_data['Ia'][i].real,
                1.0,
                places=relative_precision
            )
            self.assertAlmostEqual(
                our_freq_data.outputs[1][i].imag / correct_freq_data['Ia'][i].imag,
                1.0,
                places=relative_precision
            )

    def _check_std_deviations(self, our_freq_data, test_dir):
        all_correct_values = correct_data.get_correct_values(test_dir)
        correct_std_deviations = all_correct_values['freq_data_std_dev_eps']
        relative_precision = 13

        self.assertAlmostEqual(
            our_freq_data.input_std_devs[0] / correct_std_deviations['std_dev_eps_Vm'],
            1.0,
            places=relative_precision
        )
        self.assertAlmostEqual(
            our_freq_data.input_std_devs[1] / correct_std_deviations['std_dev_eps_Va'],
            1.0,
            places=relative_precision
        )
        self.assertAlmostEqual(
            our_freq_data.output_std_devs[0] / correct_std_deviations['std_dev_eps_Im'],
            1.0,
            places=relative_precision
        )
        self.assertAlmostEqual(
            our_freq_data.output_std_devs[1] / correct_std_deviations['std_dev_eps_Ia'],
            1.0,
            places=relative_precision
        )

    def test_fft_and_trimming(self):
        for test_dir in TEST_DIRS:
            our_freq_data = our_data.get_freq_data_after_fft_and_trimming(test_dir)
            correct_freq_data = correct_data.get_freq_data_after_fft_and_trimming(test_dir)

            # This assert have to be removed after extension of test sets
            assert len(our_freq_data.freqs) == 600

            self._check_std_deviations(our_freq_data, test_dir)
            self._check_data(
                our_freq_data=our_freq_data,
                correct_freq_data=correct_freq_data
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
                correct_freq_data=correct_freq_data
            )


class ObjectiveFunctionTests(unittest.TestCase):
    """Check the objective function (which should be minimized).

    Compute the objective function at some picked points and compare
    obtained values with true values (see test_dir/correct_values.json).
    """

    def _check_covariance_matrix(self, our_gamma_L, test_dir):
        # gamma_L is calculated at prior_mean point (see prior.json)
        correct_values = correct_data.get_correct_values(test_dir)
        correct_gamma_L = correct_values['CovarianceMatrix']

        self.assertEqual(our_gamma_L.shape[0], correct_gamma_L['size_y'])
        self.assertEqual(our_gamma_L.shape[1], correct_gamma_L['size_x'])

        self.assertEqual(our_gamma_L[0, 1], 0.0)
        self.assertEqual(our_gamma_L[1, 0], 0.0)
        self.assertEqual(our_gamma_L[0, 2], 0.0)
        self.assertEqual(our_gamma_L[2, 0], 0.0)
        self.assertEqual(our_gamma_L[1, 2], 0.0)
        self.assertEqual(our_gamma_L[2, 1], 0.0)

        for coords, matrix_element in correct_gamma_L['values'].items():
            y_coord = int(coords.split(',')[0]) - 1
            x_coord = int(coords.split(',')[1]) - 1
            self.assertAlmostEqual(
                our_gamma_L[y_coord, x_coord] / matrix_element, 1.0,
                places=13
            )

    def _check_objective_function_values(self, func, test_dir):
        correct_values = correct_data.get_correct_values(test_dir)
        correct_func_values = correct_values['ObjectiveFunction']['values']

        points = []
        correct_values = []
        for args, correct_func_value in correct_func_values.items():
            args_array = np.array([np.float(arg) for arg in args.split(',')])
            points.append(args_array)
            correct_values.append(correct_func_value)
            self.assertAlmostEqual(
                func.compute(args_array) / correct_func_value, 1.0,
                places=12
            )

        for _ in range(40):
            our_values = func.compute(np.array(points))
            for i in range(len(points)):
                self.assertAlmostEqual(
                    our_values[i] / correct_values[i], 1.0,
                    places=12
                )

        # points = np.array(points * 10)
        # start_time = time.time()
        # func.compute(points)
        # finish_time = time.time()
        # average_time = (finish_time - start_time) / len(points)
        # print('AVERAGE TIME TO COMPUTE f:', average_time)

    def test_objective_function(self):
        for test_dir in TEST_DIRS:
            correct_prepared_freq_data = (
                correct_data.get_prepared_freq_data(test_dir)
            )

            # perturbed generator's parameters
            json_prior = None
            with open(os.path.join(test_dir, 'prior.json'), 'r') as prior_file:
                json_prior = json.load(prior_file)
            gen_params_prior_mean = np.array([
                json_prior['prior_mean']['D_Ya'],
                json_prior['prior_mean']['Ef_a'],
                json_prior['prior_mean']['M_Ya'],
                json_prior['prior_mean']['X_Ya']
            ])
            gen_params_prior_std_dev = np.array([
                json_prior['prior_std_dev']['D_Ya'],
                json_prior['prior_std_dev']['Ef_a'],
                json_prior['prior_std_dev']['M_Ya'],
                json_prior['prior_std_dev']['X_Ya']
            ])

            # f -- objective function to minimize
            f = objective_function.ObjectiveFunction(
                freq_data=correct_prepared_freq_data,
                admittance_matrix=admittance_matrix.AdmittanceMatrix(),
                prior_params=gen_params_prior_mean,
                prior_params_std=gen_params_prior_std_dev
            )

            self._check_covariance_matrix(
                our_gamma_L=f._gamma_L.compute(gen_params_prior_mean),
                test_dir=test_dir
            )
            self._check_objective_function_values(
                func=f,
                test_dir=test_dir
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)

