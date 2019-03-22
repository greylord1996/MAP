import os
import os.path
import sympy
import pickle

from dynamic_equations_to_simulate import OdeSolver
from create_admittance_matrix import AdmittanceMatrix
import data



class ResidualVector:

    def __init__(self):
        pass



class CovarianceMatrix:

    def __init__(self, freq_data, is_actual=True):
        std_eps_Vm = freq_data.std_w_Vm
        std_eps_Va = freq_data.std_w_Va
        std_eps_Im = freq_data.std_w_Im
        std_eps_Ia = freq_data.std_w_Ia

        freqs = freq_data.freqs
        admittance_matrix = AdmittanceMatrix().Ys

        Y11 = admittance_matrix[0, 0]
        Y12 = admittance_matrix[0, 1]
        Y21 = admittance_matrix[1, 0]
        Y22 = admittance_matrix[1, 1]

        Y11r = sympy.re(Y11)
        Y11i = sympy.im(Y11)
        Y12r = sympy.re(Y12)
        Y12i = sympy.im(Y12)
        Y21r = sympy.re(Y21)
        Y21i = sympy.im(Y21)
        Y22r = sympy.re(Y22)
        Y22i = sympy.im(Y22)

        path_to_this_file = os.path.abspath(os.path.dirname(__file__))
        path_to_matrix_file = os.path.join(
            path_to_this_file,
            '..', 'data', 'precomputed', 'covariance_matrix.pickle'
        )

        if not is_actual:
            NrNr = (
                std_eps_Im**2
                + std_eps_Vm**2 * (Y11r**2 + Y11i**2)
                + std_eps_Va**2 * (Y12r**2 + Y12i**2)
            )
            gamma_NrNr = sympy.zeros(len(freqs))
            for i in range(len(freqs)):
                gamma_NrNr[i, i] = NrNr.subs('Omega_a', freqs[i])

            NrQr = (
                std_eps_Vm**2 * (Y11r*Y21r + Y11i*Y21i) +
                std_eps_Va**2 * (Y12r*Y22r + Y12i*Y22i)
            )
            gamma_NrQr = sympy.zeros(len(freqs))
            for i in range(len(freqs)):
                gamma_NrQr[i, i] = NrQr.subs('Omega_a', freqs[i])

            NrQi = (
                std_eps_Vm**2 * (Y11r*Y21i - Y11i*Y21r) +
                std_eps_Va**2 * (Y12r*Y22i - Y12i*Y22r)
            )
            gamma_NrQi = sympy.zeros(len(freqs))
            for i in range(len(freqs)):
                gamma_NrQi[i, i] = NrQi.subs('Omega_a', freqs[i])

            NiNi = (
                std_eps_Im**2
                + std_eps_Vm**2 * (Y11r**2 + Y11i**2)
                + std_eps_Va**2 * (Y12r**2 + Y12i**2)
            )
            gamma_NiNi = sympy.zeros(len(freqs))
            for i in range(len(freqs)):
                gamma_NiNi[i, i] = NiNi.subs('Omega_a', freqs[i])

            NiQr = (
                std_eps_Vm**2 * (Y11i*Y21r - Y11r*Y21i) +
                std_eps_Va**2 * (Y12i*Y22r - Y12r*Y22i)
            )
            gamma_NiQr = sympy.zeros(len(freqs))
            for i in range(len(freqs)):
                gamma_NiQr[i, i] = NiQr.subs('Omega_a', freqs[i])

            NiQi = (
                std_eps_Vm**2 * (Y11i*Y21i + Y11r*Y21r) +
                std_eps_Va**2 * (Y12i*Y22i + Y12r*Y22r)
            )
            gamma_NiQi = sympy.zeros(len(freqs))
            for i in range(len(freqs)):
                gamma_NiQi[i, i] = NiQi.subs('Omega_a', freqs[i])

            gamma_QrNr = gamma_NrQr
            gamma_QrNi = gamma_NiQr

            QrQr = (
                std_eps_Ia**2
                + std_eps_Vm**2 * (Y21r**2 + Y21i**2)
                + std_eps_Va**2 * (Y22r**2 + Y22i**2)
            )
            gamma_QrQr = sympy.zeros(len(freqs))
            for i in range(len(freqs)):
                gamma_QrQr[i, i] = QrQr.subs('Omega_a', freqs[i])

            gamma_QiNr = gamma_NrQi
            gamma_QiNi = gamma_NiQi

            QiQi = (
                    std_eps_Ia**2
                    + std_eps_Vm**2 * (Y21i**2 + Y21r**2)
                    + std_eps_Va**2 * (Y22r**2 + Y22i**2)
            )
            gamma_QiQi = sympy.zeros(len(freqs))
            for i in range(len(freqs)):
                gamma_QiQi[i, i] = QiQi.subs('Omega_a', freqs[i])

            # pickle.dump(self.gamma, open(path_to_matrix_file, 'wb'))

        else:
            # Load from disk
            # self.gamma = pickle.load(open(path_to_matrix_file, 'rb'))
            pass


    def compute(self, generator_params):
        pass


    def compute_and_inverse(self, generator_params):
        pass






WN = {'rnd_amp': 0.00}
GP = {'d_2': 0.25, 'e_2': 1, 'm_2': 1, 'x_d2': 0.01, 'ic_d2': 1}
IP = {'dt_step': 0.05, 'df_length': 100}
OP = {'osc_amp': 2.00, 'osc_freq': 0.005}


solver = OdeSolver(WN, GP, OP, IP)
solver.solve()

solver.simulate_time_data()
time_data = data.TimeData(
        Vm_time_data=solver.Vc1_abs,
        Va_time_data=solver.Vc1_angle,
        Im_time_data=solver.Ig_abs,
        Ia_time_data=solver.Ig_angle,
        dt=solver.dt
    )

time_data.apply_white_noise(snr=45.0, d_coi=0.0)

print('Vm_time_data =', time_data.Vm)
print('Im_time_data =', time_data.Im)
print('Va_time_data =', time_data.Va)
print('Ia_time_data =', time_data.Ia)

freq_data = data.FreqData(time_data)

cov_obj = CovarianceMatrix(freq_data)