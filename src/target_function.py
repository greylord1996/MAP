import numpy as np
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
        self.std_eps_Vm = freq_data.std_w_Vm
        self.std_eps_Va = freq_data.std_w_Va
        self.std_eps_Im = freq_data.std_w_Im
        self.std_eps_Ia = freq_data.std_w_Ia

        self.freqs = freq_data.freqs
        self.admittance_matrix = AdmittanceMatrix().Ys

        self.Y11 = self.admittance_matrix[0, 0]
        self.Y12 = self.admittance_matrix[0, 1]
        self.Y21 = self.admittance_matrix[1, 0]
        self.Y22 = self.admittance_matrix[1, 1]

        self.Y11r = sympy.re(self.Y11)
        self.Y11i = sympy.im(self.Y11)
        self.Y12r = sympy.re(self.Y12)
        self.Y12i = sympy.im(self.Y12)
        self.Y21r = sympy.re(self.Y21)
        self.Y21i = sympy.im(self.Y21)
        self.Y22r = sympy.re(self.Y22)
        self.Y22i = sympy.im(self.Y22)

        if not is_actual:
            NrNr = (
                self.std_eps_Im ** 2
                + self.std_eps_Vm ** 2 * (self.Y11r ** 2 + self.Y11i ** 2)
                + self.std_eps_Va ** 2 * (self.Y12r ** 2 + self.Y12i ** 2)
            )
            self.gamma_NrNr = sympy.zeros(len(self.freqs))
            for i in range(len(self.freqs)):
                self.gamma_NrNr[i, i] = NrNr.subs('Omega_a', self.freqs[i])

        else:
            # Load from disk
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