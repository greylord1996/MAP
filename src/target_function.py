from create_admittance_matrix import AdmittanceMatrix
from sympy import *
from dynamic_equations_to_simulate import OdeSolver
import data
import numpy as np

class ResidualVector:

    def __init__(self):
        pass


class CovarianceMatrix:

    def __init__(self, freq_data):
        self.std_eps_Vm = freq_data.std_w_Vm
        self.std_eps_Va = freq_data.std_w_Va
        self.std_eps_Im = freq_data.std_w_Im
        self.std_eps_Ia = freq_data.std_w_Ia
        self.freqs = freq_data.freqs
        self.time_points_len = len(self.freqs)


        self.admittance_matrix = AdmittanceMatrix().Ys
        self.admittance_matrix_compute = lambdify(('Ef_a', 'D_Ya', 'M_Ya', 'X_Ya', 'Omega_a'), self.admittance_matrix, 'numpy')

        self.Y11 = self.admittance_matrix[0, 0]
        self.Y12 = self.admittance_matrix[0, 1]
        self.Y21 = self.admittance_matrix[1, 0]
        self.Y22 = self.admittance_matrix[1, 1]

        self.Y11r = re(self.Y11)
        self.Y11i = im(self.Y11)
        self.Y12r = re(self.Y12)
        self.Y12i = im(self.Y12)
        self.Y21r = re(self.Y21)
        self.Y21i = im(self.Y21)
        self.Y22r = re(self.Y22)
        self.Y22i = im(self.Y22)

        self.gamma_NrNr = None
        self.gamma_NrQr = None

    def _init_gamma_NrNr(self):
        NrNr = (
            self.std_eps_Im**2
            + self.std_eps_Vm**2 * (self.Y11r**2 + self.Y11i**2)
            + self.std_eps_Va**2 * (self.Y12r**2 + self.Y12i**2)
        )

        # NrNr_compute_omega = lambdify('Omega_a', NrNr, 'numpy')

        self.gamma_NrNr = zeros((self.time_points_len + 1) // 2)
        for i in range((self.time_points_len + 1) // 2):
            for j in range((self.time_points_len + 1) // 2):
                if i == j:
                    self.gamma_NrNr[i, j] = NrNr.subs('Omega_a', self.freqs[i])

        # self.gamma_NrNr = self.gamma_NrNr.subs('Omega_a', omega)

    def _init_gamma_NrQr(self):
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