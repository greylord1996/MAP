import numpy as np
import sympy

from dynamic_equations_to_simulate import OdeSolver
from create_admittance_matrix import AdmittanceMatrix
import data
import time



class ResidualVector:

    def __init__(self, freq_data, is_actual=True):
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

        Vmr = np.real(freq_data.Vm)
        Vmi = np.imag(freq_data.Vm)
        Var = np.real(freq_data.Va)
        Vai = np.imag(freq_data.Va)
        Imr = np.real(freq_data.Im)
        Imi = np.imag(freq_data.Im)
        Iar = np.real(freq_data.Ia)
        Iai = np.imag(freq_data.Ia)

        # Mr = Imr - Y11r. * Vmr + Y11i. * Vmi - Y12r. * Var + Y12i. * Vai;
        # Mi = Imi - Y11i. * Vmr - Y11r. * Vmi - Y12i. * Var - Y12r. * Vai;
        # Pr = Iar - Y21r. * Vmr + Y21i. * Vmi - Y22r. * Var + Y22i. * Vai;
        # Pi = Iai - Y21i. * Vmr - Y21r. * Vmi - Y22i. * Var - Y22r. * Vai;




class CovarianceMatrix:

    def __init__(self, freq_data):
        std_eps_Vm = freq_data.std_w_Vm
        std_eps_Va = freq_data.std_w_Va
        std_eps_Im = freq_data.std_w_Im
        std_eps_Ia = freq_data.std_w_Ia

        self.freqs = freq_data.freqs
        self.admittance_matrix = AdmittanceMatrix().Ys

        Y11 = self.admittance_matrix[0, 0]
        Y12 = self.admittance_matrix[0, 1]
        Y21 = self.admittance_matrix[1, 0]
        Y22 = self.admittance_matrix[1, 1]

        Y11r, Y11i = sympy.re(Y11), sympy.im(Y11)
        Y12r, Y12i = sympy.re(Y12), sympy.im(Y12)
        Y21r, Y21i = sympy.re(Y21), sympy.im(Y21)
        Y22r, Y22i = sympy.re(Y22), sympy.im(Y22)

        Ef_a, D_Ya, M_Ya, X_Ya, Omega_a = sympy.symbols(
            'Ef_a D_Ya M_Ya X_Ya Omega_a'
        )

        self.NrNr = sympy.lambdify(
            [Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            (
                std_eps_Im**2
                + std_eps_Vm**2 * (Y11r**2 + Y11i**2)
                + std_eps_Va**2 * (Y12r**2 + Y12i**2)
            ),
            modules='numexpr'
        )

        self.NrQr = sympy.lambdify(
            [Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            (
                std_eps_Vm**2 * (Y11r*Y21r + Y11i*Y21i) +
                std_eps_Va**2 * (Y12r*Y22r + Y12i*Y22i)
            ),
            modules='numexpr'
        )

        self.NrQi = sympy.lambdify(
            [Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            (
                std_eps_Vm**2 * (Y11r*Y21i - Y11i*Y21r) +
                std_eps_Va**2 * (Y12r*Y22i - Y12i*Y22r)
            ),
            modules='numexpr'
        )

        self.NiNi = sympy.lambdify(
            [Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            (
                std_eps_Im**2
                + std_eps_Vm**2 * (Y11r**2 + Y11i**2)
                + std_eps_Va**2 * (Y12r**2 + Y12i**2)
            ),
            modules='numexpr'
        )

        self.NiQr = sympy.lambdify(
            [Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            (
                std_eps_Vm**2 * (Y11i*Y21r - Y11r*Y21i) +
                std_eps_Va**2 * (Y12i*Y22r - Y12r*Y22i)
            ),
            modules='numexpr'
        )

        self.NiQi = sympy.lambdify(
            [Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            (
                std_eps_Vm**2 * (Y11i*Y21i + Y11r*Y21r) +
                std_eps_Va**2 * (Y12i*Y22i + Y12r*Y22r)
            ),
            modules='numexpr'
        )

        self.QrNr = self.NrQr
        self.QrNi = self.NiQr

        self.QrQr = sympy.lambdify(
            [Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            (
                std_eps_Ia**2
                + std_eps_Vm**2 * (Y21r**2 + Y21i**2)
                + std_eps_Va**2 * (Y22r**2 + Y22i**2)
            ),
            modules='numexpr'
        )

        self.QiNr = self.NrQi
        self.QiNi = self.NiQi

        self.QiQi = sympy.lambdify(
            [Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            (
                std_eps_Ia**2
                + std_eps_Vm**2 * (Y21i**2 + Y21r**2)
                + std_eps_Va**2 * (Y22r**2 + Y22i**2)
            ),
            modules='numexpr'
        )


    def compute(self, generator_params):
        gamma_NrNr = np.diag([
            self.NrNr(*generator_params, freq) for freq in self.freqs
        ])
        gamma_NrQr = np.diag([
            self.NrQr(*generator_params, freq) for freq in self.freqs
        ])
        gamma_NrQi = np.diag([
            self.NrQi(*generator_params, freq) for freq in self.freqs
        ])
        gamma_NiNi = np.diag([
            self.NrNr(*generator_params, freq) for freq in self.freqs
        ])
        gamma_NiQr = np.diag([
            self.NrNr(*generator_params, freq) for freq in self.freqs
        ])
        gamma_NiQi = np.diag([
            self.NrNr(*generator_params, freq) for freq in self.freqs
        ])
        gamma_QrQr = np.diag([
            self.NrNr(*generator_params, freq) for freq in self.freqs
        ])
        gamma_QiQi = np.diag([
            self.NrNr(*generator_params, freq) for freq in self.freqs
        ])
        gamma_QrNr = gamma_NrQr
        gamma_QrNi = gamma_NiQr
        gamma_QiNr = gamma_NrQi
        gamma_QiNi = gamma_NiQi

        zero_matrix = np.zeros((len(self.freqs), len(self.freqs)), dtype=np.float)
        gamma = np.block([
            [gamma_NrNr, zero_matrix, gamma_NrQr, gamma_NrQi],
            [zero_matrix, gamma_NiNi, gamma_NiQr, gamma_NiQi],
            [gamma_QrNr, gamma_QrNi, gamma_QrQr, zero_matrix],
            [gamma_QiNr, gamma_QiNi, zero_matrix, gamma_QiQi]
        ])
        return gamma



    def compute_and_inverse(self, generator_params):
        return np.linalg.inv(self.compute(generator_params))






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


start_time = time.time()
print('-------------------')

cov_obj.compute([1.99987878, 2.1234567, 3.786787868, 4.123232])
# cov_obj.compute_and_inverse([1, 2, 3, 4])

print("@@@ %s seconds ---" % (time.time() - start_time))
print('-------------------')