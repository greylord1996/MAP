import numpy as np
import scipy as sp
import sympy

from dynamic_equations_to_simulate import OdeSolver
from create_admittance_matrix import AdmittanceMatrix
import data
import utils



class UncertainGeneratorParameters:
    """Wrapper around four uncertain parameters of a generator.

    Unfortunately, it is not allowed to simply add or remove uncertain
    generator parmeters from this class. If you want add or remove some
    such parameters, it will require changing some code in this file
    (you should pay attention to substitution of the fields to
    symbolic expressions). To sum up, this class is only for readability
    of the code.
    """

    def __init__(self, Ef_a, D_Ya, X_Ya, M_Ya):
        """Inits all fields which are uncertain parameters of a generator.

        Args:
            Ef_a (float): generator field voltage magnitude
            D_Ya (float): generator damping
            X_Ya (float): generator reactance (inductance)
            M_Ya (float): ???
        """
        self.Ef_a = Ef_a
        self.D_Ya = D_Ya
        self.X_Ya = X_Ya
        self.M_Ya = M_Ya



@utils.singleton
class ResidualVector:
    """Wrapper for calculations of R (residual vector)."""

    def __init__(self, freq_data):
        """Prepares for computing the covariance matrix.

        Args:
            freq_data (class FreqData): data in frequency domain

        Attributes:
            freqs (list of floats): frequencies in frequency domain
            admittance_matrix (class AdmittanceMatrix): admittance_matrix
            NrNr (function): for computing diagonal elements of gamma_NrNr
            NrQr (function): for computing diagonal elements of gamma_NrQr
            NrQi (function): for computing diagonal elements of gamma_NrQi
            NiNi (function): for computing diagonal elements of gamma_NiNi
            NiQr (function): for computing diagonal elements of gamma_NiQr
            NiQi (function): for computing diagonal elements of gamma_NiQi
            QrQr (function): for computing diagonal elements of gamma_QrQr
            QiQi (function): for computing diagonal elements of gamma_QiQi

        Note:
            attributes QrNr, QrNi, QiNr, QiNi are absent.
            It is not necessary to store them due to the following equations:
                gamma_QrNr = gamma_NrQr
                gamma_QrNi = gamma_NiQr
                gamma_QiNr = gamma_NrQi
                gamma_QiNi = gamma_NiQi
            This fact will be used in the 'compute' method.
        """
        self.freq_data = freq_data

        admittance_matrix = AdmittanceMatrix().Ys
        Y11 = admittance_matrix[0, 0]
        Y12 = admittance_matrix[0, 1]
        Y21 = admittance_matrix[1, 0]
        Y22 = admittance_matrix[1, 1]

        Y11r, Y11i = sympy.re(Y11), sympy.im(Y11)
        Y12r, Y12i = sympy.re(Y12), sympy.im(Y12)
        Y21r, Y21i = sympy.re(Y21), sympy.im(Y21)
        Y22r, Y22i = sympy.re(Y22), sympy.im(Y22)

        Vm, Va, Im, Ia = sympy.symbols('Vm Va Im Ia')
        Vmr, Vmi = sympy.re(Vm), sympy.im(Vm)
        Var, Vai = sympy.re(Va), sympy.im(Va)
        Imr, Imi = sympy.re(Im), sympy.im(Im)
        Iar, Iai = sympy.re(Ia), sympy.im(Ia)

        Ef_a, D_Ya, X_Ya, M_Ya, Omega_a = sympy.symbols(
            'Ef_a D_Ya X_Ya M_Ya Omega_a',
            real=True
        )

        self.Mr = sympy.lambdify(
            args=[Vm, Va, Im, Ia, Ef_a, D_Ya, X_Ya, M_Ya, Omega_a],
            expr=(Imr - Y11r*Vmr + Y11i*Vmi - Y12r*Var + Y12i*Vai),
            modules='numexpr'
        )

        self.Mi = sympy.lambdify(
            args=[Vm, Va, Im, Ia, Ef_a, D_Ya, X_Ya, M_Ya, Omega_a],
            expr=(Imi - Y11i*Vmr - Y11r*Vmi - Y12i*Var - Y12r*Vai),
            modules='numexpr'
        )

        self.Pr = sympy.lambdify(
            args=[Vm, Va, Im, Ia, Ef_a, D_Ya, X_Ya, M_Ya, Omega_a],
            expr=(Iar - Y21r*Vmr + Y21i*Vmi - Y22r*Var + Y22i*Vai),
            modules='numexpr'
        )

        self.Pi = sympy.lambdify(
            args=[Vm, Va, Im, Ia, Ef_a, D_Ya, X_Ya, M_Ya, Omega_a],
            expr=(Iai - Y21i*Vmr - Y21r*Vmi - Y22i*Var - Y22r*Vai),
            modules='numexpr'
        )


    def compute(self, uncertain_generator_params):
        """"""
        Ef_a = uncertain_generator_params.Ef_a
        D_Ya = uncertain_generator_params.D_Ya
        X_Ya = uncertain_generator_params.X_Ya
        M_Ya = uncertain_generator_params.M_Ya

        freq_data_len = len(self.freq_data.freqs)
        Mr = np.zeros(freq_data_len)
        Mi = np.zeros(freq_data_len)
        Pr = np.zeros(freq_data_len)
        Pi = np.zeros(freq_data_len)

        for i in range(freq_data_len):
            Mr[i] = self.Mr(
                freq_data.Vm[i], freq_data.Va[i],
                freq_data.Im[i], freq_data.Ia[i],
                Ef_a, D_Ya, X_Ya, M_Ya,
                freq_data.freqs[i]
            )

            Mi[i] = self.Mi(
                freq_data.Vm[i], freq_data.Va[i],
                freq_data.Im[i], freq_data.Ia[i],
                Ef_a, D_Ya, X_Ya, M_Ya,
                freq_data.freqs[i]
            )

            Pr[i] = self.Pr(
                freq_data.Vm[i], freq_data.Va[i],
                freq_data.Im[i], freq_data.Ia[i],
                Ef_a, D_Ya, X_Ya, M_Ya,
                freq_data.freqs[i]
            )

            Pi[i] = self.Pi(
                freq_data.Vm[i], freq_data.Va[i],
                freq_data.Im[i], freq_data.Ia[i],
                Ef_a, D_Ya, X_Ya, M_Ya,
                freq_data.freqs[i]
            )

        # Build and return R -- residual vector
        return np.concatenate([Mr, Mi, Pr, Pi])



@utils.singleton
class CovarianceMatrix:
    """Wrapper for calculations of covariance matrix."""

    def __init__(self, freq_data):
        """Prepares for computing the covariance matrix.

        Args:
            freq_data (class FreqData): data in frequency domain

        Attributes:
            freqs (list of floats): frequencies in frequency domain
            NrNr (function): for computing diagonal elements of gamma_NrNr
            NrQr (function): for computing diagonal elements of gamma_NrQr
            NrQi (function): for computing diagonal elements of gamma_NrQi
            NiNi (function): for computing diagonal elements of gamma_NiNi
            NiQr (function): for computing diagonal elements of gamma_NiQr
            NiQi (function): for computing diagonal elements of gamma_NiQi
            QrQr (function): for computing diagonal elements of gamma_QrQr
            QiQi (function): for computing diagonal elements of gamma_QiQi

        Note:
            attributes QrNr, QrNi, QiNr, QiNi are absent.
            It is not necessary to store them due to the following equations:
                gamma_QrNr = gamma_NrQr
                gamma_QrNi = gamma_NiQr
                gamma_QiNr = gamma_NrQi
                gamma_QiNi = gamma_NiQi
            This fact will be used in the 'compute' method.
        """
        self.freqs = freq_data.freqs

        std_eps_Vm = freq_data.std_w_Vm
        std_eps_Va = freq_data.std_w_Va
        std_eps_Im = freq_data.std_w_Im
        std_eps_Ia = freq_data.std_w_Ia

        admittance_matrix = AdmittanceMatrix().Ys
        Y11 = admittance_matrix[0, 0]
        Y12 = admittance_matrix[0, 1]
        Y21 = admittance_matrix[1, 0]
        Y22 = admittance_matrix[1, 1]

        Y11r, Y11i = sympy.re(Y11), sympy.im(Y11)
        Y12r, Y12i = sympy.re(Y12), sympy.im(Y12)
        Y21r, Y21i = sympy.re(Y21), sympy.im(Y21)
        Y22r, Y22i = sympy.re(Y22), sympy.im(Y22)

        Ef_a, D_Ya, X_Ya, M_Ya, Omega_a = sympy.symbols(
            'Ef_a D_Ya X_Ya M_Ya Omega_a',
            real=True
        )

        self.NrNr = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Im**2
                + std_eps_Vm**2 * (Y11r**2 + Y11i**2)
                + std_eps_Va**2 * (Y12r**2 + Y12i**2)
            ),
            modules='numexpr'
        )

        self.NrQr = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Vm**2 * (Y11r*Y21r + Y11i*Y21i) +
                std_eps_Va**2 * (Y12r*Y22r + Y12i*Y22i)
            ),
            modules='numexpr'
        )

        self.NrQi = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Vm**2 * (Y11r*Y21i - Y11i*Y21r) +
                std_eps_Va**2 * (Y12r*Y22i - Y12i*Y22r)
            ),
            modules='numexpr'
        )

        self.NiNi = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Im**2
                + std_eps_Vm**2 * (Y11r**2 + Y11i**2)
                + std_eps_Va**2 * (Y12r**2 + Y12i**2)
            ),
            modules='numexpr'
        )

        self.NiQr = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Vm**2 * (Y11i*Y21r - Y11r*Y21i) +
                std_eps_Va**2 * (Y12i*Y22r - Y12r*Y22i)
            ),
            modules='numexpr'
        )

        self.NiQi = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Vm**2 * (Y11i*Y21i + Y11r*Y21r) +
                std_eps_Va**2 * (Y12i*Y22i + Y12r*Y22r)
            ),
            modules='numexpr'
        )

        self.QrQr = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Ia**2
                + std_eps_Vm**2 * (Y21r**2 + Y21i**2)
                + std_eps_Va**2 * (Y22r**2 + Y22i**2)
            ),
            modules='numexpr'
        )

        self.QiQi = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Ia**2
                + std_eps_Vm**2 * (Y21i**2 + Y21r**2)
                + std_eps_Va**2 * (Y22r**2 + Y22i**2)
            ),
            modules='numexpr'
        )

        # self.QrNr = self.NrQr
        # self.QrNi = self.NiQr
        # self.QiNr = self.NrQi
        # self.QiNi = self.NiQi


    def compute(self, uncertain_generator_params):
        """Computes the covariance matrix in the 'generator_params' point.

        Builds and computes the numerical value of the covariance matrix
        in the point specified by 'generator_params'. The result matrix
        will have sizes (4K+4) * (4K+4) and contain only numbers (not symbols).

        Args:
            uncertain_generator_params (class UncertainGeneratorParameters):
                current uncertain parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            gamma (numpy.ndarray): value of the covariance matrix
        """
        Ef_a = uncertain_generator_params.Ef_a
        D_Ya = uncertain_generator_params.D_Ya
        X_Ya = uncertain_generator_params.X_Ya
        M_Ya = uncertain_generator_params.M_Ya

        gamma_NrNr = np.diag([
            self.NrNr(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self.freqs
        ])
        gamma_NrQr = np.diag([
            self.NrQr(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self.freqs
        ])
        gamma_NrQi = np.diag([
            self.NrQi(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self.freqs
        ])
        gamma_NiNi = np.diag([
            self.NiNi(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self.freqs
        ])
        gamma_NiQr = np.diag([
            self.NiQr(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self.freqs
        ])
        gamma_NiQi = np.diag([
            self.NiQi(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self.freqs
        ])
        gamma_QrQr = np.diag([
            self.QrQr(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self.freqs
        ])
        gamma_QiQi = np.diag([
            self.QiQi(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self.freqs
        ])
        gamma_QrNr = gamma_NrQr
        gamma_QrNi = gamma_NiQr
        gamma_QiNr = gamma_NrQi
        gamma_QiNi = gamma_NiQi

        zero_matrix = np.zeros((len(self.freqs), len(self.freqs)))
        gamma = np.block([
            [gamma_NrNr, zero_matrix, gamma_NrQr, gamma_NrQi],
            [zero_matrix, gamma_NiNi, gamma_NiQr, gamma_NiQi],
            [gamma_QrNr, gamma_QrNi, gamma_QrQr, zero_matrix],
            [gamma_QiNr, gamma_QiNi, zero_matrix, gamma_QiQi]
        ])
        return gamma


    def compute_and_inverse(self, uncertain_generator_params):
        """Computes the inversed covariance matrix.

        Does exactly the same as 'compute' method but after computing
        the covariance matrix this method make the calculated matrix inversed
        and returns it.

        Args:
            uncertain_generator_params (class UncertainGeneratorParameters):
                current parameters of a generator (at the current step of
                an optimization routine)

        Returns:
            gamma (numpy.ndarray): value of the inversed covariance matrix
        """
        return sp.sparse.linalg.inv(sp.sparse.csc_matrix(
            self.compute(uncertain_generator_params)
        )).toarray()



@utils.singleton
class ObjectiveFunction:
    """"""

    def __init__(self):
        """"""
        pass

    def __call__(self):
        """"""
        pass




# ----------------------------------------------------------------
# ----------------------- Testing now ----------------------------
# ----------------------------------------------------------------

import time


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

# print('Vm_time_data =', time_data.Vm)
# print('Im_time_data =', time_data.Im)
# print('Va_time_data =', time_data.Va)
# print('Ia_time_data =', time_data.Ia)

freq_data = data.FreqData(time_data)

print('===========================================')



start_time = time.time()
R = ResidualVector(freq_data)
print("constructing R : %s seconds" % (time.time() - start_time))

uncertain_generator_params = UncertainGeneratorParameters(
    Ef_a=1.0, D_Ya=2.0, X_Ya=3.0, M_Ya=4.0
)

start_time = time.time()
computed_R = R.compute(uncertain_generator_params)
print('len(R) =', len(computed_R))
print(computed_R)
print("computing R : %s seconds" % (time.time() - start_time))

start_time = time.time()
cov_obj = CovarianceMatrix(freq_data)
print("constructing gamma : %s seconds" % (time.time() - start_time))


#
#
# start_time = time.time()
# print('-------------------')
#
# x_test = cov_obj.compute([1.99987878, 2.1234567, 3.786787868, 4.123232])
# # cov_obj.compute_and_inverse([1, 2, 3, 4])
#
# print("@@@ %s seconds --- covariance matrix" % (time.time() - start_time))
# print('-------------------')
#
#
# start_time = time.time()
# print('-------------------')
#
# x_test_inv = sp.linalg.inv(x_test)
# print("@@@ %s seconds --- covariance matrix inverse" % (time.time() - start_time))
# print('-------------------')
#
# start_time = time.time()
# print('-------------------')
#
# x_test_inv_sparse = cov_obj.compute_and_inverse([1, 2, 3, 4])
# print("@@@ %s seconds --- covariance matrix inverse sparse" % (time.time() - start_time))
# print('-------------------')
