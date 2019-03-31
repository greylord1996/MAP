import numpy as np
import scipy as sp
import sympy

from dynamic_equations_to_simulate import OdeSolver
from create_admittance_matrix import AdmittanceMatrix
import data
import utils



class UncertainGeneratorParameters:
    """Wrapper around 4 uncertain parameters of a generator.

    Unfortunately, it is not allowed to simply add or remove uncertain
    generator parameters from this class. If you want add or remove some
    such parameters, it will require changing some code in this file
    (you should pay attention to substitution of the fields to
    symbolic expressions). To sum up, this class is only for readability
    of the code and is not convenient for extensibility.

    Attributes:
        Ef_a (float): generator field voltage magnitude
        D_Ya (float): generator damping
        X_Ya (float): generator reactance (inductance)
        M_Ya (float): ???

        std_dev_Ef_a (float): standard variance of Ef_a
        std_dev_D_Ya (float): standard variance of D_Ya
        std_dev_X_Ya (float): standard variance of X_Ya
        std_dev_M_Ya (float): standard variance of M_Ya
    """

    def __init__(self, Ef_a, D_Ya, X_Ya, M_Ya,
                 std_dev_Ef_a, std_dev_D_Ya, std_dev_X_Ya, std_dev_M_Ya):
        """Inits all fields which are uncertain parameters of a generator.

        This method requires prior values of uncertain parameters of a
        generator and their standard variances. These values will be updated
        after finishing an optimization routine.
        """
        self.Ef_a = Ef_a
        self.D_Ya = D_Ya
        self.X_Ya = X_Ya
        self.M_Ya = M_Ya

        self.std_dev_Ef_a = std_dev_Ef_a
        self.std_dev_D_Ya = std_dev_D_Ya
        self.std_dev_X_Ya = std_dev_X_Ya
        self.std_dev_M_Ya = std_dev_M_Ya


    def get_as_array(self):
        """Returns generator parameters in a numpy.array."""
        return np.array([self.Ef_a, self.D_Ya, self.X_Ya, self.M_Ya])



@utils.singleton
class ResidualVector:
    """Wrapper for calculations of R (residual vector).

    Attributes:
        _freq_data (class FreqData): data in frequency domain
        _Mr (function): for computing elements of vector Mr
        _Mi (function): for computing elements of vector Mi
        _Pr (function): for computing elements of vector Pr
        _Pi (function): for computing elements of vector Pi
    """

    def __init__(self, freq_data):
        """Prepares for computing the covariance matrix in a given point.

        Args:
            freq_data (class FreqData): data in frequency domain
        """
        self._freq_data = freq_data

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

        self._Mr = sympy.lambdify(
            args=[Vm, Va, Im, Ia, Ef_a, D_Ya, X_Ya, M_Ya, Omega_a],
            expr=(Imr - Y11r*Vmr + Y11i*Vmi - Y12r*Var + Y12i*Vai),
            modules='numexpr'
        )

        self._Mi = sympy.lambdify(
            args=[Vm, Va, Im, Ia, Ef_a, D_Ya, X_Ya, M_Ya, Omega_a],
            expr=(Imi - Y11i*Vmr - Y11r*Vmi - Y12i*Var - Y12r*Vai),
            modules='numexpr'
        )

        self._Pr = sympy.lambdify(
            args=[Vm, Va, Im, Ia, Ef_a, D_Ya, X_Ya, M_Ya, Omega_a],
            expr=(Iar - Y21r*Vmr + Y21i*Vmi - Y22r*Var + Y22i*Vai),
            modules='numexpr'
        )

        self._Pi = sympy.lambdify(
            args=[Vm, Va, Im, Ia, Ef_a, D_Ya, X_Ya, M_Ya, Omega_a],
            expr=(Iai - Y21i*Vmr - Y21r*Vmi - Y22i*Var - Y22r*Vai),
            modules='numexpr'
        )


    def compute(self, uncertain_gen_params):
        """Computes the residual vector in the given point.

        Args:
            uncertain_gen_params (class UncertainGeneratorParameters):
                current uncertain parameters of a generator
                (at the current step of an optimization routine)
        """
        Ef_a = uncertain_gen_params.Ef_a
        D_Ya = uncertain_gen_params.D_Ya
        X_Ya = uncertain_gen_params.X_Ya
        M_Ya = uncertain_gen_params.M_Ya

        freq_data_len = len(self._freq_data.freqs)
        Mr = np.zeros(freq_data_len)
        Mi = np.zeros(freq_data_len)
        Pr = np.zeros(freq_data_len)
        Pi = np.zeros(freq_data_len)

        for i in range(freq_data_len):
            Mr[i] = self._Mr(
                self._freq_data.Vm[i], self._freq_data.Va[i],
                self._freq_data.Im[i], self._freq_data.Ia[i],
                Ef_a, D_Ya, X_Ya, M_Ya,
                self._freq_data.freqs[i]
            )

            Mi[i] = self._Mi(
                self._freq_data.Vm[i], self._freq_data.Va[i],
                self._freq_data.Im[i], self._freq_data.Ia[i],
                Ef_a, D_Ya, X_Ya, M_Ya,
                self._freq_data.freqs[i]
            )

            Pr[i] = self._Pr(
                self._freq_data.Vm[i], self._freq_data.Va[i],
                self._freq_data.Im[i], self._freq_data.Ia[i],
                Ef_a, D_Ya, X_Ya, M_Ya,
                self._freq_data.freqs[i]
            )

            Pi[i] = self._Pi(
                self._freq_data.Vm[i], self._freq_data.Va[i],
                self._freq_data.Im[i], self._freq_data.Ia[i],
                Ef_a, D_Ya, X_Ya, M_Ya,
                self._freq_data.freqs[i]
            )

        # Build and return R (residual vector)
        return np.concatenate([Mr, Mi, Pr, Pi])



@utils.singleton
class CovarianceMatrix:
    """Wrapper for calculations of covariance matrix.

    Attributes:
        _freqs (np.array): frequencies in frequency domain
        _NrNr (function): for computing diagonal elements of gamma_NrNr
        _NrQr (function): for computing diagonal elements of gamma_NrQr
        _NrQi (function): for computing diagonal elements of gamma_NrQi
        _NiNi (function): for computing diagonal elements of gamma_NiNi
        _NiQr (function): for computing diagonal elements of gamma_NiQr
        _NiQi (function): for computing diagonal elements of gamma_NiQi
        _QrQr (function): for computing diagonal elements of gamma_QrQr
        _QiQi (function): for computing diagonal elements of gamma_QiQi

    Note:
        attributes _QrNr, _QrNi, _QiNr, _QiNi are absent.
        It is not necessary to store them due to the following equations:
            gamma_QrNr = gamma_NrQr
            gamma_QrNi = gamma_NiQr
            gamma_QiNr = gamma_NrQi
            gamma_QiNi = gamma_NiQi
        This fact will be used in the 'compute' method.
    """

    def __init__(self, freq_data):
        """Prepares for computing the covariance matrix.

        Args:
            freq_data (class FreqData): data in frequency domain
        """
        self._freqs = freq_data.freqs

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

        self._NrNr = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Im**2
                + std_eps_Vm**2 * (Y11r**2 + Y11i**2)
                + std_eps_Va**2 * (Y12r**2 + Y12i**2)
            ),
            modules='numexpr'
        )

        self._NrQr = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Vm**2 * (Y11r*Y21r + Y11i*Y21i) +
                std_eps_Va**2 * (Y12r*Y22r + Y12i*Y22i)
            ),
            modules='numexpr'
        )

        self._NrQi = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Vm**2 * (Y11r*Y21i - Y11i*Y21r) +
                std_eps_Va**2 * (Y12r*Y22i - Y12i*Y22r)
            ),
            modules='numexpr'
        )

        self._NiNi = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Im**2
                + std_eps_Vm**2 * (Y11r**2 + Y11i**2)
                + std_eps_Va**2 * (Y12r**2 + Y12i**2)
            ),
            modules='numexpr'
        )

        self._NiQr = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Vm**2 * (Y11i*Y21r - Y11r*Y21i) +
                std_eps_Va**2 * (Y12i*Y22r - Y12r*Y22i)
            ),
            modules='numexpr'
        )

        self._NiQi = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Vm**2 * (Y11i*Y21i + Y11r*Y21r) +
                std_eps_Va**2 * (Y12i*Y22i + Y12r*Y22r)
            ),
            modules='numexpr'
        )

        self._QrQr = sympy.lambdify(
            args=[Ef_a, D_Ya, M_Ya, X_Ya, Omega_a],
            expr=(
                std_eps_Ia**2
                + std_eps_Vm**2 * (Y21r**2 + Y21i**2)
                + std_eps_Va**2 * (Y22r**2 + Y22i**2)
            ),
            modules='numexpr'
        )

        self._QiQi = sympy.lambdify(
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


    def compute(self, uncertain_gen_params):
        """Computes the covariance matrix in the given point.

        Builds and computes the numerical value of the covariance matrix
        in the point specified by 'generator_params'. The result matrix
        will have sizes (4K+4) * (4K+4) and contain only numbers (not symbols).

        Args:
            uncertain_gen_params (class UncertainGeneratorParameters):
                current uncertain parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            gamma (numpy.ndarray): value of the covariance matrix
        """
        Ef_a = uncertain_gen_params.Ef_a
        D_Ya = uncertain_gen_params.D_Ya
        X_Ya = uncertain_gen_params.X_Ya
        M_Ya = uncertain_gen_params.M_Ya

        gamma_NrNr = np.diag([
            self._NrNr(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self._freqs
        ])
        gamma_NrQr = np.diag([
            self._NrQr(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self._freqs
        ])
        gamma_NrQi = np.diag([
            self._NrQi(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self._freqs
        ])
        gamma_NiNi = np.diag([
            self._NiNi(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self._freqs
        ])
        gamma_NiQr = np.diag([
            self._NiQr(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self._freqs
        ])
        gamma_NiQi = np.diag([
            self._NiQi(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self._freqs
        ])
        gamma_QrQr = np.diag([
            self._QrQr(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self._freqs
        ])
        gamma_QiQi = np.diag([
            self._QiQi(Ef_a, D_Ya, X_Ya, M_Ya, freq) for freq in self._freqs
        ])
        gamma_QrNr = gamma_NrQr
        gamma_QrNi = gamma_NiQr
        gamma_QiNr = gamma_NrQi
        gamma_QiNi = gamma_NiQi

        zero_matrix = np.zeros((len(self._freqs), len(self._freqs)))
        gamma_L = np.block([
            [gamma_NrNr, zero_matrix, gamma_NrQr, gamma_NrQi],
            [zero_matrix, gamma_NiNi, gamma_NiQr, gamma_NiQi],
            [gamma_QrNr, gamma_QrNi, gamma_QrQr, zero_matrix],
            [gamma_QiNr, gamma_QiNi, zero_matrix, gamma_QiQi]
        ])
        return gamma_L


    def compute_and_inverse(self, uncertain_gen_params):
        """Computes the inversed covariance matrix in the given point.

        Does exactly the same as 'compute' method but after computing
        the covariance matrix this method make the calculated matrix inversed
        and returns it.

        Args:
            uncertain_gen_params (class UncertainGeneratorParameters):
                current parameters of a generator (at the current iteration of
                an optimization routine)

        Returns:
            gamma_L^(-1) (numpy.ndarray): inversed covariance matrix
                in the given point
        """
        return sp.sparse.linalg.inv(sp.sparse.csc_matrix(
            self.compute(uncertain_gen_params)
        )).toarray()



@utils.singleton
class ObjectiveFunction:
    """Wrapper for calculations of objective function.

    Attributes:
        _prior_gen_params (class UncertainGeneratorParameters):
            start point for an optimization routine
        _R (class ResidualVector): auxiliary member to simplify
            calculations of vector R (residual vector)
            at the current step of an optimization routine
        _gamma_L (class CovarianceMatrix): auxiliary member to simplify
            calculations of gamma_L (covariance matrix)
            at the current step of an optimization routine
        _reversed_gamma_g (numpy.ndarray): diagonal matrix containing
            standard deviations of prior uncertain generator parameters
    """

    def __init__(self, freq_data, prior_gen_params):
        """Prepares for computing the objective function in a given point.

        Args:
            freq_data (class FreqData): data in frequency domain
            prior_gen_params (class UncertainGeneratorParameters):
                prior generator parameters of a generator (we are uncertain
                in their values) and the standard variances (how much we are
                uncertain in their values)
        """
        # All these members must not be changed after initialization!
        self._R = ResidualVector(freq_data)
        self._gamma_L = CovarianceMatrix(freq_data)

        self._prior_gen_params = np.array([
            prior_gen_params.Ef_a,
            prior_gen_params.D_Ya,
            prior_gen_params.X_Ya,
            prior_gen_params.M_Ya
        ])

        self._reversed_gamma_g = np.diag([
            prior_gen_params.std_dev_Ef_a,
            prior_gen_params.std_dev_D_Ya,
            prior_gen_params.std_dev_X_Ya,
            prior_gen_params.std_dev_M_Ya
        ])


    def compute(self, uncertain_gen_params):
        """Computes value of the objective function in the given point.
        Args:
            uncertain_gen_params (class UncertainGeneratorParameters):
                current values of uncertain generator parameters
                (at the current iteration of an optimization routine)
        Returns:
            value (numpy.float64) of objective function in the given point
        """
        curr_gen_params = np.array([
            uncertain_gen_params.Ef_a,
            uncertain_gen_params.D_Ya,
            uncertain_gen_params.X_Ya,
            uncertain_gen_params.M_Ya
        ])

        delta_params = curr_gen_params - self._prior_gen_params
        computed_R = self._R.compute(uncertain_gen_params)

        return (
            delta_params @ self._reversed_gamma_g @ delta_params +
            (
                computed_R @
                self._gamma_L.compute_and_inverse(uncertain_gen_params) @
                computed_R
            )
        )


    def compute_by_array(self, uncertain_gen_params):
        """Computes value of the objective function in the given point.

        This method just calls self.compute method
        transforming the sole argument from numpy.array to an instance
        of class UncertainGeneratorParameters. It is necessary
        to have such method because optimizers want to get an instance
        of numpy.array.

        Args:
            uncertain_gen_params (numpy.array):
                current values of uncertain generator parameters
                (at the current iteration of an optimization routine)
        Returns:
            value (numpy.float64) of objective function in the given point
        """
        print('### optimizing... curr_point =', uncertain_gen_params)
        return self.compute(UncertainGeneratorParameters(
            Ef_a=uncertain_gen_params[0],
            D_Ya=uncertain_gen_params[1],
            X_Ya=uncertain_gen_params[2],
            M_Ya=uncertain_gen_params[3],
            std_dev_Ef_a=self._reversed_gamma_g[0][0],
            std_dev_D_Ya=self._reversed_gamma_g[1][1],
            std_dev_X_Ya=self._reversed_gamma_g[2][2],
            std_dev_M_Ya=self._reversed_gamma_g[3][3]
        ))


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



# start_time = time.time()
# R = ResidualVector(freq_data)
# print("constructing R : %s seconds" % (time.time() - start_time))

prior_gen_params = UncertainGeneratorParameters(
    Ef_a=1.0, D_Ya=2.0, X_Ya=3.0, M_Ya=4.0,
    std_dev_Ef_a=0.1, std_dev_D_Ya=0.2, std_dev_X_Ya=0.3, std_dev_M_Ya=0.4
)

# start_time = time.time()
# computed_R = R.compute(prior_gen_params)
# print('len(R) =', len(computed_R))
# print(computed_R)
# print("computing R : %s seconds" % (time.time() - start_time))

# start_time = time.time()
# gamma_L = CovarianceMatrix(freq_data)
# print("constructing gamma_L : %s seconds" % (time.time() - start_time))
#
# start_time = time.time()
# computed_gamma_L = gamma_L.compute_and_inverse(prior_gen_params)
# print("calculating gamma_L : %s seconds" % (time.time() - start_time))

start_time = time.time()
f = ObjectiveFunction(
    freq_data=freq_data,
    prior_gen_params=prior_gen_params
)
print("constructing objective function : %s seconds" % (time.time() - start_time))

# start_time = time.time()
# f0 = f.compute(prior_gen_params)
# print('f0 =', f0, type(f0))
# print("calculating objective function : %s seconds" % (time.time() - start_time))


opt_res = sp.optimize.minimize(
    fun=f.compute_by_array,
    x0=prior_gen_params.get_as_array(),
    method='BFGS',
    # tol=...?
    options={
        # 'maxiter': 1,
        'disp': True
    }
)
