"""Admittance matrix of a system which relates input and output data.
An admittance matrix (denoted as Y) relates input and output data of an
abstract dynamical system in frequency domain. These relations should be
defined inside a class and be available as 'data' property. Due to some
implementation details, parameters of the describing system and
frequency variable must be accessible via 'params' and 'omega'
properties respectively.

Such module should be created by a user of the framework. That is,
a wrapper around admittance matrix of a system should be defined here.

"""

import numpy as np
import sympy


class AdmittanceMatrix:
    """Admittance matrix of a motor.

    Attributes:
        _params (list): Contains sympy.symbols which are parameters of
            a dynamical system (that is described by an admittance matrix).
        _omega (sympy.symbols): a sympy variable representing frequency.
        _admittance_matrix (sympy.Matrix): True admittance matrix
            of a system (this class is just a wrapper around it).
    """

    def __init__(self):
        """Define relations between input and output data."""
        R, X, H, omega = sympy.symbols(
            'R X H omega', real=True
        )
        j = sympy.I
        s = j * omega
        self._params = [R, X, H]  # do not change the order
        self._omega = omega

        omega_e_0, p_e_0, q_e_0, sigma_0 = sympy.symbols('omega_e_0 p_e_0 q_e_0 sigma_0')
        beta = 2 * H * omega_e_0 * s + (omega_e_0 * (R ** 2 - sigma_0 ** 2 * X ** 2) * p_e_0) / (
                    sigma_0 * (R ** 2 + sigma_0 ** 2 * X ** 2))

        Y_dd = p_e_0 * (1 - (2 * omega_e_0 * R * (R ** 2 - sigma_0 ** 2 * X ** 2)) / (
                    beta * ((R ** 2 + sigma_0 ** 2 * X ** 2) ** 2)))

        Y_dq = q_e_0 + ((R * (R ** 2 - sigma_0 ** 2 * X ** 2)) / ((R ** 2 + sigma_0 ** 2 * X ** 2) ** 2)) * (
                    (2 * H * (1 - sigma_0) * s ** 2 + p_e_0 * s) / (beta))

        Y_qd = -q_e_0 + ((2 * sigma_0 * R ** 2 * X) / ((R ** 2 + sigma_0 ** 2 * X ** 2) ** 2)) * (
                    (2 * omega_e_0 * p_e_0) / (beta))
        Y_qq = p_e_0 - ((2 * sigma_0 * R ** 2 * X) / ((R ** 2 + sigma_0 ** 2 * X ** 2) ** 2)) * (
                    (2 * H * (1 - sigma_0) * s ** 2 + p_e_0 * s) / (beta))

        matr = sympy.Matrix([[Y_dd, Y_dq], [Y_qd, Y_qq]])

        # matr = matr.subs(omega_e_0, 2 * np.pi * 50)
        matr = matr.subs(omega_e_0, 50.0)

        matr = matr.subs(p_e_0, 0.49504950495049505)
        matr = matr.subs(sigma_0, 0.04)
        matr = matr.subs(q_e_0, 0.04950495049504951)
        self._admittance_matrix = matr

    @property
    def params(self):
        """sympy.symbols which are parameters of a dynamical system."""
        return self._params

    @property
    def omega(self):
        """sympy.symbol which represents symbol of frequency."""
        return self._omega

    @property
    def data(self):
        """Raw sympy.Matrix that is true admittance matrix of a system."""
        return self._admittance_matrix
