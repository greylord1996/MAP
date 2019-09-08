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


import sympy


class AdmittanceMatrix:
    """Admittance matrix of a generator.

    Attributes:
        _params (list): Contains sympy.symbols which are parameters of
            a dynamical system (that is described by an admittance matrix).
        _omega (sympy.symbols): a sympy variable representing frequency.
        _admittance_matrix (sympy.Matrix): True admittance matrix
            of a system (this class is just a wrapper around it).
    """

    def __init__(self):
        """Define relations between input and output data."""
        D_Ya, Ef_a, M_Ya, X_Ya, omega = sympy.symbols(
            'D_Ya Ef_a M_Ya X_Ya omega', real=True
        )
        self._params = [D_Ya, Ef_a, M_Ya, X_Ya]
        self._omega = omega

        del_a, w_a, Pm_a, Vm_a, Va_a = sympy.symbols('del_a w_a Pm_a Vm_a Va_a')
        X = sympy.Array([del_a, w_a])
        Uv = sympy.Array([Vm_a, Va_a])

        Pe = Vm_a * Ef_a * sympy.sin(del_a - Va_a) / X_Ya  # System DAE Model
        F_vec = sympy.Matrix([w_a, (Pm_a - Pe - D_Ya * w_a) / M_Ya])

        # Output => [abs(I); angle(I)], where I flows into the generator
        G_vec = sympy.Matrix([
            (1 / X_Ya) * sympy.sqrt(
                Vm_a**2 + Ef_a**2 - 2 * Ef_a * Vm_a * sympy.cos(Va_a - del_a)
            ),
            sympy.atan(
                (Vm_a*sympy.sin(Va_a) - Ef_a*sympy.sin(del_a)) /
                (Vm_a*sympy.cos(Va_a) - Ef_a*sympy.cos(del_a))
            ) - sympy.pi
        ])

        JacFX = F_vec.jacobian(X)
        JacFUv = F_vec.jacobian(Uv)
        JacGX = G_vec.jacobian(X)
        JacGUv = G_vec.jacobian(Uv)

        j = sympy.I
        Ys = (JacGX / (j*omega*sympy.eye(len(X)) - JacFX)) * JacFUv + JacGUv
        Ys = Ys.subs(Va_a, 0.5)
        Ys = Ys.subs(del_a, 1)
        Ys = Ys.subs(Vm_a, 1)
        self._admittance_matrix = Ys

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

