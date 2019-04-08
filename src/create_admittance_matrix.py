import os
import os.path
import pickle
import sympy



class AdmittanceMatrix:
    """Represents the admittance matrix.

    Normally, it is not necessary to recompute this matrix
    every time when you create an instance of this class.
    But if you want to force recomputation from the scratch
    you should provide one argument for constructor:
    >>> admittance_matrix = AdmittanceMatrix(is_actual=False).Ys

    Attributes:
        Ys: admittance matrix
    """

    def __init__(self, is_actual=True):  # may be False just for debugging
        path_to_this_file = os.path.abspath(os.path.dirname(__file__))
        path_to_matrix_file = os.path.join(
            path_to_this_file,
            '..', 'data', 'precomputed', 'admittance_matrix.pickle'
        )
        if not is_actual:
            # j = np.complex(0, 1)
            j = sympy.I
            D_Ya, Ef_a, M_Ya, X_Ya, Omega_a = sympy.symbols(
                'D_Ya Ef_a M_Ya X_Ya Omega_a',
                real=True
            )

            # X_params -- Vector of Uncertain Generator Parameters
            # X_params = Array([D_Ya, Ef_a, M_Ya, X_Ya])

            del_a, w_a, Pm_a, Vm_a, Va_a = sympy.symbols(
                'del_a w_a Pm_a Vm_a Va_a'
            )

            X = sympy.Array([del_a, w_a])
            Uv = sympy.Array([Vm_a, Va_a])

            # System DAE Model
            Pe = Vm_a * Ef_a * sympy.sin(del_a - Va_a) / X_Ya

            F_vec = sympy.Matrix([w_a, (Pm_a - Pe - D_Ya*w_a)/M_Ya])

            # Output => [abs(I); angle(I)], where I flows into the generator
            G_vec = sympy.Matrix([
                (1/X_Ya) * sympy.sqrt(
                    Vm_a**2 + Ef_a**2 - 2*Ef_a*Vm_a*sympy.cos(Va_a - del_a)
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

            Ys = (JacGX/(j*Omega_a*sympy.eye(len(X))-JacFX)) * JacFUv + JacGUv
            Ys = Ys.subs(Va_a, 0.5)
            Ys = Ys.subs(del_a, 1)
            Ys = Ys.subs(Vm_a, 1)

            self.Ys = Ys
            pickle.dump(self.Ys, open(path_to_matrix_file, 'wb'))

        else:
            self.Ys = pickle.load(open(path_to_matrix_file, 'rb'))



# Ys = AdmittanceMatrix().Ys
