import os
import os.path
import pickle
import sympy

import utils



class AdmittanceMatrix:
    """A singletone representing the admittance matrix.

    Normally, it is not necessary to recompute this matrix
    every time when you create an instance of this class.
    But if you want to force recomputation from the scratch
    you should provide one argument for constructor:
    >>> Ys = AdmittanceMatrix(is_actual=False).Ys

    Attributes:
        Ys: admittance matrix
    """

    def __init__(self, is_actual=False):  # False now just for debugging
        path_to_this_file = os.path.abspath(os.path.dirname(__file__))
        path_to_matrix_file = os.path.join(
            path_to_this_file,
            '..', 'data', 'precomputed', 'admittance_matrix.pickle'
        )
        if not is_actual:
            # j = np.complex(0, 1)
            j = sympy.I
            Ef_a, D_Ya, X_Ya, M_Ya, Omega_a = sympy.symbols(
                'Ef_a D_Ya X_Ya M_Ya Omega_a',
                real=True
            )

            # X_params -- Vector of Uncertain Generator Paramerers
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
                    (Vm_a*sympy.sin(Va_a) - Ef_a*sympy.sin(del_a))
                    / (Vm_a*sympy.cos(Va_a) - Ef_a*sympy.cos(del_a))
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

#
# Ef_a, D_Ya, X_Ya, M_Ya, Omega_a = sympy.symbols(
#     'Ef_a D_Ya X_Ya M_Ya Omega_a',
#     real=True
# )
# # X_params = Array([D_Ya, Ef_a, M_Ya, X_Ya])
#
# Ys_compute = lambdify([Ef_a, D_Ya, M_Ya, X_Ya, Omega_a], Ys, 'numexpr')
#
# Y11 = Ys[0, 0]
# Y12 = Ys[0, 1]
# Y21 = Ys[1, 0]
# Y22 = Ys[1, 1]
#
# Y11r = re(Y11)
# Y11i = im(Y11)
# Y12r = re(Y12)
# Y12i = im(Y12)
# Y21r = re(Y21)
# Y21i = im(Y21)
# Y22r = re(Y22)
# Y22i = im(Y22)
#
# Vmrs, Vmis, Vars, Vais, Imrs, Imis, Iars, Iais = symbols('Vmrs Vmis Vars Vais Imrs Imis Iars Iais')
# Inj_mrs, Inj_mis, Inj_prs, Inj_pis = symbols('Inj_mrs Inj_mis Inj_prs Inj_pis')
#
#
# Mr = Imrs - Y11r * Vmrs + Y11i * Vmis - Y12r * Vars + Y12i * Vais
# Mi = Imis - Y11i * Vmrs - Y11r * Vmis - Y12i * Vars - Y12r * Vais
# Pr = Iars - Y21r * Vmrs + Y21i * Vmis - Y22r * Vars + Y22i * Vais
# Pi = Iais - Y21i * Vmrs - Y21r * Vmis - Y22i * Vars - Y22r * Vais
#
#
# Mrtemp = Matrix([Imrs - Y11r * Vmrs + Y11i * Vmis - Y12r * Vars + Y12i * Vais])
# Mitemp = Matrix([Imis - Y11i * Vmrs - Y11r * Vmis - Y12i * Vars - Y12r * Vais])
# Prtemp = Matrix([Iars - Y21r * Vmrs + Y21i * Vmis - Y22r * Vars + Y22i * Vais])
# Pitemp = Matrix([Iais - Y21i * Vmrs - Y21r * Vmis - Y22i * Vars - Y22r * Vais])
#
# Grad_Mr = Mrtemp.jacobian(X_params)
# Grad_Mi = Mitemp.jacobian(X_params)
# Grad_Pr = Prtemp.jacobian(X_params)
# Grad_Pi = Pitemp.jacobian(X_params)
#
#
# Mr_compute = lambdify((D_Ya, Ef_a, M_Ya, X_Ya, Omega_a, Imrs, Vmrs, Vmis, Vars, Vais), Mr, 'numpy')
# Mi_compute = lambdify((Imis, Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Mi, 'numpy')
# Pr_compute = lambdify((Iars, Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Pr, 'numpy')
# Pi_compute = lambdify((Iais, Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Pi, 'numpy')
#
# Grad_Mr_compute = lambdify((Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Grad_Mr, 'numpy')
# Grad_Mi_compute = lambdify((Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Grad_Mi, 'numpy')
# Grad_Pr_compute = lambdify((Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Grad_Pr, 'numpy')
# Grad_Pi_compute = lambdify((Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Grad_Pi, 'numpy')
#
# print("hello")
