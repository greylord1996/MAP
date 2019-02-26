import sympy as sp
from sympy import *
import numpy as np

j = np.complex(0, 1)

del_a, w_a, Pm_a, Omega, Vm_a, Va_a = symbols('del_a w_a Pm_a Omega Vm_a Va_a')
Ef_a = symbols('Ef_a', real=True)
D_Ya = symbols('D_Ya', real=True)
X_Ya = symbols('X_Ya', real=True)
M_Ya = symbols('M_Ya', real=True)


# X = symbols('X')
X = Array([del_a, w_a])
Uv = Array([Vm_a, Va_a])
X_params = Array([D_Ya, Ef_a, M_Ya, X_Ya])


# System DAE Model
Pe = Vm_a*Ef_a*sin(del_a-Va_a)/X_Ya

F_vec = Matrix([w_a, (Pm_a - Pe - D_Ya*w_a)/M_Ya])

# Output => [abs(I); angle(I)], where I flows into the generator
G_vec = Matrix([(1/X_Ya)*sqrt(Vm_a ** 2 + Ef_a ** 2 - 2*Ef_a*Vm_a*cos(Va_a-del_a)),
               atan((Vm_a*sin(Va_a)-Ef_a*sin(del_a))/(Vm_a*cos(Va_a)-Ef_a*cos(del_a)))-pi])

Omega_a = symbols('Omega_a', real=True)
JacFX = F_vec.jacobian(X)
JacFUv = F_vec.jacobian(Uv)
JacGX = G_vec.jacobian(X)
JacGUv = G_vec.jacobian(Uv)


Ys = (JacGX/(j*Omega_a*eye(len(X)) - JacFX))*JacFUv + JacGUv
Ys = Ys.subs(Va_a, 0.5)
Ys = Ys.subs(del_a, 1)
Ys = Ys.subs(Vm_a, 1)
#Ys = simplify(Ys)

Ys_compute = lambdify((Ef_a, D_Ya, M_Ya, X_Ya, Omega_a), Ys, 'numpy')

#Ys_compute = lambdify((w_a, Ef_a, D_Ya, M_Ya, X_Ya, Pm_a, Omega, Omega_a, Va_a, Vm_a, del_a), Ys, 'numpy')

Y11 = Ys[0, 0]
Y12 = Ys[0, 1]
Y21 = Ys[1, 0]
Y22 = Ys[1, 1]

Y11r = re(Y11)
Y11i = im(Y11)
Y12r = re(Y12)
Y12i = im(Y12)
Y21r = re(Y21)
Y21i = im(Y21)
Y22r = re(Y22)
Y22i = im(Y22)

Vmrs, Vmis, Vars, Vais, Imrs, Imis, Iars, Iais = symbols('Vmrs Vmis Vars Vais Imrs Imis Iars Iais')
Inj_mrs, Inj_mis, Inj_prs, Inj_pis = symbols('Inj_mrs Inj_mis Inj_prs Inj_pis')


Mr = (Imrs - Y11r * Vmrs + Y11i * Vmis - Y12r * Vars + Y12i * Vais)
Mr = Matrix([Mr])
Mi = Imis - Y11i * Vmrs - Y11r * Vmis - Y12i * Vars - Y12r * Vais
Pr = Iars - Y21r * Vmrs + Y21i * Vmis - Y22r * Vars + Y22i * Vais
Pi = Iais - Y21i * Vmrs - Y21r * Vmis - Y22i * Vars - Y22r * Vais


Mrtemp = Matrix([Imrs - Y11r * Vmrs + Y11i * Vmis - Y12r * Vars + Y12i * Vais])
Mitemp = Matrix([Imis - Y11i * Vmrs - Y11r * Vmis - Y12i * Vars - Y12r * Vais])
Prtemp = Matrix([Iars - Y21r * Vmrs + Y21i * Vmis - Y22r * Vars + Y22i * Vais])
Pitemp = Matrix([Iais - Y21i * Vmrs - Y21r * Vmis - Y22i * Vars - Y22r * Vais])

Grad_Mr = Mrtemp.jacobian(X_params)
Grad_Mi = Mitemp.jacobian(X_params)
Grad_Pr = Prtemp.jacobian(X_params)
Grad_Pi = Pitemp.jacobian(X_params)


Mr_compute = lambdify((D_Ya, Ef_a, M_Ya, X_Ya, Omega_a, Imrs, Vmrs, Vmis, Vars, Vais), Mr)
Mi_compute = lambdify((Imis, Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Mi, 'numpy')
Pr_compute = lambdify((Iars, Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Pr, 'numpy')
Pi_compute = lambdify((Iais, Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Pi, 'numpy')

Grad_Mr_compute = lambdify((Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Grad_Mr, 'numpy')
Grad_Mi_compute = lambdify((Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Grad_Mi, 'numpy')
Grad_Pr_compute = lambdify((Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Grad_Pr, 'numpy')
Grad_Pi_compute = lambdify((Vmrs, Vmis, Vars, Vais, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a), Grad_Pi, 'numpy')
