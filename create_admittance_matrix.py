import sympy as sp
from sympy import *
import numpy as np

j = np.complex(0, 1)

del_a, w_a, Ef_a, D_Ya, M_Ya, X_Ya, Pm_a, Omega, Vm_a, Va_a = symbols('del_a w_a Ef_a D_Ya M_Ya X_Ya Pm_a Omega Vm_a Va_a')
X = symbols('X')
X = Array([del_a, w_a])
Uv = Array([Vm_a, Va_a])

# System DAE Model
Pe = Vm_a*Ef_a*sin(del_a-Va_a)/X_Ya

F_vec = Matrix([w_a, (Pm_a - Pe - D_Ya*w_a)/M_Ya])

# Output => [abs(I); angle(I)], where I flows into the generator
G_vec = Matrix([(1/X_Ya)*sqrt(Vm_a ** 2 + Ef_a ** 2 - 2*Ef_a*Vm_a*cos(Va_a-del_a)),
               atan((Vm_a*sin(Va_a)-Ef_a*sin(del_a))/(Vm_a*cos(Va_a)-Ef_a*cos(del_a)))-pi])

Omega_a = symbols('Omega_a')

JacFX = F_vec.jacobian(X)
JacFUv = F_vec.jacobian(Uv)
JacGX = G_vec.jacobian(X)
JacGUv = G_vec.jacobian(Uv)


Ys = (JacGX/(j*Omega_a*eye(len(X)) - JacFX))*JacFUv + JacGUv
#Ys = simplify(Ys)

Ys_compute = lambdify((w_a, Ef_a, D_Ya, M_Ya, X_Ya, Pm_a, Omega, Omega_a, Va_a, Vm_a, del_a), Ys, 'numpy')