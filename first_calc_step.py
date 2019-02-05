from settings import j, GeneratorParameters
import sympy as sp
from sympy import *
import numpy as np


w2, d2, V1, T1, Pm2 = symbols('w2(t) d2(t) V1(t) T1(t) Pm2(t)')
#print(w2, d2, V1, Pm2)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DAE variables!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ODEvars = np.array([w2, d2])


gp = GeneratorParameters()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Define Complex Voltages!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
V1c = V1 * sp.exp(j*T1)
#print('V1c = ', V1c)
V2c = gp.e_2 * sp.exp(j*d2)
#print('V2c = ', V2c)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Electrical Power!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Pe2 = sp.re(V2c * np.conj((V2c - V1c)/(j*gp.x_d2)))
#print('Pe2 = ', Pe2)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ODEs!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ODEs = sp.array([diff(w2)==(Pm2 - Pe2 - gp.d_2*w2)/gp.m_2, diff(d2)==w2])
ODEs = [diff(w2) == (Pm2 - Pe2 - gp.d_2*w2)/gp.m_2]
print('ODEs = ', ODEs)

