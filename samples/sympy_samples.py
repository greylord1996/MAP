import sympy as sp
from sympy import *

a, b, c = symbols('a b c')

c = a + b

A = Matrix([[a, b], [a, b]])
ev_A = lambdify((a, b), A, "numpy")
print("(0, 1): ", ev_A(0, 0))

B = ev_A(1, 1)
print(B)