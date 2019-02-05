# theta''(t) + b*theta'(t) + c*sin(theta(t)) = 0
# theta'(t) = omega(t)
# omega'(t) = -b*omega(t) - c*sin(theta(t))

import scipy as sp
from scipy import *
import numpy as np


def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt


b = 0.01
c = 0.00005

y0 = [np.pi - 0.1, 0.0]
t = np.linspace(0, 10, 101)
print(t)

from scipy.integrate import odeint
sol = odeint(pend, y0, t, args=(b, c))

import matplotlib.pyplot as plt
plt.plot(t, sol[:, 0], 'b', label='theta(t)')
plt.plot(t, sol[:, 1], 'r', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
