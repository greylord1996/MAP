import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d


RND_Amp = 1
dt = 0.005
tf = 100

vec = np.random.normal(0, 0.02, 20000) * RND_Amp + 1
t = np.linspace(0, tf, 20000)

f = interp1d(t, vec, kind='cubic')
print(f)
plt.plot(t, f(t))
plt.show()

y = np.sin(f(t))
print(y)

"""
vec_new = interpolate.splev(t, tck, der=0)

plt.figure()
plt.plot(t, vec, 'x', t, vec_new, t, vec, t, vec, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.title('Cubic-spline interpolation')
plt.show()



print(vec)

"""