import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.integrate import *
import matplotlib.pyplot as plt
from settings import GeneratorParameters, OscillationParameters, WhiteNoise

j = np.complex(0, 1)
gp = GeneratorParameters(d_2=0.25, e_2=1, m_2=1, x_d2=0.01, ic_d2=1)
osc_p = OscillationParameters(osc_amp=2, osc_freq=0.001)

E2 = 1
Xd2 = 0.01
M2 = 1
D2 = 0.25

"""
Initial calculations
"""


"""
In this section we define the system of ODE with variables [w2(t); d2(t)]
"""
IC_T1 = 0.5
IC_V1 = 1
IC_d2 = 1
V1c = IC_V1 * np.exp(j * IC_T1)  # Bus Voltage (complex)
V2c = gp.e_2 * np.exp(j * IC_d2)  # Generator Voltage (complex)
Pm2_0 = np.real(V2c * np.conj((V2c - V1c) / (j * gp.x_d2)))
# print("Pm2_0 = ", Pm2_0)

"""
Define Power Functions
In this section we add normal noise to constant mode to fluctuate
constant and interpolate by cubic this discrete values
"""
# White Noise
Rnd_Amp = 0.02

# Simulation Length
dt = 0.005
tf = 100
test_length = np.arange(0, tf, dt)

vn_vec = np.random.normal(0, 1, test_length.size) * Rnd_Amp + IC_V1
tn_vec = np.random.normal(0, 1, test_length.size) * Rnd_Amp + IC_T1

t_vec = np.linspace(0, tf, test_length.size)
V1t = interp1d(t_vec, vn_vec, kind='cubic', fill_value="extrapolate")
T1t = interp1d(t_vec, tn_vec, kind='cubic', fill_value="extrapolate")

def get_system(x, t):

    w2 = x[0]
    d2 = x[1]



    # plt.plot(t_vec, V1t(t_vec))
    # plt.legend(['V1(t)'])
    # plt.show()

    #plt.plot(t_vec, T1t(t_vec))
    #plt.legend(['T1(t)'])
    #plt.show()
    V1c = V1t(t) * np.exp(j * T1t(t))
    V2c = gp.e_2 * np.exp(j * d2)
    Pe2 = np.real(V2c * np.conj((V2c - V1c)/(j * Xd2)))

    Pm2t = Pm2_0 + osc_p.osc_amp * np.sin(2 * np.pi * osc_p.osc_freq * t)

    dw2dt = Pm2t - Pe2 - gp.d_2 * w2 / gp.m_2
    d2dt = w2

    return [dw2dt, d2dt]


"""
Solve ODE
"""
def solver():
    y0 = [0, 1]
    dt = 0.005
    tf = 100
    test_length = np.arange(0, tf, dt)
    t = np.linspace(0, tf, test_length.size)
    t_vec = np.linspace(0, tf, test_length.size)
    sol = odeint(get_system, y0, t_vec)
    #sol = solve_ivp(get_system, [0, tf], y0)

    w2 = sol[:, 0]
    d2 = sol[:, 1]

    plt.plot(t, w2)
    plt.legend(['w2(t)'])
    plt.show()
    plt.plot(t, d2)
    plt.legend(['d2(t)'])
    plt.show()

    return {'t': t, 'w2': w2}

solver()

# def test(d):
#     white_noise = WhiteNoise(rnd_amp=d['rnd_amp'])
#     IC_T1 = 0.5
#     IC_V1 = 1
#
#     Rnd_Amp = 0.02
#
#     # Simulation Length
#     dt = 0.005
#     tf = 100
#     test_length = np.arange(0, tf, dt)
#
#     vn_vec = np.random.normal(0, 1, test_length.size) * white_noise.rnd_amp + IC_V1
#     tn_vec = np.random.normal(0, 1, test_length.size) * white_noise.rnd_amp + IC_T1
#     print("!!!!!!!!!!!!: ", white_noise.rnd_amp)
#
#     t_vec = np.linspace(0, tf, test_length.size)
#     V1t = interp1d(t_vec, vn_vec, kind='cubic', fill_value="extrapolate")
#     T1t = interp1d(t_vec, tn_vec, kind='cubic', fill_value="extrapolate")
#
#     # plt.plot(t_vec, V1t(t_vec))
#     # plt.legend(['V1(t)'])
#     # plt.show()
#
#     return {'t_vec': t_vec, 'V1t': V1t(t_vec)}
