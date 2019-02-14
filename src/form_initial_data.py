import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import *
import matplotlib.pyplot as plt
from settings import GeneratorParameters, OscillationParameters, WhiteNoise

i = 0
j = np.complex(0, 1)


class OdeSolver():

    #def __init__(self, rnd_amp, d_2, e_2, m_2, x_d2, ic_d2, osc_amp, osc_freq):
    def __init__(self, white_noise, gen_param, osc_param, integr_param):
        self.white_noise = WhiteNoise(white_noise['rnd_amp'])
        self.generator_param = GeneratorParameters(gen_param['d_2'], gen_param['e_2'],
                                                   gen_param['m_2'], gen_param['x_d2'],
                                                   gen_param['ic_d2'])
        self.osc_param = OscillationParameters(osc_param['osc_amp'], osc_param['osc_freq'])

        # self.white_noise = WhiteNoise(rnd_amp)
        # self.generator_param = GeneratorParameters(d_2, e_2,
        #                                            m_2, x_d2,
        #                                            ic_d2)
        # self.osc_param = OscillationParameters(osc_amp, osc_freq)

        self.IC_T1 = 0.5
        self.IC_V1 = 1.0
        self.IC_d2 = 1.0
        self.dt = integr_param['dt_step']
        self.tf = integr_param['df_length']
        self.test_length = np.arange(0, self.tf, self.dt)
        self.t_vec = np.linspace(0, self.tf, self.test_length.size)
        self.Pm2_0 = self.calculate_Pm2_0()
        self.V1t = self.get_V1t()
        self.T1t = self.get_T1t()
        self.w2 = None
        self.d2 = None

    def calculate_Pm2_0(self):
        V1c = self.IC_V1 * np.exp(j * self.IC_T1)  # Bus Voltage (complex)
        V2c = self.generator_param.e_2 * np.exp(j * self.IC_d2)  # Generator Voltage (complex)
        Pm2_0 = np.real(V2c * np.conj((V2c - V1c) / (j * self.generator_param.x_d2)))
        return Pm2_0

    def get_V1t(self):
        vn_vec = np.random.normal(0, 1, self.test_length.size) * self.white_noise.rnd_amp + self.IC_V1
        V1t = interp1d(self.t_vec, vn_vec, kind='cubic', fill_value="extrapolate")
        return V1t

    def show_V1t_in_test_mode(self):
        plt.plot(self.t_vec, self.V1t(self.t_vec))
        plt.legend(['V1t'])
        plt.show()

    def get_T1t(self):
        tn_vec = np.random.normal(0, 1, self.test_length.size) * self.white_noise.rnd_amp + self.IC_T1
        T1t = interp1d(self.t_vec, tn_vec, kind='cubic', fill_value="extrapolate")
        return T1t

    def get_t_vec(self):
        return self.t_vec

    def show_T1t_in_test_mode(self):
        plt.plot(self.t_vec, self.T1t(self.t_vec))
        plt.legend(['T1t'])
        plt.show()

    def get_system(self, x, t):
        w2 = x[0]
        d2 = x[1]

        #V1c = self.V1t(t) * np.exp(j * self.T1t(t)) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        V1c = 1.0 * np.exp(j * 1.0)
        V2c = self.generator_param.e_2 * np.exp(j * d2)

        Pe2 = np.real(V2c * np.conj((V2c - V1c) / (j * self.generator_param.x_d2)))
        Pm2t = self.Pm2_0 + self.osc_param.osc_amp * np.sin(2 * np.pi * self.osc_param.osc_freq * t)

        # Define the system
        dw2dt = Pm2t - Pe2 - self.generator_param.d_2 * w2 / self.generator_param.m_2
        d2dt = w2

        return [dw2dt, d2dt]

    def solve(self):
        y0 = [0, 1.0]
        sol = odeint(self.get_system, y0, self.t_vec)
        self.w2 = sol[:, 0]
        self.d2 = sol[:, 1]
        return {'w2': self.w2, 'd2': self.d2}

    def get_appropr_data_to_gui(self):
        return {'t_vec': self.t_vec, 'w2': self.w2, 'd2': self.d2,
                'V1t': self.V1t(self.t_vec), 'T1t': self.T1t(self.t_vec)}

    def show_results_in_test_mode(self):
        plt.plot(self.t_vec, self.w2)
        plt.legend(['w2(t)'])
        plt.show()
        plt.plot(self.t_vec, self.d2)
        plt.legend(['d2(t)'])
        plt.show()

#solver = OdeSolver(0.02, 0.25, 1.0, 1.0, 0.01, 1, 2, 0.005)
#solver.solve()