import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import *
import matplotlib.pyplot as plt

i = 0
j = np.complex(0, 1)


class OdeSolver:

    def __init__(self, noise, osc_param, integr_param):
        self.noise = noise
        self.osc_param = osc_param
        self.IC_T1 = 0.0
        self.IC_V1 = 1.0
        self.dt = integr_param['dt_step']
        self.tf = integr_param['df_length']
        self.test_length = np.arange(0, self.tf + self.dt, self.dt)
        self.t_vec = np.linspace(0, self.tf, self.test_length.size)
        self.V1t = self.get_V1t()
        self.T1t = self.get_T1t()
        self.T1t_to_simulate = None
        self.Ig = None
        self.Vc1 = None
        self.Ig_abs = None
        self.Ig_angle = None
        self.J = 2 * 0.5 * 0.5 / (2 * np.pi * 50) ** 2
        self.w_0 = 1
        self.tau = 0.02
        self.teta = 1
        self.wm = None
        self.R = 0.08
        self.z = None
        self.X = 0.2
        self.Pm = 1.0
        self.we_0 = 2 * np.pi * 50
        self.H = 0.5
        self.sigma0 = 0.04
        self.V0 = 1

    def get_system(self, t, x):
        w_m = x[0]
        z = x[1]
        A = (self.we_0 ** 2) * (self.we_0 + z / self.tau + self.T1t(t) / self.tau - w_m) * self.R * (self.V1t(t) ** 2)
        B = 2 * self.H * (self.we_0 + z / self.tau + self.T1t(t) / self.tau) ** 2
        D = w_m / (self.we_0 + z / self.tau + self.T1t(t) / self.tau)
        C = self.R ** 2 + (1 - D) ** 2 * self.X ** 2

        dwmdt = A / (B * C) - self.Pm * self.we_0 / (self.H * 2)
        dzdt = 0 - z / self.tau - self.T1t(t) / self.tau

        return [dwmdt, dzdt]

    def solve(self):
        k = self.tau * 2 * np.pi * 50 - self.tau * 2 * np.pi * 50 * 0.999 - 0.001
        y0 = [2 * np.pi * 50 * 0.96, k]
        # start_time = time.time()
        # print("--- %s seconds ---" % (time.time() - start_time))
        self.sol = solve_ivp(fun=self.get_system, t_span=[0, self.tf], y0=y0, t_eval=self.t_vec)
        # print("--- %s seconds ---" % (time.time() - start_time))
        self.w_m = self.sol.y[0, :]
        self.z = self.sol.y[1, :]
        return {'wm': self.w_m, 'z': self.z}

    def get_V1t(self):
        vn_vec = np.random.normal(0, 1, self.test_length.size) * self.noise['rnd_amp'] + self.IC_V1
        V1t = interp1d(self.t_vec, vn_vec, kind='cubic', fill_value="extrapolate")
        return V1t

    def show_V1t_in_test_mode(self):
        plt.plot(self.t_vec, self.V1t(self.t_vec))
        plt.legend(['V1t'])
        plt.show()

    def get_T1t(self):
        tn_vec = np.random.normal(0, 1, self.test_length.size) * self.noise['rnd_amp'] + self.IC_T1
        T1t = interp1d(self.t_vec, tn_vec, kind='cubic', fill_value="extrapolate")
        return T1t

    def get_t_vec(self):
        return self.t_vec

    def show_T1t_in_test_mode(self):
        plt.plot(self.t_vec, self.T1t(self.t_vec))
        plt.legend(['T1t'])
        plt.show()

    def show_results_in_test_mode(self):
        # plt.figure()
        plt.plot(self.t_vec, self.w_m)
        plt.legend(['w_m(t)'])
        plt.show()
        plt.plot(self.t_vec, self.z)
        plt.legend(['z(t)'])
        plt.show()
        plt.plot(self.t_vec, self.T1t(self.t_vec))
        plt.legend(['teta(t)'])
        plt.show()
        plt.plot(self.t_vec, self.pe)
        plt.legend(['pe(t)'])
        plt.show()
        plt.plot(self.t_vec, self.we)
        plt.legend(['we(t)'])
        plt.show()
        plt.plot(self.t_vec, self.sigma)
        plt.legend(['sigma(t)'])
        plt.show()
        plt.plot(self.t_vec, self.id)
        plt.legend(['id(t)'])
        plt.show()
        plt.plot(self.t_vec, self.iq)
        plt.legend(['iq(t)'])
        plt.show()
        # plt.savefig('data.pdf', format='pdf')

        # plt.plot(self.t_vec, self.Ig)
        # plt.legend(['Ig(t)'])
        # plt.show()
        # plt.plot(self.t_vec, self.Vc1)
        # plt.legend(['Vc1(t)'])
        # plt.show()

    def simulate(self):
        self.solve()
        self.we = self.we_0 + self.z / self.tau + self.T1t(self.t_vec) / self.tau
        self.sigma = 1 - self.w_m / self.we
        self.pe = (self.sigma * self.R * self.V1t(self.t_vec) ** 2) / (self.R ** 2 + self.sigma ** 2 * self.X ** 2)
        self.I_0 = 1 / ((self.R / self.sigma0) + j * self.X)
        self.qe0 = 0 - self.I_0.imag
        self.I = self.V1t(self.t_vec) / (self.R / self.sigma + j * self.X)
        self.id = self.I.real
        self.iq = self.I.imag
        self.vt = self.V1t(self.t_vec)
        self.tt = self.T1t(self.t_vec)

        self.T1t_to_simulate = self.T1t.y + self.osc_param['osc_amp'] * np.sin(2 * np.pi * self.osc_param['osc_freq'] * self.test_length)
        self.Vc1 = self.V1t.y * np.exp(j * self.T1t_to_simulate)
        self.vc1_real = self.Vc1.real
        self.vc1_image = self.Vc1.image

