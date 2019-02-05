"""
1. Initialize Solver Constants
"""

import numpy as np

"""
Complex Unit
"""

j = np.complex(0, 1)

"""
Frequency settings
"""


class FreqData:

    def __init__(self, lower_fb=1.99, upper_fb=2.01, max_freq=6, dt=0.005):
        self.lower_fb = lower_fb  # Set lower frequency (band edge) for current injections
        self.upper_fb = upper_fb  # Set lower frequency (band edge) for current injections
        self.max_freq = max_freq  # Set max frequency
        self.dt = dt

    def get_list_of_values(self):
        return [self.lower_fb, self.upper_fb, self.max_freq, self.dt]

    def get_values(self):
        return {'lower_fb': self.lower_fb, 'upper_fb': self.upper_fb,
                'max_freq': self.max_freq, 'dt': self.dt}


"""
Optimizer Settings
"""


class OptimizerSettings:

    def __init__(self, opt_tol=1e-6, fun_tol=1e-6, stp_tol=1e-6, max_its=1, sol_mtd=3, opt_its=50, opt_mcp=0.01):
        self.opt_tol = opt_tol
        self.fun_tol = fun_tol
        self.stp_tol = stp_tol
        self.max_its = max_its
        self.sol_mtd = sol_mtd
        self.opt_its = opt_its
        self.opt_mcp = opt_mcp

    def get_list_of_values(self):
        return [self.opt_tol, self.fun_tol, self.stp_tol, self.max_its, self.sol_mtd, self.opt_its, self.opt_mcp]

    def get_values(self):
        return {'opt_tol': self.opt_tol, 'fun_tol': self.fun_tol, 'stp_tol': self.stp_tol,
                'max_its': self.max_its, 'sol_mtd': self.sol_mtd, 'opt_its': self.opt_its, 'opt_mcp': self.opt_mcp}


"""
Generator Parameters
"""


class GeneratorParameters:

    def __init__(self, d_2=0.25, e_2=1, m_2=1, x_d2=0.01, ic_d2=1):
        self.d_2 = d_2
        self.e_2 = e_2
        self.m_2 = m_2
        self.x_d2 = x_d2
        self.ic_d2 = ic_d2

    def get_list_of_values(self):
        return [self.d_2, self.e_2, self.m_2, self.x_d2]

    def get_values(self):
        return {'d_2': self.d_2, 'e_2': self.e_2, 'm_2': self.m_2, 'x_d2': self.x_d2, 'ic_d2': self.ic_d2}


"""
Oscillation Parameters
"""


class OscillationParameters:

    def __init__(self, osc_amp=2, osc_freq=0.005):
        self.osc_amp = osc_amp
        self.osc_freq = osc_freq

    def get_list_of_values(self):
        return [self.osc_amp, self.osc_freq]

    def get_values(self):
        return {'osc_amp': self.osc_amp, 'osc_freq': self.osc_freq}


"""
White Noise Settings
"""


class WhiteNoise:

    def __init__(self, rnd_amp=0.02):
        self.rnd_amp = rnd_amp

    def get_list_of_values(self):
        return [self.rnd_amp]

    def get_values(self):
        return {'rnd_amp': self.rnd_amp}


"""
Inf Bus Initialization
"""


class InfBusInitializer:

    def __init__(self, ic_v1=1, ic_t1=0.5):
        self.ic_v1 = ic_v1
        self.ic_t1 = ic_t1

    def get_list_of_values(self):
        return [self.ic_v1, self.ic_t1]

    def get_values(self):
        return {'ic_v1': self.ic_v1, 'ic_t1': self.ic_t1}


class PriorData:
    pass
