
# Integration steps
class IntegrationSettings:

    def __init__(self, df_length, dt_step):
        self.df_length = df_length
        self.dt_step = dt_step

    def set_values_from_dict(self, new_values):
        self.df_length = new_values['df_length']
        self.dt_step = new_values['df_step']

    def get_values_as_dict(self):
        return {
            'df_length': self.df_length,
            'dt_step': self.dt_step
        }



# Frequency settings
class FreqData:

    def __init__(self, lower_fb, upper_fb, max_freq):
        self.lower_fb = lower_fb  # Set lower frequency (band edge) for current injections
        self.upper_fb = upper_fb  # Set lower frequency (band edge) for current injections
        self.max_freq = max_freq  # Set max frequency

    def set_values_from_dict(self, new_values):
        self.lower_fb = new_values['lower_fb']
        self.upper_fb = new_values['upper_fb']
        self.max_freq = new_values['max_freq']

    def get_values_as_dict(self):
        return {
            'lower_fb': self.lower_fb,
            'upper_fb': self.upper_fb,
            'max_freq': self.max_freq
        }



# Optimizer Settings
class OptimizerSettings:

    def __init__(self, opt_tol, fun_tol, stp_tol, max_its, sol_mtd, opt_its, opt_mcp):
        self.opt_tol = opt_tol
        self.fun_tol = fun_tol
        self.stp_tol = stp_tol
        self.max_its = max_its
        self.sol_mtd = sol_mtd
        self.opt_its = opt_its
        self.opt_mcp = opt_mcp

    def set_values_from_dict(self, new_values):
        self.opt_tol = new_values['opt_tol']
        self.fun_tol = new_values['fun_tol']
        self.stp_tol = new_values['stp_tol']
        self.max_its = new_values['max_its']
        self.sol_mtd = new_values['sol_mtd']
        self.opt_its = new_values['opt_its']
        self.opt_mcp = new_values['opt_mcp']

    def get_values_as_dict(self):
        return {
            'opt_tol': self.opt_tol,
            'fun_tol': self.fun_tol,
            'stp_tol': self.stp_tol,
            'max_its': self.max_its,
            'sol_mtd': self.sol_mtd,
            'opt_its': self.opt_its,
            'opt_mcp': self.opt_mcp
        }



# Generator Parameters
class GeneratorParameters:

    def __init__(self, d_2, e_2, m_2, x_d2, ic_d2):
        self.d_2 = d_2  # X_d^'  transient reactance
        self.e_2 = e_2
        self.m_2 = m_2  # M
        self.x_d2 = x_d2
        self.ic_d2 = ic_d2  # rotor angle

    def set_values_from_dict(self, new_values):
        self.d_2 = new_values['d_2']
        self.e_2 = new_values['e_2']
        self.m_2 = new_values['m_2']
        self.x_d2 = new_values['x_d2']
        self.ic_d2 = new_values['ic_d2']

    def get_values_as_dict(self):
        return {
            'd_2': self.d_2,
            'e_2': self.e_2,
            'm_2': self.m_2,
            'x_d2': self.x_d2,
            'ic_d2': self.ic_d2
        }



# Oscillation Parameters
class OscillationParameters:

    def __init__(self, osc_amp, osc_freq):
        self.osc_amp = osc_amp
        self.osc_freq = osc_freq

    def set_values_from_dict(self, new_values):
        self.osc_amp = new_values['osc_amp']
        self.osc_freq = new_values['osc_freq']

    def get_values_as_dict(self):
        return {
            'osc_amp': self.osc_amp,
            'osc_freq': self.osc_freq
        }



# Noise Settings
class Noise:

    def __init__(self, rnd_amp, snr):
        self.rnd_amp = rnd_amp
        self.snr = snr

    def set_values_from_dict(self, new_values):
        self.rnd_amp = new_values['rnd_amp']
        self.snr = new_values['snr']

    def get_values_as_dict(self):
        return {
            'rnd_amp': self.rnd_amp,
            'snr': self.snr
        }



# Inf Bus Initialization
class InfBusInitializer:

    def __init__(self, ic_v1, ic_t1):
        self.ic_v1 = ic_v1
        self.ic_t1 = ic_t1

    def set_values_from_dict(self, new_values):
        self.ic_v1 = new_values['ic_v1']
        self.ic_t1 = new_values['ic_t1']

    def get_values_as_dict(self):
        return {
            'ic_v1': self.ic_v1,
            'ic_t1': self.ic_t1
        }



# class PriorData:
#     pass



# All settings
class Settings:
    """Wrapper to hold all settings."""

    def __init__(self, all_settings_as_dict):
        self.integration_settings = None
        self.freq_data = None
        self.optimizer_settings = None
        self.generator_parameters = None
        self.oscillation_parameters = None
        self.noise = None
        self.inf_bus_initializer = None
        self.set_values_from_dict(all_settings_as_dict)


    def get_values_as_dict(self):
        return {
            'IntegrationSettings': (
                self.integration_settings.get_values_as_dict()
            ),
            'FreqData': (
                self.freq_data.get_values_as_dict()
            ),
            'OptimizerSettings': (
                self.optimizer_settings.get_values_as_dict()
            ),
            'GeneratorParameters': (
                self.generator_parameters.get_values_as_dict()
            ),
            'OscillationParameters': (
                self.oscillation_parameters.get_values_as_dict()
            ),
            'Noise': (
                self.noise.get_values_as_dict()
            ),
            'InfBusInitializer': (
                self.inf_bus_initializer.get_values_as_dict()
            )
        }


    def set_values_from_dict(self, all_settings_as_dict):
        self.integration_settings = IntegrationSettings(
            **all_settings_as_dict['IntegrationSettings']
        )
        self.freq_data = FreqData(
            **all_settings_as_dict['FreqData']
        )
        self.optimizer_settings = OptimizerSettings(
            **all_settings_as_dict['OptimizerSettings']
        )
        self.generator_parameters = GeneratorParameters(
            **all_settings_as_dict['GeneratorParameters']
        )
        self.oscillation_parameters = OscillationParameters(
            **all_settings_as_dict['OscillationParameters']
        )
        self.noise = Noise(
            **all_settings_as_dict['Noise']
        )
        self.inf_bus_initializer = InfBusInitializer(
            **all_settings_as_dict['InfBusInitializer']
        )

