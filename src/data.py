import abc
import numpy as np

import utils



class Data(abc.ABC):
    """Abstract class to hold the data.

    Attributes:
        Vm (np.array): voltage magnitudes depending on time or frequency
        Va (np.array): voltage phases depending on time or frequency
        Im (np.array): current magnitudes depending on time or frequency
        Ia (np.array): current phases depending on time or frequency
    """

    def __init__(self, Vm_data, Va_data, Im_data, Ia_data):
        self.Vm = Vm_data
        self.Va = Va_data
        self.Im = Im_data
        self.Ia = Ia_data



@utils.singleton
class TimeData(Data):
    """Represents data in the time domain."""

    def __init__(self, Vm_time_data, Va_time_data, Im_time_data, Ia_time_data):
        """"""
        super().__init__(
            Vm_data=Vm_time_data,
            Va_data=Va_time_data,
            Im_data=Im_time_data,
            Ia_data=Ia_time_data
        )


    def apply_white_noise(self, snr, d_coi):
        """Applies Additive white Gaussian noise (AWGN) to storing data.

        Args:
            snr (float): desired Signal to Noise Ratio (SNR) in dB (decibels)
            d_coi (float): Center of inertia to subtract from angular data.
                If there is no COI, just set it to 0.0.
        """
        assert(len(self.Vm) == len(self.Im))
        assert(len(self.Im) == len(self.Va))
        assert(len(self.Va) == len(self.Ia))
        pure_time_data_len = len(self.Vm)

        self.std_dev_Vm = np.std(self.Vm) / (10.0**(snr/20.0))
        self.std_dev_Im = np.std(self.Im) / (10.0**(snr/20.0))
        self.std_dev_Va = np.std(self.Va - d_coi) / (10.0**(snr/20.0))
        self.std_dev_Ia = np.std(self.Ia - d_coi) / (10.0**(snr/20.0))

        # Magnitude Data
        self.Vm = self.Vm + np.multiply(
            np.random.normal(loc=0.0, scale=1.0, size=pure_time_data_len),
            self.std_dev_Vm
        )
        self.Im = self.Im + np.multiply(
            np.random.normal(loc=0.0, scale=1.0, size=pure_time_data_len),
            self.std_dev_Im
        )

        # Angle Data: Subtract out COI (might be 0.0)
        self.Va = self.Va + np.multiply(
            np.random.normal(loc=0.0, scale=1.0, size=pure_time_data_len),
            self.std_dev_Va
        )
        self.Ia = self.Ia + np.multiply(
            np.random.normal(loc=0.0, scale=1.0, size=pure_time_data_len),
            self.std_dev_Ia
        )



@utils.singleton
class FreqData(Data):
    """Represents data in the frequency domain."""

    def __init__(self, Vm_freq_data, Va_freq_data, Im_freq_data, Ia_freq_data):
        """"""
        super().__init__(
            Vm_data=Vm_freq_data,
            Va_data=Va_freq_data,
            Im_data=Im_freq_data,
            Ia_data=Ia_freq_data
        )


    def have_no_idea_how_to_name_this_method(self):
        # See MAP_MeasNoise.m
        # std_w.Vm = STD_ns.Vm * sqrt(2 / N);
        # std_w.Va = STD_ns.Va * sqrt(2 / N);
        # std_w.Im = STD_ns.Im * sqrt(2 / N);
        # std_w.Ia = STD_ns.Ia * sqrt(2 / N);
        pass



# if __name__ == '__main__':
#     Vm_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
#     Im_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
#     Va_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
#     Ia_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
#     time_data = TimeData(
#         Vm_data, Im_data, Va_data, Ia_data
#     )
#     time_data.apply_white_noise(2.0)
#     print('Vm_data =', time_data.Vm)
#     print('Im_data =', time_data.Im)
#     print('Va_data =', time_data.Va)
#     print('Ia_data =', time_data.Ia)

