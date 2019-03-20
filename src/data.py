import abc
import numpy as np
import scipy
import scipy.signal

import utils



class Data(abc.ABC):
    """Abstract class to hold the data.

    This class contains only odd number of data points.
    If input data contain even number of data points,
    the last point will not be used.

    Attributes:
        Vm (np.array): voltage magnitudes depending on time or frequency
        Va (np.array): voltage phases depending on time or frequency
        Im (np.array): current magnitudes depending on time or frequency
        Ia (np.array): current phases depending on time or frequency
    """

    def __init__(self, Vm_data, Va_data, Im_data, Ia_data):
        assert(len(Vm_data) == len(Im_data))
        assert(len(Im_data) == len(Va_data))
        assert(len(Va_data) == len(Ia_data))
        self.Vm = Vm_data
        self.Va = Va_data
        self.Im = Im_data
        self.Ia = Ia_data

        if len(self.Vm) % 2 == 0:
            self.Vm = self.Vm[:-1]
            self.Va = self.Va[:-1]
            self.Im = self.Im[:-1]
            self.Ia = self.Ia[:-1]



@utils.singleton
class TimeData(Data):
    """Represents data in the time domain."""

    def __init__(self, Vm_time_data, Va_time_data,
                 Im_time_data, Ia_time_data, dt):
        """Inits data in time domain.

        Args:
            Vm_time_data (np.array): time domain voltage magnitudes
            Va_time_data (np.array): time domain voltage phases
            Im_time_data (np.array): time domain current magnitudes
            Ia_time_data (np.array): time domain current phases
            dt (float): time step between signal data points
        """
        super().__init__(
            Vm_data=Vm_time_data,
            Va_data=Va_time_data,
            Im_data=Im_time_data,
            Ia_data=Ia_time_data
        )
        self.dt = dt


    def apply_white_noise(self, snr, d_coi):
        """Applies Additive white Gaussian noise (AWGN) to storing data.

        Args:
            snr (float): desired Signal to Noise Ratio (SNR) in dB (decibels)
            d_coi (float): Center of inertia to subtract from angular data.
                If there is no COI, just set it to 0.0.
        """
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

    def __init__(self, time_data):
        """Inits data in frequency domain based on data in time domain.

        It takes (2K + 1) points in time domain
        and constructs (K + 1) points of data in frequency domain
        (via Discrete Fourier transform).

        Args:
            time_data (TimeData): holding data in time domain
        """
        dt = time_data.dt  # time step between signal data points
        fs = 1.0 / dt  # sampling frequency
        time_points_len = len(time_data.Vm)  # N = number of data points
        assert(time_points_len % 2 == 1)  # Ensure that N is odd (N = 2K + 1)

        # f_vec is the same as self.freqs (what about pi?)
        self.freqs = (2.0 * fs / time_points_len
                      * np.arange(0, (time_points_len + 1) / 2, 1))

        super().__init__(
            Vm_data=self._apply_dft(time_data.Vm, dt),
            Va_data=self._apply_dft(time_data.Va, dt),
            Im_data=self._apply_dft(time_data.Im, dt),
            Ia_data=self._apply_dft(time_data.Ia, dt)
        )

        self.std_w_Vm = time_data.std_dev_Vm * np.sqrt(2.0 / time_points_len)
        self.std_w_Va = time_data.std_dev_Va * np.sqrt(2.0 / time_points_len)
        self.std_w_Im = time_data.std_dev_Im * np.sqrt(2.0 / time_points_len)
        self.std_w_Ia = time_data.std_dev_Ia * np.sqrt(2.0 / time_points_len)


    def _apply_dft(self, time_points, dt):
        time_points_len = len(time_points)
        assert(time_points_len % 2 == 1)

        freq_points_len = (time_points_len + 1) // 2
        freq_points = np.fft.fft(
            scipy.signal.detrend(time_points) / time_points_len
        )
        freq_points = freq_points[0:freq_points_len]
        # freq_points[0] = 0.0  # equivalent to subtracting mean value before FFT?
        return freq_points

