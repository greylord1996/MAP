import abc
import copy
import numpy as np
import scipy
import scipy.signal

import dynamic_equations_to_simulate
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
        assert len(Vm_data) == len(Im_data)
        assert len(Im_data) == len(Va_data)
        assert len(Va_data) == len(Ia_data)
        self.Vm = Vm_data
        self.Va = Va_data
        self.Im = Im_data
        self.Ia = Ia_data

        if len(self.Vm) % 2 == 0:
            self.Vm = self.Vm[:-1]
            self.Va = self.Va[:-1]
            self.Im = self.Im[:-1]
            self.Ia = self.Ia[:-1]



class TimeData(Data):
    """Represents data in time domain.

    Attributes:
        dt (float): time between two adjacent points in time domain
        std_dev_Vm (float): standard deviation of voltage magnitude
        std_dev_Im (float): standard deviation of voltage phase
        std_dev_Va (float): standard deviation of current magnitude
        std_dev_Ia (float): standard deviation of current phase
    """

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

        self.std_dev_Vm = None
        self.std_dev_Va = None
        self.std_dev_Im = None
        self.std_dev_Ia = None


    def apply_white_noise(self, snr, d_coi):
        """Applies Additive white Gaussian noise (AWGN) to storing data.

        Args:
            snr (float): desired Signal to Noise Ratio (SNR) in dB (decibels)
            d_coi (float): Center of inertia to subtract from angular data.
                If there is no COI, just set it to 0.0.
        """
        assert len(self.Vm) == len(self.Va)
        assert len(self.Va) == len(self.Im)
        assert len(self.Im) == len(self.Ia)
        pure_time_data_len = len(self.Vm)

        self.std_dev_Vm = np.std(self.Vm, ddof=1) / (10.0**(snr/20.0))
        self.std_dev_Im = np.std(self.Im, ddof=1) / (10.0**(snr/20.0))
        self.std_dev_Va = np.std(self.Va - d_coi, ddof=1) / (10.0**(snr/20.0))
        self.std_dev_Ia = np.std(self.Ia - d_coi, ddof=1) / (10.0**(snr/20.0))

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



class FreqData(Data):
    """Represents data in frequency domain.

    Attributes:
        freqs (np.array): frequencies in frequency domain
        std_w_Vm (float): standard deviation of voltage magnitude
        std_w_Va (float): standard deviation of voltage phase
        std_w_Im (float): standard deviation of current magnitude
        std_w_Ia (float): standard deviation of current phase
    """

    def __init__(self, time_data):
        """Initializes data in frequency domain based on data in time domain.

        It takes (2K + 1) points in time domain (white noise
        has been already applied) and constructs (K + 1) points of data
        in frequency domain (applying Discrete Fourier transform).

        Args:
            time_data (TimeData): holding data in time domain
        """
        dt = time_data.dt  # time step between signal data points
        fs = 1.0 / dt  # sampling frequency
        time_points_len = len(time_data.Vm)  # N = number of data points
        assert time_points_len % 2 == 1  # Ensure that N is odd (N = 2K + 1)

        # f_vec is the same as self.freqs (don't forget about the 2*pi factor!)
        self.freqs = (fs / time_points_len *
                      np.arange(0, (time_points_len + 1) / 2, 1))

        super().__init__(
            Vm_data=self._apply_dft(time_data.Vm),
            Va_data=self._apply_dft(time_data.Va),
            Im_data=self._apply_dft(time_data.Im),
            Ia_data=self._apply_dft(time_data.Ia)
        )

        self.std_w_Vm = time_data.std_dev_Vm * np.sqrt(2.0 / time_points_len)
        self.std_w_Va = time_data.std_dev_Va * np.sqrt(2.0 / time_points_len)
        self.std_w_Im = time_data.std_dev_Im * np.sqrt(2.0 / time_points_len)
        self.std_w_Ia = time_data.std_dev_Ia * np.sqrt(2.0 / time_points_len)


    def _apply_dft(self, time_points):
        # Apply FFT to each array of data (Vm, Va, Im, Ia)
        time_points_len = len(time_points)
        assert time_points_len % 2 == 1

        window = scipy.signal.windows.hann(time_points_len)
        windowed_time_points = np.multiply(
            window,
            scipy.signal.detrend(data=time_points, type='constant')
        )

        freq_points = np.fft.fft(windowed_time_points / time_points_len)
        freq_points_len = (time_points_len + 1) // 2
        freq_points = freq_points[0:freq_points_len]

        # Amplitude of DC = (1/N) * |F(0)|, other amplitudes = (2/N) * |F(k)|
        freq_points[1:] *= 2.0  # Double all but DC

        # We have removed DC by using detrend function
        freq_points[0] = 0.0  # Zero DC

        return freq_points


    def remove_zero_frequency(self):
        """Removes the first point from each array of data (Vm, Va, Im, Ia).

        After applying DFT the first numbers of each array (Vm, Va, Im, Ia)
        are equal to 0 (due to applying detrend function).
        It can be convenient to remove these zeros, that is to exclude
        the first number from each array. Zero frequency also should be
        excluded from the array of frequencies (self.freqs).
        """
        self.freqs = self.freqs[1:]
        self.Vm = self.Vm[1:]
        self.Va = self.Va[1:]
        self.Im = self.Im[1:]
        self.Ia = self.Ia[1:]


    def trim(self, min_freq, max_freq):
        """Removes all data which are not located in [min_freq; max_freq].

        Leaves only those data which are located in [min_freq; max_freq].
        This method implies that frequencies in self.freqs are sorted
        in ascending order.

        Args:
            min_freq (float): minimum remaining frequency in the data
            max_freq (float): maximum remaining frequency in the data
        """
        begin = np.searchsorted(self.freqs, min_freq, side='left')
        end = np.searchsorted(self.freqs, max_freq, side='right')

        self.freqs = self.freqs[begin:end]
        self.Vm = self.Vm[begin:end]
        self.Va = self.Va[begin:end]
        self.Im = self.Im[begin:end]
        self.Ia = self.Ia[begin:end]


    def remove_data_from_fo_band(self, min_fo_freq, max_fo_freq):
        """Removes data which are located in a forced oscillation band.

        Removes the range [min_fo_freq; max_fo_freq] of frequencies
        where the forced oscillation has significant effect. Moreover,
        it removes corresponding data from frequency data (Vm, Va, Im, Ia).

        Args:
            min_fo_freq (float): begin forced oscillation band
            max_fo_freq (float): end forced oscillation band
        """
        begin = np.searchsorted(self.freqs, min_fo_freq, side='left')
        end = np.searchsorted(self.freqs, max_fo_freq, side='right')
        removing_indexes = np.arange(begin, end)

        self.freqs = np.delete(self.freqs, removing_indexes)
        self.Vm = np.delete(self.Vm, removing_indexes)
        self.Va = np.delete(self.Va, removing_indexes)
        self.Im = np.delete(self.Im, removing_indexes)
        self.Ia = np.delete(self.Ia, removing_indexes)



@utils.singleton
class DataHolder:
    """Wrapper for storing different prepared data from a generator.

    Attributes:
        _initial_time_data (class TimeData): initial data in time domain
        _time_data_after_snr (class TimeData): data in time domain
            after applying white noise (specified by SNR)
        _freq_data_after_fft (class FreqData): data after FFT
        _freq_data_after_trim (class FreqData): data (in frequency domain)
            after removing zero frequency and trimming data
            (for stage2 of the optimization routine)
        _freq_data_after_remove_fo_band (class FreqData): data
            (in frequency domain) after excluding data from FO band
            (for stage1 of the optimization routine)

    Note:
        All attributes are private. Don't use them outside this class.
        Communicate with an instance of this class only via
        its public methods.
    """

    def __init__(self, initial_params):
        """Prepares different data for the optimization routine.

        Args:
            initial_params (class Settings): all configuration parameters
                (should be obtained from GUI)
        """
        # Simulate data in time domain (initial data)
        ode_solver_object = dynamic_equations_to_simulate.OdeSolver(
            noise=initial_params.noise,
            gen_param=initial_params.generator_parameters,
            osc_param=initial_params.oscillation_parameters,
            integr_param=initial_params.integration_settings
        )
        ode_solver_object.simulate_time_data()
        self._initial_time_data = TimeData(
            Vm_time_data=ode_solver_object.Vc1_abs,
            Va_time_data=ode_solver_object.Vc1_angle,
            Im_time_data=ode_solver_object.Ig_abs,
            Ia_time_data=ode_solver_object.Ig_angle,
            dt=ode_solver_object.dt
        )

        # Apply white noise to simulated data in time domain
        self._time_data_after_snr = copy.copy(self._initial_time_data)
        self._time_data_after_snr.apply_white_noise(
            snr=initial_params.noise.snr,
            d_coi=0.0
        )

        # Moving from time domain to frequency domain
        self._freq_data_after_fft = FreqData(self._time_data_after_snr)

        # Trim data
        self._freq_data_after_trim = copy.copy(self._freq_data_after_fft)
        self._freq_data_after_trim.remove_zero_frequency()
        self._freq_data_after_trim.trim(
            min_freq=0.0,
            max_freq=initial_params.freq_data.max_freq
        )

        # Remove data from forced oscillation band
        # (it is necessary only for running stage1, not stage2)
        self._freq_data_after_remove_fo_band = (
            copy.copy(self._freq_data_after_trim)
        )
        self._freq_data_after_remove_fo_band.remove_data_from_fo_band(
            min_fo_freq=initial_params.freq_data.lower_fb,
            max_fo_freq=initial_params.freq_data.upper_fb
        )


    def get_data(self, stage):
        """Returns prepared data for stage1 or stage2.

        If you need data (in frequency domain) for stage1
        of the optimization routine, shallow copy of
        self._freq_data_after_remove_fo_band will be returned.
        Data for stage2 are the same except for
        it has data in the forced oscillation band (FO band).

        Args:
            stage (int): stage number (1 or 2)
                1 -- clarify generator parameters
                2 -- find the source of forced oscillations

        Returns:
            freq_data (class FreqData): data in frequency domain
                which has been prepared for stage1 or stage2
                of the optimization routine
        """
        if stage not in (1, 2):
            raise ValueError('You should specify the stage (1 or 2) '
                             'for the objective function.')
        freq_data = None
        if stage == 1:
            freq_data = copy.copy(self._freq_data_after_remove_fo_band)
        if stage == 2:
            freq_data = copy.copy(self._freq_data_after_trim)
        return freq_data

