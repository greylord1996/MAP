"""Classes to store data in time and frequency domains.

This module defines two classes: TimeData and FreqData (to store data
in time and frequency domain respectively). Both classes derive from
the abstract class Data. Typically, TimeData holds initial data.
White noise can be applied to these data. After that, the data should
be transformed to frequency domain by DFT (Discrete Fourier Transform).
Finally, additional preliminary operations are also possible (removing
zero frequency, trimming data and removing some frequency bands).

"""

import abc
import numpy as np
import scipy as sp
import scipy.signal


class Data(abc.ABC):
    """Abstract base class to hold data (in time or frequency domain).

    This abstract class contains only odd number of data points.
    If input data contain even number of data points, the last point
    will not be used.

    Attributes:
        inputs (numpy.ndarray): Input data (denoted as u in the paper)
            with shape (n_inputs, n_data_points).
        outputs (numpy.ndarray): Output data (denoted as y in the paper)
            with shape (n_outputs, n_data_points).
    """

    def __init__(self, inputs, outputs):
        """Just save input and output data inside the class.

        Args:
            inputs (numpy.ndarray): Input data (see the 'inputs'
                attribute of the class).
            outputs (numpy.ndarray): Output data (see the 'outputs'
                attribute of the class).
        """
        if not isinstance(inputs, np.ndarray):
            raise TypeError('inputs must be an instance of a numpy.ndarray.')
        if not isinstance(outputs, np.ndarray):
            raise TypeError('outputs must be an instance of a numpy.ndarray.')
        if inputs.shape[1] != outputs.shape[1]:
            raise ValueError('Inconsistent number of data points'
                             'in inputs and outputs.')
        if inputs.shape[1] < 10:
            raise ValueError('Not enough points.')

        self.inputs = inputs
        self.outputs = outputs


class TimeData(Data):
    """Represent data in time domain.

    Attributes:
        dt (float): Time between two adjacent points in time domain.
        input_std_devs (numpy.ndarray): Array with shape (n_inputs,)
            containing measurement noise of input data.
        output_std_devs (numpy.ndarray): Array with shape (n_outputs,)
            containing measurement noise of output data.
    """

    def __init__(self, inputs, outputs, dt,
                 input_std_devs=None, output_std_devs=None):
        """Initialize data in time domain.

        Args:
            inputs (numpy.ndarray): Input data (see the 'inputs'
                attribute of the base class 'Data').
            outputs (numpy.ndarray): Output data (see the 'outputs'
                attribute of the base class 'Data').
            dt (float): time step between data points
            input_std_devs (numpy.ndarray): Measurement noise
                of input data (see the 'input_std_devs' attribute).
            output_std_devs (numpy.ndarray): Measurement noise
                of output data (see the 'output_std_devs' attribute).
        """
        if input_std_devs is not None and len(inputs) != len(input_std_devs):
            raise ValueError('Number of inputs must be equal'
                             'to number of input standard deviations.')
        if output_std_devs is not None and len(outputs) != len(output_std_devs):
            raise ValueError('Number of outputs must be equal'
                             'to number of output standard deviations.')

        super().__init__(inputs, outputs)
        if self.inputs.shape[1] % 2 == 0:
            self.inputs = np.delete(self.inputs, -1, axis=1)
            self.outputs = np.delete(self.outputs, -1, axis=1)

        self.dt = dt
        self.input_std_devs = input_std_devs
        self.output_std_devs = output_std_devs

    def apply_white_noise(self, snr):
        """Apply AWGN (Additive White Gaussian Noise) to storing data.

        Initialize the 'input_std_devs' and 'output_std_devs'
        attributes. Then, it slightly modifies both input and output
        data by applying noise specified by the 'snr' argument.
        If 'input_std_devs' and 'output_std_devs' attributes have been
        already constructed, it is considered as an error.

        Note:
            Be careful with the definition of SNR (Signal to Noise Ratio).
            Should signal-mean be used when capturing the signal power?
            In many cases, we don't care about steady state offset:
            it is useless information. The signal that we care about
            is the perturbation on top of steady state. Let's call that
            our signal. For example, if we have voltage vector
            x = V + dV, and we define our signal as dV, then
            var(dV)/var(noise) is our SNR.

        Args:
            snr (float): desired SNR (Signal to Noise Ratio)
                specified in dB (decibels).
        """
        assert self.inputs.shape[1] == self.outputs.shape[1]
        if snr < 0.0:
            raise ValueError('SNR can not be negative.')
        if self.input_std_devs is not None or self.output_std_devs is not None:
            raise ValueError('Attempt to apply noise to data having initialized'
                             'noise standard deviations. It is incorrect.')

        self.input_std_devs = np.zeros(self.inputs.shape[0])
        self.output_std_devs = np.zeros(self.outputs.shape[0])
        n_time_points = self.inputs.shape[1]

        # applying white noise to inputs
        for input_idx in range(len(self.input_std_devs)):
            self.input_std_devs[input_idx] = np.std(
                self.inputs[input_idx], ddof=1
            ) / (10.0**(snr/20.0))
            self.inputs[input_idx] += np.multiply(
                np.random.normal(loc=0.0, scale=1.0, size=n_time_points),
                self.input_std_devs[input_idx]
            )

        # applying white noise to outputs
        for output_idx in range(len(self.output_std_devs)):
            self.output_std_devs[output_idx] = np.std(
                self.outputs[output_idx], ddof=1
            ) / (10.0**(snr/20.0))
            self.outputs[output_idx] += np.multiply(
                np.random.normal(loc=0.0, scale=1.0, size=n_time_points),
                self.output_std_devs[output_idx]
            )


class FreqData(Data):
    """Represent data in frequency domain.

    Attributes:
        freqs (np.ndarray): frequencies in frequency domain
        input_std_devs (numpy.ndarray): Noise of input data.
        output_std_devs (numpy.ndarray): Noise of output data.
    """

    def __init__(self, time_data, remove_zero_freq=True):
        """Initialize data in frequency domain based on data in time domain.

        It takes (2K + 1) points in time domain (white noise
        has been already applied) and constructs (K + 1) points of data
        in frequency domain (applying Discrete Fourier transform).
        After that, zero frequency can be excluded because in many cases
        of signal processing steady state has no useful information.

        Args:
            time_data (TimeData): Data in time domain.
            remove_zero_freq (bool, optional): whether to remove
                zero frequency from data
        """
        dt = time_data.dt  # time step between signal data points
        fs = 1.0 / dt  # sampling frequency (in Hz)
        n_time_points = time_data.inputs.shape[1]  # N = number of data points
        assert n_time_points % 2 == 1  # Ensure that N is odd (N = 2K + 1)

        # don't forget about the 2*pi factor!
        self.freqs = 2 * np.pi * (fs / n_time_points *
                      np.arange(0, (n_time_points + 1) / 2, 1))

        # perform DFT
        super().__init__(
            inputs=np.array([
                self._apply_dft(time_data.inputs[input_idx])
                for input_idx in range(time_data.inputs.shape[0])
            ]),
            outputs=np.array([
                self._apply_dft(time_data.outputs[output_idx])
                for output_idx in range(time_data.outputs.shape[0])
            ])
        )

        # calculate epsilons (variables representing noise)
        # in frequency domain based on epsilons in time domain
        transform_factor = np.sqrt(2.0 / n_time_points)
        self.input_std_devs = time_data.input_std_devs * transform_factor
        self.output_std_devs = time_data.output_std_devs * transform_factor

        # zero frequency amplitude can be too large
        # compared to amplitudes at other frequencies
        if remove_zero_freq:
            self.freqs = self.freqs[1:]
            self.inputs = np.delete(self.inputs, 0, axis=1)
            self.outputs = np.delete(self.outputs, 0, axis=1)

    def _apply_dft(self, time_points):
        # apply DFT to an array representing one time series
        assert len(time_points.shape) == 1
        n_time_points = len(time_points)
        assert n_time_points % 2 == 1

        window = sp.signal.windows.hann(n_time_points)
        windowed_time_points = np.multiply(
            window,
            scipy.signal.detrend(data=time_points, type='constant')
        )

        freq_points = np.fft.fft(windowed_time_points / n_time_points)
        freq_points_len = (n_time_points + 1) // 2
        freq_points = freq_points[0:freq_points_len]

        # steady component = (1/N) * |F(0)|, other amplitudes = (2/N) * |F(k)|
        freq_points[1:] *= 2.0  # Double all except for DC

        # we have removed steady component by using the 'detrend' function
        freq_points[0] = 0.0

        return freq_points

    def trim(self, min_freq, max_freq):
        """Remove all data which are not located at [min_freq; max_freq].

        Note:
            This method implies that frequencies in self.freqs
            are sorted in ascending order.

        Args:
            min_freq (float): minimum remaining frequency in the data
            max_freq (float): maximum remaining frequency in the data
        """
        assert len(self.freqs) == self.inputs.shape[1] == self.outputs.shape[1]
        if min_freq < 0.0:
            raise ValueError('min_freq can not be negative.')
        if min_freq > max_freq:
            raise ValueError('min_freq must be less than max_freq.')

        begin = (np.searchsorted(self.freqs, min_freq, side='left')
                 if min_freq is not None else 0)
        end = (np.searchsorted(self.freqs, max_freq, side='right')
               if max_freq is not None else len(self.freqs))

        self.freqs = self.freqs[begin:end]
        self.inputs = np.delete(
            self.inputs,
            list(range(begin)) + list(range(end, self.inputs.shape[1])),
            axis=1
        )
        self.outputs = np.delete(
            self.outputs,
            list(range(begin)) + list(range(end, self.outputs.shape[1])),
            axis=1
        )

    def remove_band(self, min_freq, max_freq):
        """Remove all data which are located at [min_freq; max_freq].

        Note:
            This method implies that frequencies in self.freqs
            are sorted in ascending order.

        Args:
            min_freq (float): begin of frequency band
            max_freq (float): end of frequency band
        """
        begin = np.searchsorted(self.freqs, min_freq, side='left')
        end = np.searchsorted(self.freqs, max_freq, side='right')
        removing_indexes = np.arange(begin, end)

        self.freqs = np.delete(self.freqs, removing_indexes)
        self.inputs = np.delete(self.inputs, removing_indexes, axis=1)
        self.outputs = np.delete(self.outputs, removing_indexes, axis=1)

