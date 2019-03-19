import numpy as np


class Dft:

    def __init__(self, signal, tstep, N):
        self.signal = signal
        self.tstep = tstep
        self.point_number = N - 1 if N % 2 == 0 else N
        self.y_fft = None
        self.fs = 1 / self.tstep
        self.f_vec = self.fs / self.point_number * np.arange(0, (self.point_number - 1) / 2, 1)

    def map_dft(self):

        self.y_fft = np.fft.fft(self.signal/self.point_number, self.point_number)
        self.y_fft = self.y_fft[0:int((self.point_number+1)/2)]
        self.y_fft[1:] = 2.0 * self.y_fft[1:]


