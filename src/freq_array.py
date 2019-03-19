import numpy as np


class FreqArray:

    def __init__(self, freq_start, freq_step, freq_count):
        self.freq_start = freq_start
        self.freq_step = freq_step
        self.freq_count = freq_count
        self.data = np.linspace(start=self.freq_start,  stop=self.freq_start+(self.freq_step*self.freq_count),
                                num=self.freq_count)

