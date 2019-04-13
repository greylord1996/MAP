import sys
import os
import os.path

sys.path.append(os.path.join(
    os.path.abspath(os.path.dirname(__file__)), '..', '..', 'src')
)

import settings



FREQ_DATA = settings.FreqData(
    lower_fb=1.99,
    upper_fb=2.01,
    max_freq=6.00,
    dt=0.05
)

WHITE_NOISE = settings.WhiteNoise(
    rnd_amp=0.000
)

GENERATOR_PARAMS = settings.GeneratorParameters(
    d_2=0.25,
    e_2=1.0,
    m_2=1.0,
    x_d2=0.01,
    ic_d2=1.0
)

INTEGRATION_SETTINGS = settings.IntegrationSettings(
    dt_step=0.05,
    df_length=100.0
)

OSCILLATION_PARAMS = settings.OscillationParameters(
    osc_amp=2.00,
    osc_freq=0.005
)



SNR = 45.0
D_COI = 0.0
