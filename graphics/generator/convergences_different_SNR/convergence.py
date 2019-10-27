import os
import os.path

import numpy as np
import matplotlib.pyplot as plt


# snrs2.txt
SNRS = 1.0 * np.arange(1, 21, 1)


PRIORS = np.array([
    [0.2387, 1.4752, 1.4201, 0.0123],  # SNR = 1
    [0.3401, 1.3165, 1.2877, 0.0071],
    [0.3243, 1.1617, 1.3325, 0.0147],
    [0.3447, 0.8116, 1.2369, 0.0139],
    [0.1398, 1.2376, 0.5615, 0.0101],  # SNR = 5
    [0.1298, 1.2373, 0.5826, 0.0088],
    [0.2113, 0.6338, 1.0067, 0.0085],
    [0.2257, 0.5196, 1.4982, 0.0100],
    [0.1286, 0.9354, 0.9682, 0.0133],
    [0.3515, 1.4611, 1.2190, 0.0070],  # SNR = 10
    [0.2754, 0.6340, 0.6584, 0.0065],
    [0.1902, 0.5213, 1.3326, 0.0101],
    [0.3002, 1.4218, 0.8734, 0.0103],
    [0.1877, 1.1119, 1.4730, 0.0139],
    [0.3107, 0.7675, 1.1172, 0.0075],  # SNR = 15
    [0.2280, 0.8151, 1.1685, 0.0119],
    [0.1680, 1.2581, 0.5495, 0.0113],
    [0.1747, 0.6639, 1.2770, 0.0112],
    [0.3432, 0.7924, 0.8948, 0.0103],
    [0.3488, 0.9901, 0.9447, 0.0051],  # SNR = 20
])


INTERIOR_POINT_POSTERIORS = np.array([
    [0.1442, 0.9549, 0.5267, 0.0183],  # SNR = 1
    [0.2522, 0.9698, 0.7252, 0.0133],
    [0.2106, 0.9716, 0.7358, 0.0135],
    [0.2339, 0.9740, 0.7899, 0.0124],
    [0.4461, 0.9228, 0.0520, 0.0372],  # SNR = 5
    [0.5766, 0.9332, 0.0573, 0.0359],
    [0.2797, 0.9892, 0.9024, 0.0110],
    [0.2579, 0.9897, 0.8796, 0.0112],
    [0.2540, 0.9928, 0.9030, 0.0110],
    [0.2539, 0.9948, 0.9347, 0.0107],  # SNR = 10
    [0.2468, 0.9910, 0.8944, 0.0111],
    [0.3152, 1.0018, 0.9902, 0.0102],
    [0.2534, 0.9985, 0.9834, 0.0102],
    [0.2762, 0.9977, 0.9747, 0.0102],
    [0.2716, 0.9989, 0.9873, 0.0101],  # SNR = 15
    [0.2760, 0.9973, 0.9680, 0.0103],
    [0.3183, 0.9996, 1.0002, 0.0100],
    [0.2879, 0.9988, 1.0101, 0.0099],
    [0.2665, 0.9973, 0.9744, 0.0102],
    [0.2746, 0.9979, 0.9829, 0.0102],  # SNR = 20
])


CROSS_ENTROPY_POSTERIORS = np.array([
    [0.17416188, 1.00379023, 1.02721645, 0.00972579],
    [0.32766158, 1.00783447, 1.08077997, 0.00920762],
    [0.29162322, 1.00121491, 1.0509551 , 0.00940959],
    [0.3541871 , 1.00897057, 1.14489418, 0.00884248],
    [0.17689166, 1.0001574 , 1.02200354, 0.00975974],
    [0.21960271, 1.00078041, 0.99274872, 0.00998317],
    [0.32551754, 1.01634441, 1.03188503, 0.00974359],
    [0.29463252, 1.01372883, 1.13308804, 0.00895801],
    [0.28421389, 1.00743014, 1.0664785 , 0.00940163],
    [0.31878418, 1.00427787, 1.08697383, 0.00917171],
    [0.23972328, 1.00229691, 1.00889343, 0.00989795],
    [0.24691047, 1.00027796, 1.02037939, 0.00985702],
    [0.27041441, 1.00841567, 1.06053852, 0.00945376],
    [0.32915983, 1.02038341, 1.23526537, 0.00827123],
    [0.26064275, 1.00054564, 1.02406197, 0.00978537],
    [0.27980254, 1.00788728, 1.08179302, 0.00929357],
    [0.24571398, 0.99946314, 0.98591753, 0.01015071],
    [0.24683129, 0.99996458, 1.00087566, 0.00997596],
    [0.29728097, 1.00888688, 1.09204126, 0.00922346],
    [0.23507493, 0.99725534, 0.99528115, 0.01002435]
])


TRUES = np.array([0.25, 1.00, 1.00, 0.01])
PARAMS_NAMES = ("D", "E^{'}", "M", r"X_{\! d}^{'}")

n_points = len(SNRS)
n_params = len(TRUES)
YLIMS = np.array([[0.0, 2.0 * TRUES[i]] for i in range(n_params)])


assert len(SNRS) == len(PRIORS) == len(INTERIOR_POINT_POSTERIORS) == len(CROSS_ENTROPY_POSTERIORS)
assert len(TRUES) == len(PARAMS_NAMES)
assert (PRIORS.shape[1] == INTERIOR_POINT_POSTERIORS.shape[1] == CROSS_ENTROPY_POSTERIORS.shape[1]
        == len(TRUES) == len(PARAMS_NAMES) == len(YLIMS))
assert YLIMS.shape[1] == 2


plt.rc('font', family='serif')
plt.rc('text', usetex=True)
fig, axes = plt.subplots(n_params, 1, figsize=(24, 12 * n_params))

for param_idx in range(n_params):
    ax = axes[param_idx]
    ax.plot(
        SNRS, PRIORS[:, param_idx],
        label='prior', linewidth=4, marker='o', color='grey', alpha=0.5
    )
    ax.plot(
        SNRS, INTERIOR_POINT_POSTERIORS[:, param_idx],
        label='interior point posterior', linewidth=4, marker='o', color='blue'
    )
    ax.plot(
        SNRS, CROSS_ENTROPY_POSTERIORS[:, param_idx],
        label='cross entropy posterior', linewidth=4, marker='o', color='magenta'
    )
    ax.plot(
        SNRS, [TRUES[param_idx] for _ in range(n_points)],
        label='true', linewidth=4, linestyle='dashed', color='green'
    )

    ax.grid(alpha=0.75)
    ax.tick_params(
        axis='both', labelsize=60, direction='in',
        length=12, width=3, pad=12
    )
    n_ticks = 5
    y_min = YLIMS[param_idx][0]
    y_max = YLIMS[param_idx][1]
    step = (y_max - y_min) / (n_ticks - 1)
    ax.set_yticks(np.arange(y_min, y_max + step, step))
    ax.set_ylim(YLIMS[param_idx])
    ax.set_xticks(range(0, n_points + 1, 5))

    ax.set_xlabel('SNR', fontsize=60)
    param_name = PARAMS_NAMES[param_idx]
    ax.set_ylabel('$' + param_name + '$', labelpad=20, fontsize=60)
    ax.legend(loc='upper right', prop={'size': 40}, frameon=True, ncol=2)

fig.tight_layout()
plt.savefig(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'params_convergences.pdf'
    ),
    dpi=180, format='pdf'
)











