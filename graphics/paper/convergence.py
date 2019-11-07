import numpy as np
import matplotlib.pyplot as plt


SNRS = 1.0 * np.arange(1, 21, 1)
assert len(SNRS) == 20


def get_posteriors(filename):
    raw_posteriors = np.load(filename)
    posteriors_mean = raw_posteriors.mean(axis=1)
    posteriors_std = raw_posteriors.std(axis=1, ddof=1)
    assert raw_posteriors.shape == (20, 50, 4)
    assert posteriors_mean.shape == posteriors_std.shape == (20, 4)
    return posteriors_mean, posteriors_std


IP_POSTERIORS_MEAN, IP_POSTERIORS_STD = get_posteriors('IP_posteriors.npy')
BFGS_POSTERIORS_MEAN, BFGS_POSTERIORS_STD = get_posteriors('BFGS_posteriors.npy')
CE_POSTERIORS_MEAN, CE_POSTERIORS_STD = get_posteriors('CE_posteriors.npy')

TRUES = np.array([0.25, 1.00, 1.00, 0.01])
PARAMS_NAMES = ("D", "E^{'}", "M", r"X_{\! d}^{'}")

n_points = len(SNRS)
n_params = len(TRUES)
YLIMS = np.array([[0.0 * TRUES[i], 2.0 * TRUES[i]] for i in range(n_params)])


assert len(SNRS) == len(IP_POSTERIORS_MEAN) == len(IP_POSTERIORS_STD) == len(BFGS_POSTERIORS_MEAN) == len(BFGS_POSTERIORS_STD) == len(CE_POSTERIORS_MEAN) == len(CE_POSTERIORS_STD)
assert len(TRUES) == len(PARAMS_NAMES)
assert (IP_POSTERIORS_MEAN.shape[1] == IP_POSTERIORS_STD.shape[1] == CE_POSTERIORS_MEAN.shape[1] == CE_POSTERIORS_STD.shape[1] == BFGS_POSTERIORS_MEAN.shape[1] == BFGS_POSTERIORS_STD.shape[1]
        == len(TRUES) == len(PARAMS_NAMES) == len(YLIMS))
assert YLIMS.shape[1] == 2


plt.rc('font', family='serif')
plt.rc('text', usetex=True)
fig, axes = plt.subplots(n_params, 1, figsize=(24, 12 * n_params))


for param_idx in range(n_params):
    ax = axes[param_idx]

    ax.plot(
        SNRS, IP_POSTERIORS_MEAN[:, param_idx],
        label='interior point posterior', linewidth=4, marker='o', color='red'
    )
    ax.fill_between(
        SNRS,
        IP_POSTERIORS_MEAN[:, param_idx] - IP_POSTERIORS_STD[:, param_idx],
        IP_POSTERIORS_MEAN[:, param_idx] + IP_POSTERIORS_STD[:, param_idx],
        alpha=0.05, color='red'
    )

    ax.plot(
        SNRS, BFGS_POSTERIORS_MEAN[:, param_idx],
        label='BFGS posterior', linewidth=4, marker='o', color='blue'
    )
    ax.fill_between(
        SNRS,
        BFGS_POSTERIORS_MEAN[:, param_idx] - BFGS_POSTERIORS_STD[:, param_idx],
        BFGS_POSTERIORS_MEAN[:, param_idx] + BFGS_POSTERIORS_STD[:, param_idx],
        alpha=0.1, color='blue'
    )

    ax.plot(
        SNRS, CE_POSTERIORS_MEAN[:, param_idx],
        label='cross entropy posterior', linewidth=4, marker='o', color='green'
    )
    ax.fill_between(
        SNRS,
        CE_POSTERIORS_MEAN[:, param_idx] - CE_POSTERIORS_STD[:, param_idx],
        CE_POSTERIORS_MEAN[:, param_idx] + CE_POSTERIORS_STD[:, param_idx],
        alpha=0.15, color='green'
    )

    ax.plot(
        SNRS, [TRUES[param_idx] for _ in range(n_points)],
        label='true', linewidth=4, linestyle='dashed', color='black'
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
    'params_convergences.pdf',
    dpi=180, format='pdf'
)


