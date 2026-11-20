"""IMU measurement variance from the data itself (first-difference variance).

For a white-noise sensor var(x_k - x_{k-1}) = 2 var(x) and var(2nd difference) = 6 var(x); smooth motion adds
little at the sampling rate, while shocks, vibration and aliasing (legged gait) raise it. The per-sample,
per-axis variance is the causal (trailing) mean of the squared differences over win_s, floored at the
nominal density^2 / dt. It becomes the preintegration noise of that sample, so the IMU factor covariance
follows how noisy the IMU actually is at that moment. No learning, no variance-component estimation.
"""

import numpy as np


def diff_var(ts, x, density, win_s=0.1, order=1):
    dt = np.diff(ts, prepend=ts[0] - np.median(np.diff(ts)))
    dt = np.clip(dt, 1e-4, None)
    d2 = np.vstack([np.zeros((order, x.shape[1])), np.diff(x, n=order, axis=0) ** 2 / (2 if order == 1 else 6)])
    c = np.cumsum(d2, 0)
    n = max(int(round(win_s / np.median(dt))), 1)
    m = (c - np.vstack([np.zeros((n, x.shape[1])), c[:-n]])) / np.minimum(np.arange(1, len(x) + 1), n)[:, None]
    return np.maximum(m, (density ** 2 / dt)[:, None])
