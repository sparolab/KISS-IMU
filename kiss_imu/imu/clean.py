"""IMU glitch removal. The ANYmal built-in IMU (GrandTour) has ~25 ms bursts of accelerometer samples at
100-450 m/s^2 (10 of 36 missions; the STIM320 is quiet at the same instants); integrated they put
~10 m/s into the velocity at once. A sample is a glitch when it is further than `thresh` from the median
of the surrounding `win_s`; glitches are replaced by linear interpolation of the clean neighbours."""

import numpy as np
from scipy.ndimage import median_filter


def despike(ts, x, thresh, win_s=0.2):
    n = max(int(round(win_s / np.median(np.diff(ts)))) | 1, 3)
    med = median_filter(x, size=(n, 1), mode='nearest')
    bad = np.linalg.norm(x - med, axis=1) > thresh
    if not bad.any():
        return x, 0
    good = ~bad
    y = x.copy()
    for k in range(x.shape[1]):
        y[bad, k] = np.interp(ts[bad], ts[good], x[good, k])
    return y, int(bad.sum())
