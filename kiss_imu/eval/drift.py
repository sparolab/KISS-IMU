"""Long-horizon gyro-only attitude drift: integrate the gyro over H seconds from the GT attitude and
compare with the GT relative rotation. Short (0.1 s) segment errors are dominated by white noise in the
common slow motion; drift over tens of seconds exposes the systematic errors (scale, misalignment) that
build up in the rarer fast turns."""

import numpy as np
from scipy.spatial.transform import Rotation


def integrate_gyro(ts, gyr):
    """(N, 3, 3) attitude from integrating the gyro, R[0] = I."""
    dt = np.r_[np.diff(ts), 0.0]
    inc = Rotation.from_rotvec(gyr * dt[:, None]).as_matrix()
    R = np.empty_like(inc)
    R[0] = np.eye(3)
    for k in range(len(dt) - 1):
        R[k + 1] = R[k] @ inc[k]
    return R


def attitude_drift(seq, gyr, horizons=(10.0, 30.0, 60.0), stride_frac=0.5, t_from=None):
    """{H: (total_deg, yaw_deg)} arrays of per-window errors (windows every H * stride_frac seconds, starting
    at t_from or later, e.g. only after a warm-start hand-over)."""
    Rc = integrate_gyro(seq.imu_t, gyr)
    out = {}
    start = seq.gt_t[0] + 1.0 if t_from is None else max(seq.gt_t[0] + 1.0, float(t_from))
    for H in horizons:
        t0 = np.arange(start, seq.gt_t[-1] - H - 1.0, H * stride_frac)
        if len(t0) == 0:
            continue
        i0, i1 = np.searchsorted(seq.imu_t, t0), np.searchsorted(seq.imu_t, t0 + H)
        A, B = seq.gt_states(seq.imu_t[i0])[0], seq.gt_states(seq.imu_t[i1])[0]
        Rg = np.einsum('nji,njk->nik', A, B)
        Ri = np.einsum('nji,njk->nik', Rc[i0], Rc[i1])
        E = np.einsum('nji,njk->nik', Ri, Rg)                       # body-frame error
        tot = np.degrees(np.linalg.norm(Rotation.from_matrix(E).as_rotvec(), axis=1))
        Ew = np.einsum('nij,njk,nlk->nil', A, E, A)                 # error in the world frame
        yaw = np.degrees(np.abs(Rotation.from_matrix(Ew).as_rotvec()[:, 2]))
        out[H] = (tot, yaw)
    return out


def drift_rmse(per_seq):
    """Pool per-sequence attitude_drift outputs into {H: {'rot': rmse, 'yaw': rmse}}."""
    res = {}
    for H in sorted({h for d in per_seq for h in d}):
        tot = np.concatenate([d[H][0] for d in per_seq if H in d])
        yaw = np.concatenate([d[H][1] for d in per_seq if H in d])
        res[H] = {'rot': float(np.sqrt(np.mean(tot ** 2))), 'yaw': float(np.sqrt(np.mean(yaw ** 2)))}
    return res


def velocity_drift(seq, acc, horizons=(1.0, 5.0, 10.0), stride_frac=0.5, t_from=None):
    """{H: (err_m_s,)} accelerometer-only velocity drift: integrate the specific force rotated with the GT
    attitude (so gyro errors do not enter) over H seconds and compare with the GT velocity change."""
    t = seq.imu_t
    m = (t >= seq.gt_t[0]) & (t <= seq.gt_t[-1])
    Rw = seq.gt_states(t)[0]
    dt = np.r_[np.diff(t), 0.0]
    a_w = np.einsum('nij,nj->ni', Rw, acc) + np.array([0.0, 0.0, -9.81])
    cum = np.r_[np.zeros((1, 3)), np.cumsum(a_w * dt[:, None], 0)]
    out = {}
    start = seq.gt_t[0] + 1.0 if t_from is None else max(seq.gt_t[0] + 1.0, float(t_from))
    for H in horizons:
        t0 = np.arange(start, seq.gt_t[-1] - H - 1.0, H * stride_frac)
        if len(t0) == 0 or not m.any():
            continue
        i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t0 + H)
        dv = cum[i1] - cum[i0]
        v0, v1 = seq.gt_states(t[i0])[2], seq.gt_states(t[i1])[2]
        out[H] = (np.linalg.norm(dv - (v1 - v0), axis=1),)
    return out


def vel_rmse(per_seq):
    res = {}
    for H in sorted({h for d in per_seq for h in d}):
        e = np.concatenate([d[H][0] for d in per_seq if H in d])
        res[H] = float(np.sqrt(np.mean(e ** 2)))
    return res
