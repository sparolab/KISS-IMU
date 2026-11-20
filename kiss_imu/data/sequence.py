"""One recorded sequence: IMU stream, GT trajectory and LiDAR scans, all in the IMU frame."""

import copy
import json
import os

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from .profiles import get_profile


def _load_csv(path):
    """Comma-separated, or whitespace-separated (e.g. ENWIDE's gt-<seq>.csv) when the first line has no comma."""
    with open(path) as f:
        comma = ',' in f.readline()
    return np.loadtxt(path, delimiter=',' if comma else None, ndmin=2)


class Sequence:
    def __init__(self, root, name, data_type='generic', gt_time_offset=0.0, min_range=1.0, max_range=100.0,
                 imu_dir=None, label_dir=None, despike=None, imu_ext=False, gt_file='gt_pose.csv', scan_time_offset=0.0,
                 lidar_rotvec_deg=None):
        """imu_dir: optional directory with <name>.csv used in place of the sequence's imu.csv (same format,
        same frame and clock), e.g. another IMU of the same rig.
        label_dir: optional <label_dir>/<name>/gt_pose.csv used in place of the GT (pseudo-labels, e.g. LO/PGO
        poses; already on the sensor clock, so no time offset) with <label_dir>/<name>/invalid.txt holding the
        stamps of poses not to be trusted (windows near them are dropped).
        imu_ext: the imu_dir IMU is in its own frame (axes and origin) with <imu_dir>/<name>.ext.json holding its
        pose in the dataset IMU frame S (R_sa, p_sa: x_S = R_sa x_A + p_sa). The sequence then lives in that IMU's
        frame A: LiDAR points go S -> A and the GT body S is moved to A (T_W_A = T_W_S T_S_A); no lever-arm
        transfer of the accelerometer is needed. Pseudo-labels (label_dir) are taken to be in A already.
        gt_file: the GT file in the sequence directory.
        scan_time_offset: s added to the scan stamps to put them on the IMU clock at the time the points describe
        (e.g. end-of-sweep stamps; estimate it with tools/check_extrinsics.py).
        lidar_rotvec_deg: extra rotation applied to the points after the profile's R_I_L, x <- R x (a refined
        LiDAR->IMU rotation, estimated GT-free by tools/check_extrinsics.py --pool)."""
        self.root, self.name = root, name
        self.dir = os.path.join(root, name)
        self.profile = get_profile(data_type)
        self.min_range, self.max_range = min_range, max_range
        self.R_sa, self.p_sa = np.eye(3), np.zeros(3)
        self.R_fix = Rotation.from_rotvec(np.radians(lidar_rotvec_deg)).as_matrix() if lidar_rotvec_deg else np.eye(3)
        if imu_ext:
            e = json.load(open(os.path.join(imu_dir, name + '.ext.json')))
            self.R_sa, self.p_sa = np.asarray(e['R_sa'], float), np.asarray(e['p_sa'], float)

        imu = _load_csv(os.path.join(imu_dir, name + '.csv') if imu_dir else os.path.join(self.dir, 'imu.csv'))
        keep = np.r_[True, np.diff(imu[:, 0]) > 0]              # drop repeated stamps
        imu = imu[keep]
        self.imu_t = imu[:, 0]
        g, a = self.profile.gyr_col, self.profile.acc_col
        self.gyr = imu[:, g:g + 3].copy()
        self.acc = imu[:, a:a + 3].copy()
        self.n_despiked = {}
        if despike:                      # e.g. {acc: 50.0, gyr: 5.0}: see kiss_imu.imu.clean
            from kiss_imu.imu.clean import despike as _ds
            for key in ('acc', 'gyr'):
                if despike.get(key):
                    v, nb = _ds(self.imu_t, getattr(self, key), float(despike[key]), float(despike.get('win_s', 0.2)))
                    setattr(self, key, v)
                    self.n_despiked[key] = nb

        self.gt_t, self.gt_R, self.gt_p = None, None, None
        gt_path = os.path.join(self.dir, gt_file.format(name=name))   # e.g. 'gt-{name}.csv' (ENWIDE)
        self.invalid_t = np.zeros(0)
        self.anchor_t = None             # label stamps when labels exist only there (pseudo-labels)
        self.label_q = None              # (n, 2) per-label LO quality, see tools/make_pseudo_labels.py
        if label_dir:
            gt_path, gt_time_offset = os.path.join(label_dir, name, 'gt_pose.csv'), 0.0
            inv = os.path.join(label_dir, name, 'invalid.txt')
            if os.path.exists(inv) and os.path.getsize(inv) > 0:
                self.invalid_t = np.sort(np.loadtxt(inv, ndmin=1))
            q = os.path.join(label_dir, name, 'quality.txt')
            if os.path.exists(q):        # stamp, LO Hessian min eigenvalue / median (1 = typical, small = weakly constrained)
                self.label_q = np.loadtxt(q, ndmin=2)
        if os.path.exists(gt_path):
            self._load_gt(gt_path, gt_time_offset, to_imu=not label_dir)
            if label_dir:
                self.anchor_t = self.gt_t.copy()

        pts_dir = os.path.join(self.dir, 'points', 'data')
        self.scan_files, self.scan_t = [], np.zeros(0)
        if os.path.isdir(pts_dir):
            files = sorted(f for f in os.listdir(pts_dir) if f.endswith('.bin'))
            ts = np.loadtxt(os.path.join(self.dir, 'points', 'timestamps.txt'), ndmin=1)
            n = min(len(files), len(ts))
            self.scan_files = [os.path.join(pts_dir, f) for f in files[:n]]
            self.scan_t = ts[:n].astype(np.float64) + float(scan_time_offset)

    # ------------------------------------------------------------------ IMU
    @property
    def imu_rate(self):
        return 1.0 / np.median(np.diff(self.imu_t))

    # ------------------------------------------------------------------ GT
    def _load_gt(self, path, offset, to_imu=True):
        gt = _load_csv(path)
        gt = gt[np.r_[True, np.diff(gt[:, 0]) > 0]]
        t = gt[:, 0] + offset
        if gt.shape[1] == 8:
            p, Rm = gt[:, 1:4], Rotation.from_quat(gt[:, 4:8]).as_matrix()
        elif gt.shape[1] == 13:
            M = gt[:, 1:].reshape(-1, 3, 4)
            p, Rm = M[:, :, 3], M[:, :, :3]
        else:
            raise ValueError(f'{path}: expected 7 or 12 pose columns, got {gt.shape[1] - 1}')
        self._set_gt(t, p, Rm, to_imu)

    def _set_gt(self, t, p, Rm, to_imu):
        if to_imu:                       # GT body S -> IMU frame A
            p = p + Rm @ self.p_sa
            Rm = Rm @ self.R_sa
        Rwg = self.profile.R_W_G
        self.gt_t = t
        self.gt_R = np.einsum('ij,njk->nik', Rwg, Rm)
        self.gt_p = (p - p[0]) @ Rwg.T
        self._slerp = Slerp(t, Rotation.from_matrix(self.gt_R))
        self._gt_v = None

    def set_labels(self, P, invalid_t=None, quality=None):
        """Pseudo-labels from memory, as label_dir would load them: P (n, 8) TUM rows on the sensor clock, in the
        sequence's IMU frame; invalid_t stamps not to trust; quality (m, 2) [stamp, LO quality]. Replaces the GT."""
        P = np.asarray(P, float)
        P = P[np.r_[True, np.diff(P[:, 0]) > 0]]
        self._set_gt(P[:, 0], P[:, 1:4], Rotation.from_quat(P[:, 4:8]).as_matrix(), to_imu=False)
        self.anchor_t = self.gt_t.copy()
        self.invalid_t = np.zeros(0) if invalid_t is None else np.sort(np.asarray(invalid_t, float).ravel())
        self.label_q = None if quality is None else np.asarray(quality, float).reshape(-1, 2)
        return self

    def until(self, t_end):
        """A shallow copy that holds only the data up to t_end (IMU samples and scans), e.g. what an online
        system has seen when it trains; the GT/labels are left as they are."""
        c = copy.copy(self)
        n = int(np.searchsorted(self.imu_t, t_end, side='right'))
        c.imu_t, c.gyr, c.acc = self.imu_t[:n], self.gyr[:n], self.acc[:n]
        m = int(np.searchsorted(self.scan_t, t_end, side='right'))
        c.scan_files, c.scan_t = self.scan_files[:m], self.scan_t[:m]
        return c

    @property
    def has_gt(self):
        return self.gt_t is not None

    def gt_velocity(self, window_s=0.5, order=2):
        """World velocity at the GT stamps, Savitzky-Golay derivative of the positions."""
        if self._gt_v is None:
            dt = np.median(np.diff(self.gt_t))
            w = max(int(round(window_s / dt)) | 1, order + 2 + (order % 2 == 0))
            w = w if w % 2 else w + 1
            self._gt_v = savgol_filter(self.gt_p, w, order, deriv=1, delta=dt, axis=0)
        return self._gt_v

    def gt_states(self, t):
        """GT rotation (N,3,3), position (N,3) and velocity (N,3) at times t (clipped to the GT span)."""
        t = np.clip(np.atleast_1d(t), self.gt_t[0], self.gt_t[-1])
        Rm = self._slerp(t).as_matrix()
        p = np.stack([np.interp(t, self.gt_t, self.gt_p[:, i]) for i in range(3)], 1)
        v_gt = self.gt_velocity()
        v = np.stack([np.interp(t, self.gt_t, v_gt[:, i]) for i in range(3)], 1)
        return Rm, p, v

    def label_quality(self, t):
        """LO quality at times t (nearest label stamp; 1 without quality info, nan -> 0)."""
        if self.label_q is None:
            return np.ones(len(t))
        i = np.searchsorted(self.label_q[:, 0], t).clip(1, len(self.label_q) - 1)
        j = np.where(np.abs(self.label_q[i - 1, 0] - t) < np.abs(self.label_q[i, 0] - t), i - 1, i)
        return np.nan_to_num(self.label_q[j, 1], nan=0.0)

    def in_gt_span(self, t, margin=0.0, invalid_radius=0.15):
        ok = (t >= self.gt_t[0] + margin) & (t <= self.gt_t[-1] - margin)
        if len(self.invalid_t):
            i = np.searchsorted(self.invalid_t, t)
            lo = np.abs(t - self.invalid_t[np.clip(i - 1, 0, len(self.invalid_t) - 1)])
            hi = np.abs(t - self.invalid_t[np.clip(i, 0, len(self.invalid_t) - 1)])
            ok &= np.minimum(lo, hi) > invalid_radius
        return ok

    # ------------------------------------------------------------------ LiDAR
    def __len__(self):
        return len(self.scan_files)

    def scan(self, i):
        """Points of scan i in the IMU frame, (N, 3) float64, range-filtered."""
        raw = np.fromfile(self.scan_files[i], dtype=np.float64).reshape(-1, self.profile.lidar_fields)[:, :3]
        raw = raw[np.isfinite(raw).all(1)]
        r = np.linalg.norm(raw, axis=1)
        raw = raw[(r > self.min_range) & (r < self.max_range)]
        x = (raw @ self.profile.R_I_L.T + self.profile.t_I_L) @ self.R_fix.T
        return (x - self.p_sa) @ self.R_sa              # S -> A (identity without imu_ext)
