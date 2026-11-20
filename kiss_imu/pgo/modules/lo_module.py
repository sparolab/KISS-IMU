"""LiDAR odometry module: scan-to-local-map registration -> a world pose measurement.

    measure(scan, T_guess)  register the scan (IMU frame) against the local map, seeded with
                            the IMU prediction; None when the map is empty or the result is rejected
    insert(scan, T)         add the scan to the map at pose T if it is a keyframe

Registration backend (lo.registration) and map (lo.map) are chosen by name from their
registries, so either can be swapped without touching the graph.
"""

from dataclasses import dataclass, field

import numpy as np

from kiss_imu.geometry import so3

from ..mapping import build_map, voxel_downsample
from ..registration import build_registration


@dataclass
class LOMeasurement:
    R: np.ndarray
    p: np.ndarray
    info: np.ndarray                       # 6x6 for PoseFactor's residual [rot (body), pos (world)]
    stats: dict = field(default_factory=dict)


def transform(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


class LOModule:
    def __init__(self, cfg):
        self.reg = build_registration(cfg['registration'])
        self.map = build_map(cfg['map'])
        self.info_cfg = cfg.get('info', {'mode': 'fixed', 'rot_sigma': 0.01, 'pos_sigma': 0.05})
        g = cfg.get('gate', {})
        self.min_inlier_ratio = g.get('min_inlier_ratio', 0.3)
        self.require_converged = g.get('require_converged', True)   # small_gicp hitting max_iterations is often still fine
        self.max_jump = (g.get('max_jump_m', 1.0), np.radians(g.get('max_jump_deg', 10.0)))
        kf = cfg.get('keyframe', {})
        self.kf_dist, self.kf_angle = kf.get('distance', 0.5), np.radians(kf.get('angle_deg', 5.0))
        self.map_voxel = cfg['map'].get('voxel', 0.5)
        self._target, self._target_version = None, -1
        self._last_kf = None
        self.jump_scale = 1.0     # set by the PGO: widens the jump gate after a LiDAR gap (the prediction drifted)

    # ------------------------------------------------------------------ measurement
    def _info(self, res, R_m):
        c = self.info_cfg
        if c['mode'] == 'fixed':
            return np.diag(np.r_[np.full(3, c['rot_sigma'] ** -2), np.full(3, c['pos_sigma'] ** -2)])
        if c['mode'] == 'hessian':
            # small_gicp H is in [rot, trans] with T Exp(d); our position residual is in the world frame
            J = np.eye(6)
            J[3:, 3:] = R_m
            H = np.asarray(res.H)
            if c.get('normalize', 'none') == 'residual':
                # GICP sums Mahalanobis terms under unit-free point covariances, so H has no physical scale:
                # divide by the residual variance per inlier dof 
                dof = max(3 * res.num_inliers - 6, 1)
                H = H / max(2.0 * res.error / dof, 1e-12)
            info = J @ H @ J.T * c.get('scale', 1.0)
            floor = np.r_[np.full(3, c.get('rot_sigma_max', 1.0) ** -2), np.full(3, c.get('pos_sigma_max', 10.0) ** -2)]
            return info + np.diag(floor)
        raise ValueError(f"unknown lo.info.mode {c['mode']!r}")

    def measure(self, scan, T_guess):
        if self.map.empty:
            return None
        if self._target_version != self.map.version:
            self._target = self.reg.prepare_target(self.map.points())
            self._target_version = self.map.version
        res = self.reg.align(self.reg.prepare_source(scan), self._target, T_guess)
        T = res.T
        jump = np.linalg.inv(T_guess) @ T
        stats = dict(converged=res.converged, inlier_ratio=res.inlier_ratio, iterations=res.iterations,
                     error=res.error, jump_m=float(np.linalg.norm(jump[:3, 3])),
                     jump_deg=float(np.degrees(np.linalg.norm(so3.log(jump[:3, :3])))))
        if res.H is not None:
            ev = np.linalg.eigvalsh(0.5 * (res.H + res.H.T))
            stats.update(h_min_eig=float(ev[0]), h_max_eig=float(ev[-1]))      # LO quality indicators
        ok = ((res.converged or not self.require_converged) and res.inlier_ratio >= self.min_inlier_ratio
              and stats['jump_m'] <= self.jump_scale * self.max_jump[0]
              and np.radians(stats['jump_deg']) <= self.jump_scale * self.max_jump[1])
        info = self._info(res, T[:3, :3]) if ok else None
        stats['accepted'] = bool(ok)
        return LOMeasurement(T[:3, :3], T[:3, 3], info, stats)

    # ------------------------------------------------------------------ map
    def is_keyframe(self, T):
        if self._last_kf is None:
            return True
        d = np.linalg.inv(self._last_kf) @ T
        return np.linalg.norm(d[:3, 3]) >= self.kf_dist or np.linalg.norm(so3.log(d[:3, :3])) >= self.kf_angle

    def insert(self, scan, T, force=False):
        if not (force or self.is_keyframe(T)):
            return False
        self.map.insert(transform(voxel_downsample(scan, self.map_voxel), T), T)
        self._last_kf = T.copy()
        return True
