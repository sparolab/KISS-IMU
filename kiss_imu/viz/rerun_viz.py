"""Live view of the PGO in Rerun (https://rerun.io), as an observer of SlidingWindowPGO.

    world/map/kf_*        keyframes as they enter the LO map (coloured by height): the map building up
    world/submap          the current registration target (the sliding submap), refreshed every few changes
    world/scan            the current scan at its optimised pose
    world/robot           the newest node (axes + box); the 'Follow' view tracks it
    world/window          the nodes of the sliding window
    world/imu_pred        where the IMU alone put the node (before LO)
    world/lo_meas         the LO measurement (green accepted, red rejected)
    world/traj/{online,gt}
    plots/...             LO quality (Hessian min eigenvalue), error vs GT, IMU raw vs the
                          IO stream (raw or corrected), biases, timing

Sinks: 'spawn' (open a viewer), 'save' (.rrd file, open with `rerun file.rrd`), 'connect' (a running viewer).
"""

import os
import sys

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from kiss_imu.geometry import so3

TEAL, ROSE, AMBER, GREEN, GRAY = (0, 190, 190), (230, 60, 110), (255, 170, 0), (60, 200, 90), (150, 150, 160)
_RAMP = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)


def height_colors(z, lo, hi):
    u = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1) * (len(_RAMP) - 1)
    return np.stack([np.interp(u, np.arange(len(_RAMP)), _RAMP[:, c]) for c in range(3)], 1).astype(np.uint8)


def voxel(points, size):
    _, idx = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    return points[idx]


def transform(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def blueprint():
    follow = rrb.Spatial3DView(name='Follow', origin='world', background=[18, 20, 28],
                               eye_controls=rrb.EyeControls3D(tracking_entity='world/robot'))
    overview = rrb.Spatial3DView(name='Overview', origin='world', background=[18, 20, 28])
    plots = rrb.Vertical(
        rrb.TimeSeriesView(name='LO quality', origin='plots/lo'),
        rrb.TimeSeriesView(name='LO quality: log10 min eig(H)', origin='plots/lo_quality'),
        rrb.TimeSeriesView(name='Error vs GT', origin='plots/error'),
        rrb.TimeSeriesView(name='IMU raw vs IO stream', origin='plots/imu'),
        rrb.TimeSeriesView(name='Biases', origin='plots/bias'),
        rrb.TextLogView(name='Events', origin='events'),
    )
    return rrb.Blueprint(rrb.Horizontal(rrb.Tabs(follow, overview), plots, column_shares=[3, 2]),
                         collapse_panels=True)


class RerunViz:
    def __init__(self, seq, sink='spawn', path=None, scan_points=10000, kf_voxel=0.4, submap_every=5,
                 traj_every=5, app_id='kiss_imu'):
        self.seq = seq
        self.scan_points, self.kf_voxel = scan_points, kf_voxel
        self.submap_every, self.traj_every = submap_every, traj_every
        os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')
        rr.init(app_id, recording_id=f'{app_id}_{seq.name}')
        if sink == 'spawn':
            rr.spawn(memory_limit='40%')
        elif sink == 'save':
            rr.save(path)
        elif sink == 'connect':
            rr.connect_grpc(path) if path else rr.connect_grpc()
        else:
            raise ValueError(f'unknown rerun sink {sink!r} (spawn | save | connect)')
        rr.send_blueprint(blueprint())
        self.k, self.n_kf, self._map_version = 0, 0, -1
        self.zrange = (-2.0, 8.0)
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ helpers
    def _time(self, t):
        rr.set_time('time', timestamp=t)
        rr.set_time('scan', sequence=self.k)

    def _scalar(self, path, v):
        if v is not None and np.isfinite(v):
            rr.log(path, rr.Scalars(float(v)))

    # ------------------------------------------------------------------ callbacks
    def on_start(self, pgo, t, scan):
        rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        s = self.seq
        if s.has_gt:
            rr.log('world/traj/gt', rr.LineStrips3D([s.gt_p], colors=[GRAY], radii=[0.06]), static=True)
            z = s.gt_p[:, 2]
            self.zrange = (z.min() - 2.0, z.max() + 6.0)
        rr.log('world/robot/body', rr.Boxes3D(half_sizes=[[0.4, 0.25, 0.15]], colors=[AMBER], fill_mode='solid'),
               static=True)
        rr.log('world/robot/axes', rr.Arrows3D(vectors=np.eye(3) * 1.2, colors=[(230, 60, 60), (60, 200, 60), (60, 120, 240)],
                                               radii=0.04), static=True)
        for name, col in (('plots/imu/gyr_z_raw', GRAY), ('plots/imu/gyr_z_io', TEAL),
                          ('plots/imu/acc_x_raw', GRAY), ('plots/imu/acc_x_io', ROSE),
                          ('plots/error/pos_m', ROSE), ('plots/error/rot_deg', AMBER),
                          ('plots/lo/inlier_ratio', TEAL), ('plots/lo/accepted', GREEN),
                          ('plots/lo/correction_m', AMBER), ('plots/lo_quality/min_eig', ROSE),
                          ('plots/lo_quality/max_eig', TEAL)):
            rr.log(name, rr.SeriesLines(colors=[col], names=[name.rsplit('/', 1)[1]]), static=True)
        for n, c in enumerate('xyz'):
            rr.log(f'plots/bias/bg_{c}', rr.SeriesLines(names=[f'bg_{c} (deg/s)']), static=True)
        rr.log('events', rr.TextLog(f'start {self.seq.name}: io.mode={pgo.io.mode}, '
                                    f'registration={pgo.lo.reg.name}, map={pgo.lo.map.name}'), static=True)
        self._time(t)
        self._pose(pgo)

    def on_keyframe(self, pgo, t, scan, T):
        pts = transform(voxel(scan, self.kf_voxel), T)
        rr.log(f'world/map/kf_{self.n_kf:05d}', rr.Points3D(pts, colors=height_colors(pts[:, 2], *self.zrange), radii=0.05))
        self.n_kf += 1
        m = pgo.lo.map
        if m.version != self._map_version and (m.version % self.submap_every == 0 or self._map_version < 0):
            self._map_version = m.version
            sub = voxel(m.points(), 0.5)
            rr.log('world/submap', rr.Points3D(sub, colors=[(240, 240, 255)], radii=0.03))

    def on_step(self, pgo, info):
        self.k += 1
        t, rec = info['t'], info['rec']
        self._time(t)
        new = pgo.graph.states[pgo.graph.newest_key]

        scan = info['scan']
        if len(scan) > self.scan_points:
            scan = scan[self.rng.choice(len(scan), self.scan_points, replace=False)]
        rr.log('world/scan', rr.Points3D(transform(scan, new.T), colors=[TEAL], radii=0.04))
        self._pose(pgo)

        pred, meas = info['pred'], info['meas']
        rr.log('world/imu_pred', rr.Points3D([pred.p], colors=[AMBER], radii=0.15))
        if meas is not None:
            ok = meas.info is not None
            rr.log('world/lo_meas', rr.Points3D([meas.p], colors=[GREEN if ok else ROSE], radii=0.15))
            st = meas.stats
            self._scalar('plots/lo/inlier_ratio', st.get('inlier_ratio'))
            self._scalar('plots/lo/accepted', float(ok))
            self._scalar('plots/lo/correction_m', st.get('jump_m'))
            if 'h_min_eig' in st:
                self._scalar('plots/lo_quality/min_eig', np.log10(max(st['h_min_eig'], 1e-12)))
                self._scalar('plots/lo_quality/max_eig', np.log10(max(st['h_max_eig'], 1e-12)))
            if not ok:
                rr.log('events', rr.TextLog(f'scan {self.k}: LO rejected (inliers {st["inlier_ratio"]:.2f}, '
                                            f'jump {st["jump_m"]:.2f} m / {st["jump_deg"]:.1f} deg)', level='WARN'))

        if self.k % self.traj_every == 0:
            rr.log('world/traj/online', rr.LineStrips3D([np.array(pgo.online)[:, 1:4]], colors=[TEAL], radii=0.08))

        s = self.seq
        if s.has_gt and s.in_gt_span(np.array([t]))[0]:
            gR, gp, _ = (x[0] for x in s.gt_states(np.array([t])))
            rr.log('world/gt_now', rr.Points3D([gp], colors=[(255, 255, 255)], radii=0.12))
            self._scalar('plots/error/pos_m', np.linalg.norm(new.p - gp))
            self._scalar('plots/error/rot_deg', np.degrees(np.linalg.norm(so3.log(gR.T @ new.R))))

        io = pgo.io
        k0, k1 = np.searchsorted(io.ts, [pred.t - info['pre'].dt, t])
        if k1 > k0:
            self._scalar('plots/imu/gyr_z_raw', np.degrees(s.gyr[k0:k1, 2].mean()))
            self._scalar('plots/imu/gyr_z_io', np.degrees(io.gyr[k0:k1, 2].mean()))
            self._scalar('plots/imu/acc_x_raw', s.acc[k0:k1, 0].mean())
            self._scalar('plots/imu/acc_x_io', io.acc[k0:k1, 0].mean())
        for n, c in enumerate('xyz'):
            self._scalar(f'plots/bias/bg_{c}', np.degrees(new.bg[n]))

    def on_finish(self, pgo):
        rr.log('world/traj/online', rr.LineStrips3D([np.array(pgo.online)[:, 1:4]], colors=[TEAL], radii=0.08))
        rr.log('events', rr.TextLog(f'finished {self.seq.name}: {self.k} scans'))

    # ------------------------------------------------------------------
    def _pose(self, pgo):
        g = pgo.graph
        new = g.states[g.newest_key]
        rr.log('world/robot', rr.Transform3D(translation=new.p, mat3x3=new.R))
        win = np.array([x.p for x in g.states.values()])
        rr.log('world/window', rr.LineStrips3D([win], colors=[AMBER], radii=0.05))
        rr.log('world/window/nodes', rr.Points3D(win, colors=[AMBER], radii=0.1))
