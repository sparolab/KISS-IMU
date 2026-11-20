"""Sliding-window LiDAR-inertial PGO.

One node per LiDAR scan. For scan k:

    1. IOModule preintegrates the IMU since node k-1 (at its bias) and predicts node k
    2. LOModule registers scan k to the local map seeded with the prediction -> pose factor
    3. node k joins the window with IMU, bias-random-walk and (if accepted) pose factors
    4. the window is optimised (LM); past `window` nodes the oldest is marginalised
    5. the map gets a scan per the insert policy:
         'newest'        the newest node's scan at its just-optimised pose (keyframes only)
         'marginalized'  each node's scan when it leaves the window, at its final pose

Two trajectories come out: `online` (newest node after each step) and `smoothed`
(each node when it leaves the window).

Observers (e.g. kiss_imu/viz/rerun_viz.py) get on_start / on_keyframe / on_step / on_finish callbacks;
they only read, so the estimate is the same with or without them.
"""

import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

from kiss_imu.geometry import so3
from kiss_imu.imu.preintegration import predict

from .factors import BiasFactor, IMUFactor, PoseFactor, PriorFactor
from .graph import SlidingWindowGraph
from .modules import IOModule, LOModule
from .state import NodeState


def tum_row(s):
    return np.r_[s.t, s.p, Rotation.from_matrix(s.R).as_quat()]


class SlidingWindowPGO:
    def __init__(self, cfg, io, lo):
        self.cfg, self.io, self.lo = cfg, io, lo
        g = cfg['graph']
        # graph.estimate_bias: false -> nodes are (R, p, v) only, as in the original KISS-IMU PVGO: no bias
        #   states, no bias random-walk factors; the IMU is preintegrated at zero bias
        self.estimate_bias = bool(g.get('estimate_bias', True))
        self.nd = 15 if self.estimate_bias else 9
        self.bias_jump = None            # pending [dbg, dba] for the next node (set at an IMU stream switch)
        self.graph = SlidingWindowGraph(g['window'], g.get('iters', 6), g.get('lm_lambda', 1e-4), nd=self.nd)
        self.insert_policy = cfg['lo'].get('insert', 'newest')
        self.huber = cfg['lo'].get('huber')
        self.online, self.smoothed, self.log = [], [], []
        self.scans = {}
        self.observers = []
        # lo.gate.mahalanobis: chi2 threshold (6 dof) of the LO pose against the IMU prediction, using the
        #   prediction covariance (window marginal of the last node + preintegration) plus the LO covariance;
        #   pred_floor adds model slack. null = off (only the LOModule jump gate).
        gate = cfg['lo'].get('gate', {})
        self.maha = gate.get('mahalanobis')
        fl = gate.get('pred_floor', {'rot_deg': 0.5, 'pos_m': 0.05})
        self.pred_floor = np.diag(np.r_[np.full(3, np.radians(fl['rot_deg']) ** 2), np.full(3, fl['pos_m'] ** 2)])
        # recovery: after `after` consecutive rejected LO measurements, re-initialise the window at the LO pose
        #   with the velocity of a line fit through those rejected LO positions (biases kept)
        rc = cfg.get('recovery') or {}
        self.recover_after = int(rc['after']) if rc.get('enabled') else None
        self.recover_sigmas = rc.get('prior_sigmas', [0.01, 0.05, 0.3, 1e-3, 1e-2])
        self.rejected = []                 # (t, R_m, p_m) of the current run of rejected LO measurements
        self.n_reinit = 0
        # lo.gate.gap_ref_s: the jump gate grows as (time since the last accepted LO) / gap_ref_s, so the first
        #   scan after a LiDAR gap can pull the drifted prediction back (None: fixed gate)
        self.gap_ref = gate.get('gap_ref_s')
        self.t_last_lo = None
        self.log_lo_pose = bool(cfg.get('log', {}).get('lo_pose', False))   # raw LO pose + info in steps.jsonl

    def _in_outage(self, t):
        o = getattr(self, 'outages', None)
        return o is not None and len(o) > 0 and bool(np.any((o[:, 0] <= t) & (t < o[:, 1])))

    def _sigmas(self, sigmas):
        return list(sigmas) if self.estimate_bias else list(sigmas)[:3]

    def _notify(self, event, *args):
        for o in self.observers:
            getattr(o, event)(self, *args)

    def _insert(self, scan, T, t, force=False):
        if self.lo.insert(scan, T, force=force):
            self._notify('on_keyframe', t, scan, T)

    def start(self, t, scan, R, p, v, bg=None, ba=None):
        x = NodeState(t, R, p, v, np.zeros(3) if bg is None else bg, np.zeros(3) if ba is None else ba)
        self.graph.add_node(0, x)
        self.graph.add_factor(PriorFactor(0, x, self._sigmas(self.cfg['init']['prior_sigmas'])))
        self.graph.optimize()                  # so the first gate has a marginal covariance
        self._notify('on_start', t, scan)
        self._insert(scan, x.T, t, force=True)
        if self.insert_policy == 'marginalized':
            self.scans[0] = scan
        self.online.append(tum_row(x))

    def step(self, t, scan):
        tic = time.perf_counter()
        i = self.graph.newest_key
        xi = self.graph.states[i]
        j = i + 1
        pre = self.io.preintegrate(xi.t, t, xi.bg, xi.ba)
        Rj, pj, vj = predict(xi.R, xi.p, xi.v, pre)
        jump = self.bias_jump if self.estimate_bias else None
        self.bias_jump = None
        xj = NodeState(t, Rj, pj, vj, xi.bg.copy(), xi.ba.copy())
        if jump is not None:
            xj.bg, xj.ba = xj.bg + jump[:3], xj.ba + jump[3:]
        pred = xj.copy()
        dark = self._in_outage(t)        # forced LiDAR outage: no registration, no map insertion
        if self.gap_ref is not None and self.t_last_lo is not None:   # lo.gate.gap_ref_s
            self.lo.jump_scale = max(1.0, (t - self.t_last_lo) / self.gap_ref)
        meas = None if dark else self.lo.measure(scan, xj.T)
        lo_raw = None                    # the LO pose and information as registered (before any gating)
        if meas is not None:
            lo_raw = dict(lo_pose=np.r_[meas.p, Rotation.from_matrix(meas.R).as_quat()].round(9).tolist(),
                          lo_info=None if meas.info is None else np.asarray(meas.info).reshape(-1).tolist())
        if meas is not None and meas.info is not None and self.maha is not None:
            d2 = self._mahalanobis(xi, pre, xj, meas)
            meas.stats['maha'] = float(d2)
            if d2 > self.maha:
                meas.stats['accepted'] = False
                meas.info = None
        if meas is not None and self.recover_after is not None:
            if meas.info is None:
                self.rejected.append((t, meas.R, meas.p))
                if len(self.rejected) >= self.recover_after:
                    return self._recover(t, scan, xi, pred, meas, pre, tic)
            else:
                self.rejected = []

        self.graph.add_node(j, xj)
        self.graph.add_factor(IMUFactor(i, j, pre, self.io.cov_scale))
        if self.estimate_bias:
            self.graph.add_factor(BiasFactor(i, j, pre.dt, self.io.gyr_rw, self.io.acc_rw, jump))
        if meas is not None and meas.info is not None:
            self.graph.add_factor(PoseFactor(j, meas.R, meas.p, meas.info, self.huber))
        cost = self.graph.optimize()

        if len(self.graph) > self.graph.size:
            k0 = self.graph.oldest_key
            final = self.graph.marginalize_oldest()
            self.smoothed.append(tum_row(final))
            if self.insert_policy == 'marginalized':
                self._insert(self.scans.pop(k0), final.T, final.t)
        new = self.graph.states[j]
        if dark:
            pass
        elif self.insert_policy == 'newest':
            self._insert(scan, new.T, t)
        else:
            self.scans[j] = scan

        self.online.append(tum_row(new))
        rec = dict(t=t, cost=cost, dt_ms=1e3 * (time.perf_counter() - tic), lo=meas is not None and meas.info is not None)
        if dark:
            rec['outage'] = True
        if rec['lo']:
            self.t_last_lo = t
        if self.estimate_bias:
            rec.update(bg=new.bg.round(7).tolist(), ba=new.ba.round(6).tolist())
        if meas is not None:
            rec.update({f'lo_{k}': v for k, v in meas.stats.items()})
        if lo_raw is not None and self.log_lo_pose:
            rec.update(lo_raw)
        self.log.append(rec)
        self._notify('on_step', dict(t=t, scan=scan, pred=pred, meas=meas, pre=pre, rec=rec))

    def _pred_cov(self, xi, pre):
        """6x6 covariance of the predicted [rot, pos] of the new node (right perturbation, world position)."""
        from .state import POS, ROT, VEL
        Ci = self.graph.marginal_cov(self.graph.newest_key)
        dt = pre.dt
        F = np.zeros((6, 15))                                  # d[θj, pj] / d[θi, pi, vi, ...]
        F[0:3, ROT] = pre.dR.T
        F[3:6, ROT] = -xi.R @ so3.hat(pre.dP)
        F[3:6, POS] = np.eye(3)
        F[3:6, VEL] = np.eye(3) * dt
        G = np.zeros((6, 9))                                   # d[θj, pj] / d[preintegration noise (dR, dV, dP)]
        G[0:3, 0:3] = np.eye(3)
        G[3:6, 6:9] = xi.R
        F = F[:, :Ci.shape[0]]                                 # no bias columns when biases are not estimated
        return F @ Ci @ F.T + G @ (pre.cov * self.io.cov_scale) @ G.T

    def _mahalanobis(self, xi, pre, xj, meas):
        r = np.r_[so3.log(meas.R.T @ xj.R), xj.p - meas.p]
        Cm = np.linalg.inv(meas.info + 1e-12 * np.eye(6))
        S = self._pred_cov(xi, pre) + Cm + self.pred_floor
        return float(r @ np.linalg.solve(S, r))

    def _recover(self, t, scan, xi, pred, meas, pre, tic):
        """Re-initialise the window at the latest LO pose; velocity from the rejected LO run."""
        T_ = np.array([x[0] for x in self.rejected])
        P_ = np.array([x[2] for x in self.rejected])
        v = np.polyfit(T_ - T_[-1], P_, 1)[0] if len(T_) >= 2 else np.zeros(3)
        x = NodeState(t, meas.R, meas.p, v, xi.bg.copy(), xi.ba.copy())
        key = self.graph.newest_key + 1
        for s_ in self.graph.states.values():                  # flush the old window into the smoothed track
            self.smoothed.append(tum_row(s_))
        self.graph = SlidingWindowGraph(self.graph.size, self.graph.iters, self.graph.lm_lambda0, nd=self.nd)
        self.graph.add_node(key, x)
        self.graph.add_factor(PriorFactor(key, x, self._sigmas(self.recover_sigmas)))
        self.graph.optimize()
        self.scans = {}
        self.rejected = []
        self.n_reinit += 1
        self._insert(scan, x.T, t)
        self.online.append(tum_row(x))
        meas.stats['recovered'] = True
        rec = dict(t=t, cost=0.0, dt_ms=1e3 * (time.perf_counter() - tic), lo=False, reinit=True)
        rec.update({f'lo_{k}': v_ for k, v_ in meas.stats.items()})
        self.log.append(rec)
        self._notify('on_step', dict(t=t, scan=scan, pred=pred, meas=meas, pre=pre, rec=rec))

    def finish(self):
        for s in self.graph.states.values():
            self.smoothed.append(tum_row(s))
        self._notify('on_finish')

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(os.path.join(out_dir, 'online.txt'), np.array(self.online), fmt='%.9f')
        np.savetxt(os.path.join(out_dir, 'smoothed.txt'), np.array(self.smoothed), fmt='%.9f')
        with open(os.path.join(out_dir, 'steps.jsonl'), 'w') as f:
            for r in self.log:
                f.write(json.dumps(r) + '\n')


def build_pipeline(cfg, seq, device=None):
    return SlidingWindowPGO(cfg, IOModule(seq, cfg['io'], device), LOModule(cfg['lo']))


def static_init(seq, t0, window_s=1.0):
    """GT-free initial state: roll / pitch from the mean specific force over the window_s before t0 (the robot is
    taken to be at rest), yaw 0, position 0, velocity 0. The world frame is then gravity-aligned with an arbitrary
    heading and origin; metrics.evaluate aligns the first pose's yaw and position to the GT."""
    from scipy.spatial.transform import Rotation
    m = (seq.imu_t >= t0 - window_s) & (seq.imu_t < t0)
    f = seq.acc[m].mean(0) if m.any() else seq.acc[:max(1, int(window_s * seq.imu_rate))].mean(0)
    R0 = Rotation.align_vectors([[0.0, 0.0, 1.0]], [f / np.linalg.norm(f)])[0].as_matrix()   # R0 f_hat = up
    return R0, np.zeros(3), np.zeros(3)


def run_sequence(cfg, seq, out_dir=None, device=None, progress=True, observers=(), hook=None):
    """Run the PGO over a sequence; initial state from GT at the first scan used.
    hook(pgo, t): called before every step with the next scan's stamp (e.g. the warm-start hand-over)."""
    pgo = build_pipeline(cfg, seq, device)
    pgo.observers.extend(observers)
    of = cfg['lo'].get('outage_file')        # {seq name: [[t0, t1], ...]} absolute stamps (tools/make_outages.py)
    if of:
        import json
        pgo.outages = np.array(json.load(open(of)).get(seq.name, []), float).reshape(-1, 2)
    r = cfg.get('run', {})
    first = int(r.get('start_scan', 0))
    # first scan with IMU history behind it and inside the GT span
    while first < len(seq) and (seq.scan_t[first] < seq.imu_t[0] + 0.1
                                or (seq.has_gt and not seq.in_gt_span(seq.scan_t[first], 0.05))):
        first += 1
    last = len(seq) if not r.get('max_scans') else min(len(seq), first + int(r['max_scans']))
    t0 = seq.scan_t[first]
    if r.get('init', 'gt') == 'static':
        R0, p0, v0 = static_init(seq, t0, float(r.get('init_window_s', 1.0)))
    else:                                    # GT state at the first scan (not available online)
        R0, p0, v0 = (x[0] for x in seq.gt_states(t0))
    pgo.start(t0, seq.scan(first), R0, p0, v0)
    tic = time.time()
    for k in range(first + 1, last):
        if seq.has_gt and not seq.in_gt_span(seq.scan_t[k]):
            break
        if hook is not None:
            hook(pgo, seq.scan_t[k])
        pgo.step(seq.scan_t[k], seq.scan(k))
        if progress and (k - first) % 200 == 0:
            n_lo = sum(x['lo'] for x in pgo.log[-200:])
            print(f'  scan {k - first}/{last - first - 1}  {time.time() - tic:6.0f}s  LO accepted {n_lo}/{len(pgo.log[-200:])}',
                  flush=True)
    pgo.finish()
    if out_dir:
        pgo.save(out_dir)
    return pgo
