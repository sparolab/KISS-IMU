"""Top-down animation of a warm-start run: the map built from the scans at the online poses, the trajectory
growing (warm-up on the raw IMU, then the online PGO on the corrected IMU), the raw-IMU baseline and GT.

    python3 tools/make_gif.py configs/diter_os/warmstart.yaml Forest_new runs/warmstart/Forest_new \
        --raw runs/raw/Forest_new --out fig/forest_new.gif
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from kiss_imu.config import load_config  # noqa: E402
from kiss_imu.data import Sequence  # noqa: E402
from kiss_imu.eval.metrics import umeyama_se3  # noqa: E402

BG, FG, MUTED = '#0f1115', '#e8e8ea', '#8a8f98'
C_WARM, C_ONLINE, C_RAW, C_GT, C_MAP = '#f2a33a', '#3fb8af', '#e0607e', '#c9ccd1', '#5b6270'


def load_tum(p):
    a = np.loadtxt(p)
    return a[:, 0], a[:, 1:4], Rotation.from_quat(a[:, 4:8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('seq'); ap.add_argument('run')
    ap.add_argument('--raw', help='raw-IMU PGO run dir of the same sequence (optional)')
    ap.add_argument('--out', default='fig/warmstart.gif')
    ap.add_argument('--frames', type=int, default=160)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--map-every', type=int, default=3, help='scans skipped between map scans')
    ap.add_argument('--points', type=int, default=1500, help='points kept per map scan')
    ap.add_argument('--zmax', type=float, default=3.0, help='map points above the sensor by more than this are dropped')
    ap.add_argument('--set', nargs='*', default=[], dest='overrides')
    a = ap.parse_args()
    d = load_config(a.config, a.overrides)['data']
    seq = Sequence(d['root'], a.seq, d.get('type', 'generic'), (d.get('gt_time_offset') or {}).get(a.seq, 0.0),
                   min_range=d.get('min_range', 1.0), max_range=d.get('max_range', 100.0), gt_file=d.get('gt_file', 'gt_pose.csv'),
                   scan_time_offset=d.get('scan_time_offset', 0.0), lidar_rotvec_deg=d.get('lidar_rotvec_deg'))
    t, p, R = load_tum(os.path.join(a.run, 'online.txt'))
    ws = json.load(open(os.path.join(a.run, 'result.json'))).get('warmstart', {})
    t_sw = t[0] + ws.get('t_switch', np.inf)
    # everything in the GT frame (one SE(3) alignment of the online trajectory), so GT / raw / map overlay
    ok = seq.in_gt_span(t)
    gR, gp, _ = seq.gt_states(t[ok])
    Ra, ta = umeyama_se3(p[ok], gp)
    P = p @ Ra.T + ta
    Rw = Rotation.from_matrix(Ra) * R
    raw = None
    if a.raw:
        tr, pr, _ = load_tum(os.path.join(a.raw, 'online.txt'))
        okr = seq.in_gt_span(tr)
        Rr, trr = umeyama_se3(pr[okr], seq.gt_states(tr[okr])[1])
        raw = (tr, pr @ Rr.T + trr)
    gt = seq.gt_states(seq.gt_t)[1]

    # map: every map_every-th scan at its online pose, sub-sampled, ground-ish band only
    k_scan = np.searchsorted(seq.scan_t, t).clip(0, len(seq.scan_t) - 1)
    rng = np.random.default_rng(0)
    mp, mt = [], []
    for i in range(0, len(t), a.map_every):
        x = seq.scan(k_scan[i])[:, :3]
        x = x[rng.choice(len(x), min(a.points, len(x)), replace=False)]
        x = x[x[:, 2] < a.zmax]
        mp.append(Rw[i].apply(x) + P[i]); mt.append(np.full(len(x), t[i]))
    mp, mt = np.concatenate(mp), np.concatenate(mt)

    allxy = np.vstack([P[:, :2], gt[:, :2]])
    lo, hi = allxy.min(0), allxy.max(0)
    pad = 0.08 * (hi - lo).max() + 2.0
    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad); ax.set_aspect('equal')
    ax.axis('off')
    sc = ax.scatter([], [], s=0.15, c=C_MAP, alpha=0.5, linewidths=0)
    ax.plot(gt[:, 0], gt[:, 1], color=C_GT, lw=1.0, ls=(0, (2, 2)), alpha=0.6, label='GT')
    l_raw, = ax.plot([], [], color=C_RAW, lw=1.2, alpha=0.8, label='raw IMU + PGO')
    l_w, = ax.plot([], [], color=C_WARM, lw=2.2, label='warm-up (raw IMU, training)')
    l_o, = ax.plot([], [], color=C_ONLINE, lw=2.2, label='online PGO (corrected IMU)')
    dot, = ax.plot([], [], 'o', color=FG, ms=5)
    leg = ax.legend(loc='lower left', frameon=False, fontsize=8, labelcolor=FG)
    if raw is None:
        l_raw.set_label('_'); leg = ax.legend(loc='lower left', frameon=False, fontsize=8, labelcolor=FG)
    title = ax.text(0.02, 0.97, '', transform=ax.transAxes, color=FG, fontsize=12, fontweight='bold', va='top')
    sub = ax.text(0.02, 0.91, '', transform=ax.transAxes, color=MUTED, fontsize=9, va='top')
    stamps = np.linspace(t[0] + 0.5, t[-1], a.frames)
    stamps = np.r_[stamps, np.full(a.fps, t[-1])]                 # hold the last frame ~1 s

    def draw(f):
        tc = stamps[f]
        n = int(np.searchsorted(t, tc, side='right'))
        sc.set_offsets(mp[mt <= tc][:, :2])
        w = t[:n] < t_sw
        l_w.set_data(P[:n][w, 0], P[:n][w, 1])
        l_o.set_data(P[:n][~w, 0], P[:n][~w, 1])
        if raw is not None:
            m = raw[0] <= tc
            l_raw.set_data(raw[1][m, 0], raw[1][m, 1])
        dot.set_data([P[n - 1, 0]], [P[n - 1, 1]])
        phase = 'Warm-up: collecting LO pseudo-labels, training' if tc < t_sw else 'Online PGO with the self-trained IMU'
        title.set_text(phase)
        sub.set_text(f'{a.seq}   t = {tc - t[0]:5.1f} s   (hand-over at {t_sw - t[0]:.0f} s)')
        return sc, l_w, l_o, l_raw, dot, title, sub

    anim = FuncAnimation(fig, draw, frames=len(stamps), blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=a.fps), savefig_kwargs={'facecolor': BG})
    print('saved', a.out)


if __name__ == '__main__':
    main()
