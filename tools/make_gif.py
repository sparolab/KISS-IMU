"""Pipeline animation of one online KISS-IMU run: a block diagram on top (IMU, LiDAR, the network, the PGO, the
30 s label buffer) whose data flow follows the run, and the map being built below.

    1. Data selection  - the IMU goes straight to the PGO (raw); LiDAR odometry poses of the first warmup_s
                         seconds fill the label buffer (coloured by GMM motion regime once training starts)
    2. Training        - the buffer flows into the network; the regime shares morph to the balanced ones
    3. Learned IO      - the IMU flows through the network into the PGO; the map is built to the end

Best with a run whose hand-over is at warmup_s (a released checkpoint replayed without training):

    python3 scripts/run_warmstart.py configs/diter_os/warmstart.yaml --seqs Forest_new --out runs/ws_pre \
        --set warmstart.adapt=false warmstart.pretrained=checkpoints/diter_os/Forest_new/kiss_imu_ws30.pt
    python3 tools/make_gif.py configs/diter_os/warmstart.yaml Forest_new runs/ws_pre/Forest_new \
        --train checkpoints/diter_os/Forest_new --out fig/kiss_imu_online.gif
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
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from kiss_imu.config import load_config  # noqa: E402
from kiss_imu.data import Sequence  # noqa: E402
from kiss_imu.training.reweight import MotionGMM, features_fn  # noqa: E402

BG, FG, MUTED, DIM, BOX = '#0d1117', '#e6edf3', '#8b949e', '#3a414b', '#161b22'
C_IMU, C_LID, C_SEL, C_NET = '#e0607e', '#4a90d9', '#f2a33a', '#3fb8af'
C_MAP_SEL, C_MAP = '#8a6a3a', '#5b6878'
PAL = ['#3fb8af', '#f2a33a', '#e0607e', '#8f7ce8', '#6cc24a', '#4a90d9']


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Path:
    """A polyline in figure coordinates with dots running along it."""

    def __init__(self, ax, pts, color, n=6, lw=1.6, ls='-'):
        self.p = np.asarray(pts, float)
        seg = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.cum = np.r_[0, np.cumsum(seg)] / seg.sum()
        self.color, self.n = color, n
        self.line, = ax.plot(self.p[:, 0], self.p[:, 1], color=color, lw=lw, ls=ls, alpha=0.25, solid_capstyle='round')
        self.head = ax.annotate('', xy=self.p[-1], xytext=self.p[-1] - 0.001 * (self.p[-1] - self.p[-2]) / max(seg[-1], 1e-9),
                                arrowprops=dict(arrowstyle='-|>', color=color, lw=0, mutation_scale=12, alpha=0.25))
        self.dots = ax.scatter(np.zeros(n), np.zeros(n), s=22, color=color, zorder=5, alpha=0)

    def at(self, u):
        return np.stack([np.interp(u, self.cum, self.p[:, k]) for k in (0, 1)], 1)

    def set(self, f, active, speed=0.03):
        self.line.set_alpha(0.85 if active else 0.18)
        self.head.arrow_patch.set_alpha(0.9 if active else 0.18)
        if active:
            self.dots.set_offsets(self.at((np.arange(self.n) / self.n + f * speed) % 1.0)); self.dots.set_alpha(1)
        else:
            self.dots.set_alpha(0)


def box(ax, xy, w, h, label, color, fs=10):
    b = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h, boxstyle='round,pad=0.004,rounding_size=0.012',
                       fc=BOX, ec=color, lw=1.6)
    ax.add_patch(b)
    t = ax.text(xy[0], xy[1], label, ha='center', va='center', color=FG, fontsize=fs, fontweight='bold')
    return b, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('seq'); ap.add_argument('run')
    ap.add_argument('--train', help="dir with history.json (default: the run's last train* dir)")
    ap.add_argument('--out', default='fig/kiss_imu_online.gif')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--map-every', type=int, default=3)
    ap.add_argument('--points', type=int, default=1200)
    ap.add_argument('--zmax', type=float, default=3.0)
    ap.add_argument('--set', nargs='*', default=[], dest='overrides')
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    d, w = cfg['data'], cfg['warmstart']
    tpath = w['train_config'] if os.path.exists(w['train_config']) else os.path.join(os.path.dirname(a.config), w['train_config'])
    rw = load_config(tpath).get('reweight', {})
    seq = Sequence(d['root'], a.seq, d.get('type', 'generic'), 0.0, min_range=d.get('min_range', 1.0),
                   max_range=d.get('max_range', 100.0), scan_time_offset=d.get('scan_time_offset', 0.0),
                   lidar_rotvec_deg=d.get('lidar_rotvec_deg'))
    on = np.loadtxt(os.path.join(a.run, 'online.txt'))
    t, P, Rq = on[:, 0], on[:, 1:4] - on[0, 1:4], Rotation.from_quat(on[:, 4:8])
    t0 = t[0]
    t_sel = t0 + float(w['warmup_s'])

    k_scan = np.searchsorted(seq.scan_t, t).clip(0, len(seq.scan_t) - 1)
    rng = np.random.default_rng(0)
    mp, mt = [], []
    for i in range(0, len(t), a.map_every):
        x = seq.scan(k_scan[i])
        x = x[rng.choice(len(x), min(a.points, len(x)), replace=False)]
        x = x[x[:, 2] < a.zmax]
        mp.append(Rq[i].apply(x) + P[i]); mt.append(np.full(len(x), t[i]))
    mp, mt = np.concatenate(mp), np.concatenate(mt)

    m = (seq.imu_t >= t0) & (seq.imu_t <= t_sel)
    F = features_fn(rw)(seq.imu_t, seq.gyr, seq.acc, rw.get('win_s', 0.2))[m]
    gm = MotionGMM(rw.get('n_components'), tuple(rw.get('k_range', (2, 5))), rw.get('win_s', 0.2),
                   rw.get('method', 'effective'), rw.get('beta', 0.999), tuple(rw.get('clamp', (0.1, 10.0))), seed=0).fit([F])
    mu = gm.scaler.inverse_transform(gm.gmm.means_)
    order = np.argsort(mu[:, 1] + 0.05 * mu[:, 0])
    rank = np.empty_like(order); rank[order] = np.arange(gm.K)
    counts = np.bincount(rank[gm.proba(F).argmax(1)], minlength=gm.K).astype(float)
    cw = np.clip(gm.component_weights(counts), *gm.clamp)
    s_raw, s_bal = counts / counts.sum(), counts * cw / (counts * cw).sum()
    tdir = a.train
    if tdir is None:
        tdirs = sorted(x for x in os.listdir(a.run) if x.startswith('train'))
        tdir = os.path.join(a.run, tdirs[-1]) if tdirs else None
    hp = os.path.join(tdir, 'history.json') if tdir else ''
    L = np.array([h['train_loss'] for h in json.load(open(hp))]) if os.path.exists(hp) else np.zeros(0)

    # ---- figure: diagram on top, map below
    fig = plt.figure(figsize=(12, 7.2), dpi=85)
    fig.patch.set_facecolor(BG)
    axD = fig.add_axes([0, 0, 1, 1]); axD.set_xlim(0, 1); axD.set_ylim(0, 1); axD.axis('off'); axD.set_zorder(2)
    axD.patch.set_alpha(0)
    axM = fig.add_axes([0.02, 0.02, 0.96, 0.55]); axM.set_facecolor(BG); axM.axis('off'); axM.set_zorder(1)
    lo, hi = np.percentile(mp[:, :2], 0.5, axis=0), np.percentile(mp[:, :2], 99.5, axis=0)
    axM.set_xlim(lo[0], hi[0]); axM.set_ylim(lo[1], hi[1]); axM.set_aspect('equal')
    sc = axM.scatter(np.zeros(0), np.zeros(0), s=0.25, linewidths=0)
    ln_sel, = axM.plot([], [], color=C_SEL, lw=2.4, solid_capstyle='round')
    ln_lrn, = axM.plot([], [], color=C_NET, lw=2.4, solid_capstyle='round')
    dot, = axM.plot([], [], 'o', color=FG, ms=6, mec=BG, mew=1.5)

    # blocks
    IMU, LID, NET, PGO, BUF = (0.08, 0.875), (0.08, 0.66), (0.38, 0.84), (0.66, 0.84), (0.38, 0.665)
    box(axD, IMU, 0.10, 0.07, 'IMU', C_IMU)
    box(axD, LID, 0.10, 0.07, 'LiDAR', C_LID)
    b_pgo, _ = box(axD, PGO, 0.12, 0.10, 'PGO', FG, fs=12)
    axD.text(PGO[0], PGO[1] - 0.075, 'sliding window → map ↓', ha='center', color=MUTED, fontsize=8)
    # network: rounded box with a small layered net inside
    b_net = FancyBboxPatch((NET[0] - 0.09, NET[1] - 0.085), 0.18, 0.17, boxstyle='round,pad=0.004,rounding_size=0.015',
                           fc=BOX, ec=C_NET, lw=2)
    axD.add_patch(b_net)
    axD.text(NET[0], NET[1] + 0.098, 'KISS-IMU network', ha='center', color=FG, fontsize=10, fontweight='bold')
    layers = [3, 5, 5, 3]
    xs = np.linspace(NET[0] - 0.065, NET[0] + 0.065, len(layers))
    nodes = [np.stack([np.full(n, x), NET[1] + np.linspace(-0.055, 0.055, n) * (n / 5)], 1) for x, n in zip(xs, layers)]
    edges = []
    for A, B in zip(nodes[:-1], nodes[1:]):
        for p in A:
            for q in B:
                edges.append(axD.plot([p[0], q[0]], [p[1], q[1]], color=C_NET, lw=0.6, alpha=0.15)[0])
    node_sc = axD.scatter(np.concatenate(nodes)[:, 0], np.concatenate(nodes)[:, 1], s=38, color=DIM, zorder=4,
                          edgecolors=C_NET, linewidths=0.8)
    loss_txt = axD.text(NET[0], NET[1] - 0.105, '', ha='center', color=MUTED, fontsize=8.5)
    # label buffer: frame + stacked bar of GMM regime shares
    bw, bh = 0.18, 0.045
    axD.add_patch(FancyBboxPatch((BUF[0] - bw / 2, BUF[1] - bh / 2), bw, bh, boxstyle='round,pad=0.003,rounding_size=0.008',
                                 fc=BOX, ec=C_SEL, lw=1.6))
    buf_lbl = axD.text(BUF[0], BUF[1] - 0.045, '', ha='center', color=MUTED, fontsize=8.5)
    axD.text(BUF[0] - bw / 2 - 0.01, BUF[1], '30 s\nlabels', ha='right', va='center', color=C_SEL, fontsize=8.5, fontweight='bold')
    segs = [Rectangle((BUF[0], BUF[1] - bh / 2 + 0.006), 0, bh - 0.012, fc=PAL[k % len(PAL)], lw=0) for k in range(gm.K)]
    fill = Rectangle((BUF[0] - bw / 2 + 0.005, BUF[1] - bh / 2 + 0.006), 0, bh - 0.012, fc=MUTED, alpha=0.6, lw=0)
    for r in segs + [fill]:
        axD.add_patch(r)

    paths = {
        'imu_raw': Path(axD, [(0.13, 0.875), (0.20, 0.875), (0.20, 0.965), (0.64, 0.965), (0.64, 0.89)], C_IMU, n=7, ls=(0, (3, 2))),
        'imu_net': Path(axD, [(0.13, 0.86), (0.29, 0.86)], C_IMU, n=4),
        'net_pgo': Path(axD, [(0.47, 0.84), (0.60, 0.84)], C_NET, n=4),
        'lidar': Path(axD, [(0.13, 0.62), (0.69, 0.62), (0.69, 0.79)], C_LID, n=8),
        'pgo_buf': Path(axD, [(0.64, 0.79), (0.64, 0.665), (0.47, 0.665)], C_SEL, n=5),
        'buf_net': Path(axD, [(0.38, 0.688), (0.38, 0.755)], C_SEL, n=3),
    }
    axD.text(0.42, 0.975, 'raw IMU', color=C_IMU, fontsize=8, ha='center', va='bottom')
    axD.text(0.535, 0.852, 'learned IO', color=C_NET, fontsize=8.5, ha='center', va='bottom', fontweight='bold')
    axD.text(0.555, 0.672, 'LO poses', color=C_SEL, fontsize=8, ha='center', va='bottom')

    steps = [axD.text(0.78, 0.93 - 0.075 * k, s, color=DIM, fontsize=12.5, fontweight='bold')
             for k, s in enumerate(['①  Data selection', '②  Training', '③  Learned IO → PGO'])]
    cap = axD.text(0.78, 0.66, '', color=MUTED, fontsize=9, va='top', linespacing=1.4)
    clock = axD.text(0.78, 0.705, '', color=FG, fontsize=10.5, family='monospace')

    col_sel, col_map = np.array(to_rgba(C_MAP_SEL)), np.array(to_rgba(C_MAP))
    mcol = np.where((mt < t_sel)[:, None], col_sel, col_map)

    def show_map(tc):
        k = mt <= tc
        sc.set_offsets(mp[k, :2]); sc.set_color(mcol[k])
        n = int(np.searchsorted(t, tc, side='right'))
        s = t[:n] < t_sel
        ln_sel.set_data(P[:n][s, 0], P[:n][s, 1]); ln_lrn.set_data(P[:n][~s, 0], P[:n][~s, 1])
        dot.set_data([P[max(n - 1, 0), 0]], [P[max(n - 1, 0), 1]])

    def set_buffer(level, shares):
        x0, W = BUF[0] - bw / 2 + 0.005, (bw - 0.01)
        if shares is None:
            fill.set_width(W * level)
            for r in segs:
                r.set_width(0)
        else:
            fill.set_width(0)
            x = x0
            for r, s in zip(segs, shares):
                r.set_x(x); r.set_width(W * s); x += W * s

    def net_glow(f, on):
        if on:
            ph = np.concatenate([np.full(n, i) for i, n in enumerate(layers)])
            b = 0.5 + 0.5 * np.sin(0.6 * f - 1.2 * ph)
            c = np.array(to_rgba(C_NET)); dim = np.array(to_rgba(DIM))
            node_sc.set_color(dim[None] * (1 - b[:, None]) + c[None] * b[:, None])
            for e in edges:
                e.set_alpha(0.35)
        else:
            node_sc.set_color(DIM)
            for e in edges:
                e.set_alpha(0.12)

    def set_step(k):
        for i, s in enumerate(steps):
            s.set_color(FG if i == k else DIM)

    N1, N2, N3, HOLD = 50, 70, 120, 15

    def frame(f):
        if f < N1:
            q = (f + 1) / N1
            tc = t0 + q * (t_sel - t0)
            set_step(0); show_map(tc); net_glow(f, False)
            for k, p in paths.items():
                p.set(f, k in ('imu_raw', 'lidar', 'pgo_buf'))
            set_buffer(q, None)
            buf_lbl.set_text(f'collecting  {tc - t0:4.1f} / {t_sel - t0:.0f} s')
            loss_txt.set_text('waiting for data')
            cap.set_text('the robot runs on the raw IMU;\nLiDAR odometry poses of the\nfirst 30 s become pseudo-labels')
            clock.set_text(f't = {tc - t0:5.1f} s')
        elif f < N1 + N2:
            g = f - N1
            set_step(1); show_map(t_sel); net_glow(f, True)
            for k, p in paths.items():
                p.set(f, k == 'buf_net', speed=0.05)
            q = ease((g - 15) / 30)
            set_buffer(1, (1 - q) * s_raw + q * s_bal)
            buf_lbl.set_text('GMM motion regimes: raw share' if q < 0.05 else 'motion-balanced share')
            if len(L):
                e = int(ease(g / (N2 - 5)) * (len(L) - 1))
                loss_txt.set_text(f'training  epoch {e + 1:2d}/{len(L)}   loss {L[e]:.2f}')
            cap.set_text('the network learns from the 30 s\nonly (no GT); rare motion regimes\nare up-weighted (GMM balancing)')
            clock.set_text(f't = {t_sel - t0:5.1f} s')
        else:
            g = min(f - N1 - N2, N3 - 1)
            tc = t_sel + ease((g + 1) / N3) * (t[-1] - t_sel)
            set_step(2); show_map(tc); net_glow(f, True)
            for k, p in paths.items():
                p.set(f, k in ('imu_net', 'net_pgo', 'lidar'))
            set_buffer(1, s_bal)
            buf_lbl.set_text('motion-balanced share')
            loss_txt.set_text('corrected IMU, causal')
            cap.set_text('every IMU sample now passes the\nnetwork; the running PGO uses\nthe learned IO and builds the map')
            clock.set_text(f't = {tc - t0:5.1f} s')
        return []

    anim = FuncAnimation(fig, frame, frames=N1 + N2 + N3 + HOLD, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=a.fps), savefig_kwargs={'facecolor': BG})
    print('saved', a.out)


if __name__ == '__main__':
    main()
