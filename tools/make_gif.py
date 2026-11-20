"""Pipeline animation of one online KISS-IMU run, in three acts:

    1. Data selection  - the first warmup_s seconds on the raw IMU: the map and trajectory grow, the IMU stream
                         is collected (LO pseudo-labels)
    2. Training        - motion features -> GMM regimes -> balanced share per regime -> training loss
    3. Learned IO      - the PGO continues with the corrected IMU, the map is built to the end

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
from matplotlib.patches import Ellipse  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from kiss_imu.config import load_config  # noqa: E402
from kiss_imu.data import Sequence  # noqa: E402
from kiss_imu.training.reweight import MotionGMM, features_fn  # noqa: E402

BG, FG, MUTED, GRID = '#0d1117', '#e6edf3', '#8b949e', '#30363d'
C_SEL, C_LEARN, C_MAP_SEL, C_MAP = '#f2a33a', '#3fb8af', '#8a6a3a', '#5b6878'
PAL = ['#3fb8af', '#f2a33a', '#e0607e', '#8f7ce8', '#6cc24a', '#4a90d9']


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('seq'); ap.add_argument('run')
    ap.add_argument('--train', help='dir with history.json (default: the run\'s last train* dir)')
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
    t, P, Rq = on[:, 0], on[:, 1:4], Rotation.from_quat(on[:, 4:8])
    t0 = t[0]
    t_sel = t0 + float(w['warmup_s'])
    # level the map: the static init already aligns gravity with z; centre on the start
    P = P - P[0]

    # map points per scan (every map_every-th), with the scan time
    k_scan = np.searchsorted(seq.scan_t, t).clip(0, len(seq.scan_t) - 1)
    rng = np.random.default_rng(0)
    mp, mt = [], []
    for i in range(0, len(t), a.map_every):
        x = seq.scan(k_scan[i])
        x = x[rng.choice(len(x), min(a.points, len(x)), replace=False)]
        x = x[x[:, 2] < a.zmax]
        mp.append(Rq[i].apply(x) + P[i]); mt.append(np.full(len(x), t[i]))
    mp, mt = np.concatenate(mp), np.concatenate(mt)

    # warm-up IMU, motion features, GMM, balancing
    m = (seq.imu_t >= t0) & (seq.imu_t <= t_sel)
    ti, gn = seq.imu_t[m] - t0, np.linalg.norm(seq.gyr[m], axis=1)
    F = features_fn(rw)(seq.imu_t, seq.gyr, seq.acc, rw.get('win_s', 0.2))[m]
    gm = MotionGMM(rw.get('n_components'), tuple(rw.get('k_range', (2, 5))), rw.get('win_s', 0.2),
                   rw.get('method', 'effective'), rw.get('beta', 0.999), tuple(rw.get('clamp', (0.1, 10.0))), seed=0).fit([F])
    mu = gm.scaler.inverse_transform(gm.gmm.means_)
    order = np.argsort(mu[:, 1] + 0.05 * mu[:, 0])
    rank = np.empty_like(order); rank[order] = np.arange(gm.K)
    comp = rank[gm.proba(F).argmax(1)]
    counts = np.bincount(comp, minlength=gm.K).astype(float)
    cw = np.clip(gm.component_weights(counts), *gm.clamp)
    s_raw, s_bal = counts / counts.sum(), counts * cw / (counts * cw).sum()
    tdir = a.train
    if tdir is None:
        tdirs = sorted(x for x in os.listdir(a.run) if x.startswith('train'))
        tdir = os.path.join(a.run, tdirs[-1]) if tdirs else None
    hist = json.load(open(os.path.join(tdir, 'history.json'))) if tdir and os.path.exists(os.path.join(tdir, 'history.json')) else []
    L = np.array([h['train_loss'] for h in hist]) if hist else np.zeros(0)
    sub = rng.choice(len(F), min(2500, len(F)), replace=False)

    # ---- figure
    fig = plt.figure(figsize=(12, 5.2), dpi=90)
    fig.patch.set_facecolor(BG)
    axM = fig.add_axes([0.01, 0.03, 0.63, 0.80]); axM.set_facecolor(BG); axM.axis('off')
    lo, hi = np.r_[P[:, :2], mp[:, :2]].min(0), np.r_[P[:, :2], mp[:, :2]].max(0)
    lo, hi = np.percentile(mp[:, :2], 0.5, axis=0), np.percentile(mp[:, :2], 99.5, axis=0)
    axM.set_xlim(lo[0], hi[0]); axM.set_ylim(lo[1], hi[1]); axM.set_aspect('equal')
    sc = axM.scatter(np.zeros(0), np.zeros(0), s=0.25, linewidths=0)
    ln_sel, = axM.plot([], [], color=C_SEL, lw=2.4, solid_capstyle='round')
    ln_lrn, = axM.plot([], [], color=C_LEARN, lw=2.4, solid_capstyle='round')
    dot, = axM.plot([], [], 'o', color=FG, ms=6, mec=BG, mew=1.5)
    axR = fig.add_axes([0.70, 0.17, 0.27, 0.56]); style(axR)
    step = [fig.text(0.04 + 0.30 * k, 0.905, s, fontsize=12.5, fontweight='bold')
            for k, s in enumerate(['①  Data selection', '②  Self-supervised training', '③  Learned IO → online PGO'])]
    cap = fig.text(0.04, 0.855, '', color=MUTED, fontsize=9.5)
    rtitle = fig.text(0.70, 0.76, '', color=FG, fontsize=10, fontweight='bold')
    rsub = fig.text(0.70, 0.08, '', color=MUTED, fontsize=8.5)

    col_sel, col_map = np.array(to_rgba(C_MAP_SEL)), np.array(to_rgba(C_MAP))
    mcol = np.where((mt < t_sel)[:, None], col_sel, col_map)

    def set_step(k):
        for i, s in enumerate(step):
            s.set_color(FG if i == k else '#484f58')

    def show_map(tc):
        k = mt <= tc
        sc.set_offsets(mp[k, :2]); sc.set_color(mcol[k])
        n = int(np.searchsorted(t, tc, side='right'))
        s = t[:n] < t_sel
        ln_sel.set_data(P[:n][s, 0], P[:n][s, 1]); ln_lrn.set_data(P[:n][~s, 0], P[:n][~s, 1])
        dot.set_data([P[max(n - 1, 0), 0]], [P[max(n - 1, 0), 1]])

    N1, N2a, N2b, N2c, N3, HOLD = 45, 22, 26, 22, 120, 15
    total = N1 + N2a + N2b + N2c + N3 + HOLD
    state = {'mode': None}

    def reset_right(mode):
        if state['mode'] == mode:
            return
        axR.cla(); style(axR); state['mode'] = mode

    def frame(f):
        if f < N1:                                                       # 1. data selection
            q = (f + 1) / N1
            tc = t0 + q * (t_sel - t0)
            set_step(0); show_map(tc)
            cap.set_text(f'first {t_sel - t0:.0f} s on the raw IMU: LiDAR odometry poses become the pseudo-labels   (t = {tc - t0:4.1f} s)')
            reset_right('imu')
            if not hasattr(axR, '_ln'):
                axR._ln, = axR.plot([], [], color=C_SEL, lw=0.8)
                axR.set_xlim(0, t_sel - t0); axR.set_ylim(0, np.percentile(gn, 99.5) * 1.1)
                axR.set_xlabel('t [s]', color=MUTED, fontsize=8); axR.set_ylabel('|gyr| [rad/s]', color=MUTED, fontsize=8)
            k = ti <= tc - t0
            axR._ln.set_data(ti[k], gn[k])
            rtitle.set_text('raw IMU stream, collected'); rsub.set_text('')
        elif f < N1 + N2a:                                               # 2a. GMM regimes
            q = ease((f - N1 + 1) / N2a)
            set_step(1); show_map(t_sel)
            cap.set_text('the denoiser is trained on the selected 30 s only: no GT, no other data')
            reset_right('gmm')
            if not hasattr(axR, '_sc'):
                axR._sc = axR.scatter(F[sub, 0], F[sub, 1], s=2, linewidths=0)
                axR.set_xlim(0, np.percentile(F[:, 0], 99.5) * 1.1); axR.set_ylim(0, np.percentile(F[:, 1], 99.5) * 1.1)
                axR.set_xlabel('|gravity-free acc| [m/s²]', color=MUTED, fontsize=8); axR.set_ylabel('|gyr| [rad/s]', color=MUTED, fontsize=8)
                axR._e = []
                for k in range(gm.K):
                    j = order[k]
                    C = gm.scaler.scale_[:, None] * gm.gmm.covariances_[j] * gm.scaler.scale_[None]
                    ev, V = np.linalg.eigh(C)
                    e = Ellipse(mu[j], 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0]), angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])),
                                fill=False, lw=1.6, color=PAL[k % len(PAL)])
                    axR.add_patch(e); axR._e.append(e)
            base = np.array(to_rgba(MUTED)); cols = np.array([to_rgba(PAL[c % len(PAL)]) for c in comp[sub]])
            axR._sc.set_color((1 - q) * base + q * cols)
            for e in axR._e:
                e.set_alpha(q)
            rtitle.set_text(f'motion regimes: GMM, K = {gm.K} (BIC)'); rsub.set_text('motion features of the warm-up')
        elif f < N1 + N2a + N2b:                                         # 2b. balancing
            q = ease((f - N1 - N2a + 1 - 5) / (N2b - 10))
            set_step(1); show_map(t_sel)
            reset_right('bars')
            x = np.arange(gm.K)
            if not hasattr(axR, '_b'):
                axR.bar(x, s_raw, color='none', edgecolor=MUTED, ls='--', lw=1.0, width=0.62)
                axR._b = axR.bar(x, s_raw, color=[PAL[k % len(PAL)] for k in range(gm.K)], width=0.62)
                axR.set_xticks(x); axR.set_xticklabels([f'M{k + 1}' for k in range(gm.K)], color=MUTED)
                axR.set_ylim(0, max(s_raw.max(), s_bal.max()) * 1.15); axR.set_ylabel('share of training signal', color=MUTED, fontsize=8)
            for b, s in zip(axR._b, (1 - q) * s_raw + q * s_bal):
                b.set_height(s)
            rtitle.set_text('motion-balanced reweighting'); rsub.set_text('dashed: raw share · rare motion is no longer drowned out')
        elif f < N1 + N2a + N2b + N2c:                                   # 2c. training loss
            q = ease((f - N1 - N2a - N2b + 1) / N2c)
            set_step(1); show_map(t_sel)
            reset_right('loss')
            if len(L):
                if not hasattr(axR, '_l'):
                    axR._l, = axR.plot([], [], color=FG, lw=1.8)
                    axR.set_xlim(0, len(L)); axR.set_ylim(np.nanmin(L) * 0.9, np.nanpercentile(L, 99) * 1.05)
                    axR.set_xlabel('epoch', color=MUTED, fontsize=8); axR.set_ylabel('loss', color=MUTED, fontsize=8)
                n = max(1, int(q * len(L)))
                axR._l.set_data(np.arange(n), L[:n])
            rtitle.set_text('training the IMU denoiser'); rsub.set_text('multi-horizon loss on the LO pseudo-labels')
        else:                                                            # 3. learned IO
            g = min(f - N1 - N2a - N2b - N2c, N3 - 1)
            tc = t_sel + ease((g + 1) / N3) * (t[-1] - t_sel)
            set_step(2); show_map(tc)
            n = int(np.searchsorted(t, tc, side='right'))
            cap.set_text(f'the running PGO now integrates the corrected IMU and builds the map   (t = {tc - t0:5.1f} s)')
            reset_right('io')
            if not hasattr(axR, '_txt'):
                axR.axis('off')
                axR._txt = axR.text(0.0, 0.95, '', color=FG, fontsize=10, va='top', family='monospace', transform=axR.transAxes)
            axR._txt.set_text('raw IMU ─▶ [ learned IO ] ─▶ PGO\n\n'
                              f'  scans       {n:5d}\n  time        {tc - t0:5.0f} s\n\n'
                              '  LiDAR odometry + corrected\n  IMU in a sliding window')
            rtitle.set_text('learned IO in the loop'); rsub.set_text('')
        return []

    anim = FuncAnimation(fig, frame, frames=total, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=a.fps), savefig_kwargs={'facecolor': BG})
    print('saved', a.out)


if __name__ == '__main__':
    main()
