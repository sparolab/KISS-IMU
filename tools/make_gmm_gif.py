"""Animated explainer of KISS-IMU's motion-balanced learning on a run's warm-up (GT-free, as in training):

    1. motion features (|gravity-free acc|, |gyr| over 0.2 s) of the warm-up IMU stream appear over time
    2. the GMM (K by BIC) splits them into motion regimes
    3. per-regime share of the label pairs, raw -> after the effective-number weights (histogram morph)
    4. the balanced regimes drive the denoiser training (training-loss curve of the warm-start run)

    python3 tools/make_gmm_gif.py configs/diter_os/warmstart.yaml Forest_new runs/warmstart/Forest_new \
        --out fig/gmm_balance.gif
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
from matplotlib.patches import Ellipse  # noqa: E402

from kiss_imu.config import load_config  # noqa: E402
from kiss_imu.data import Sequence  # noqa: E402
from kiss_imu.training.reweight import MotionGMM, features_fn  # noqa: E402

BG, FG, MUTED, GRID = '#0f1115', '#e8e8ea', '#8a8f98', '#2a2e36'
PAL = ['#3fb8af', '#f2a33a', '#e0607e', '#8f7ce8', '#6cc24a', '#4a90d9', '#d9c84a', '#c9ccd1']


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('seq'); ap.add_argument('run')
    ap.add_argument('--out', default='fig/gmm_balance.gif')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--set', nargs='*', default=[], dest='overrides')
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    d, w = cfg['data'], cfg['warmstart']
    tpath = w['train_config'] if os.path.exists(w['train_config']) else os.path.join(os.path.dirname(a.config), w['train_config'])
    rw = load_config(tpath).get('reweight', {})
    seq = Sequence(d['root'], a.seq, d.get('type', 'generic'), 0.0, gt_file=d.get('gt_file', 'gt_pose.csv'))
    t0 = float(np.loadtxt(os.path.join(a.run, 'online.txt'), max_rows=1)[0])
    res = json.load(open(os.path.join(a.run, 'result.json')))['warmstart']
    t1 = t0 + res['t_train']
    m = (seq.imu_t >= t0) & (seq.imu_t <= t1)
    F = features_fn(rw)(seq.imu_t, seq.gyr, seq.acc, rw.get('win_s', 0.2))[m]
    ts = seq.imu_t[m] - t0
    gm = MotionGMM(rw.get('n_components'), tuple(rw.get('k_range', (2, 5))), rw.get('win_s', 0.2),
                   rw.get('method', 'effective'), rw.get('beta', 0.999), tuple(rw.get('clamp', (0.1, 10.0))), seed=0).fit([F])
    P = gm.proba(F)
    comp = P.argmax(1)
    # order the regimes by motion intensity (|gyr| mean) so colours read calm -> aggressive
    mu = gm.scaler.inverse_transform(gm.gmm.means_)
    order = np.argsort(mu[:, 1] + 0.05 * mu[:, 0])
    rank = np.empty_like(order); rank[order] = np.arange(gm.K)
    comp = rank[comp]
    counts = np.bincount(comp, minlength=gm.K).astype(float)
    cw = np.clip(gm.component_weights(counts), *gm.clamp)
    share_raw = counts / counts.sum()
    share_bal = counts * cw / (counts * cw).sum()
    tdirs = sorted(x for x in os.listdir(a.run) if x.startswith('train'))
    hp = os.path.join(a.run, tdirs[-1], 'history.json') if tdirs else ''
    hist = json.load(open(hp)) if os.path.exists(hp) else None
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(F), min(4000, len(F)), replace=False))
    Fs, cs, tss = F[sub], comp[sub], ts[sub]

    fig = plt.figure(figsize=(10, 4.6), dpi=100)
    fig.patch.set_facecolor(BG)
    axA = fig.add_axes([0.06, 0.14, 0.42, 0.68]); axB = fig.add_axes([0.56, 0.14, 0.40, 0.68])
    for ax in (axA, axB):
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    title = fig.text(0.06, 0.93, '', color=FG, fontsize=14, fontweight='bold')
    subt = fig.text(0.06, 0.875, '', color=MUTED, fontsize=9.5)

    xmax, ymax = np.percentile(F[:, 0], 99.5) * 1.1, np.percentile(F[:, 1], 99.5) * 1.1
    N1, N2, N3, N4, HOLD = 45, 25, 40, 35, 20
    total = N1 + N2 + N3 + N4 + HOLD
    ells = []
    for k in range(gm.K):
        j = order[k]
        C = gm.scaler.scale_[:, None] * gm.gmm.covariances_[j] * gm.scaler.scale_[None]
        ev, V = np.linalg.eigh(C)
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        ells.append(Ellipse(mu[j], 2 * 2 * np.sqrt(ev[1]), 2 * 2 * np.sqrt(ev[0]), angle=ang, fill=False,
                            lw=1.8, color=PAL[k % len(PAL)], alpha=0.0))
        axA.add_patch(ells[-1])
    scat = axA.scatter([], [], s=3, linewidths=0)
    axA.set_xlim(0, xmax); axA.set_ylim(0, ymax)
    axA.set_xlabel('|gravity-free acc| [m/s²]', color=MUTED, fontsize=9); axA.set_ylabel('|gyr| [rad/s]', color=MUTED, fontsize=9)
    x = np.arange(gm.K)
    bars = axB.bar(x, share_raw, color=[PAL[k % len(PAL)] for k in range(gm.K)], width=0.62)
    ghost = axB.bar(x, share_raw, color='none', edgecolor=MUTED, lw=1.0, ls='--', width=0.62)
    for g in ghost:
        g.set_alpha(0)
    axB.set_xticks(x); axB.set_xticklabels([f'M{k + 1}' for k in range(gm.K)], color=MUTED)
    axB.set_ylim(0, max(share_raw.max(), share_bal.max()) * 1.18); axB.set_ylabel('share of training signal', color=MUTED, fontsize=9)
    labs = [axB.text(k, 0, '', ha='center', va='bottom', color=FG, fontsize=8) for k in range(gm.K)]
    loss_ln, = axB.plot([], [], color=FG, lw=2)
    axB2 = None

    def frame(f):
        nonlocal axB2
        if f < N1:                                         # 1. features stream in
            q = ease((f + 1) / N1)
            k = tss <= q * ts[-1]
            scat.set_offsets(Fs[k]); scat.set_color(MUTED)
            title.set_text('1  Motion features of the warm-up IMU stream')
            subt.set_text(f'{a.seq}: first {ts[-1]:.0f} s, GT-free  (t = {q * ts[-1]:4.1f} s)')
            for b in bars:
                b.set_alpha(0)
        elif f < N1 + N2:                                  # 2. GMM
            q = ease((f - N1 + 1) / N2)
            scat.set_offsets(Fs)
            base = np.array(matplotlib.colors.to_rgba(MUTED))
            cols = np.array([matplotlib.colors.to_rgba(PAL[c % len(PAL)]) for c in cs])
            scat.set_color((1 - q) * base + q * cols)
            for e in ells:
                e.set_alpha(0.9 * q)
            title.set_text(f'2  GMM motion regimes (K = {gm.K} by BIC)')
            subt.set_text('each sample is assigned to its most likely regime M1 (calm) ... M%d (aggressive)' % gm.K)
            for b, s in zip(bars, share_raw):
                b.set_alpha(q); b.set_height(s)
            for lb, s in zip(labs, share_raw):
                lb.set_text(f'{100 * s:.0f}%'); lb.set_y(s); lb.set_alpha(q)
        elif f < N1 + N2 + N3:                             # 3. reweighting morph
            q = ease((f - N1 - N2 + 1 - 8) / (N3 - 16))
            sh = (1 - q) * share_raw + q * share_bal
            for b, s in zip(bars, sh):
                b.set_height(s)
            for g, s in zip(ghost, share_raw):
                g.set_alpha(0.8 * min(1, q * 3)); g.set_height(s)
            for lb, s in zip(labs, sh):
                lb.set_text(f'{100 * s:.0f}%'); lb.set_y(s)
            title.set_text('3  Motion-balanced reweighting')
            subt.set_text('effective-number weights per regime: rare motion is no longer drowned out (dashed: raw share)')
        else:                                              # 4. training
            q = ease((f - N1 - N2 - N3 + 1) / N4)
            title.set_text('4  Balanced regimes drive the denoiser training')
            subt.set_text('+ frequency gate, rare-regime boost, per-regime convergence; then online PGO with the corrected IMU')
            if hist and axB2 is None:
                axB2 = fig.add_axes([0.30, 0.53, 0.17, 0.26]); axB2.set_facecolor(BG)
                for s in axB2.spines.values():
                    s.set_color(GRID)
                axB2.tick_params(colors=MUTED, labelsize=7)
                axB2.set_title('training loss', color=MUTED, fontsize=8)
                L = np.array([h['train_loss'] for h in hist])
                axB2.set_xlim(0, len(L)); axB2.set_ylim(np.nanmin(L) * 0.9, np.nanpercentile(L, 98) * 1.05)
                axB2._L = L
                axB2._ln, = axB2.plot([], [], color=FG, lw=1.6)
            if axB2 is not None:
                n = max(1, int(q * len(axB2._L)))
                axB2._ln.set_data(np.arange(n), axB2._L[:n])
        return []

    anim = FuncAnimation(fig, frame, frames=total, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=a.fps), savefig_kwargs={'facecolor': BG})
    print('saved', a.out)


if __name__ == '__main__':
    main()
