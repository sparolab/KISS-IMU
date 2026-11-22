"""Animated explainer of KISS-IMU's motion-balanced learning on a run's warm-up (GT-free, as in training):

    1. how a motion feature is made: a 0.2 s window slides over the IMU stream; the window's mean rate
       |mean w| and mean gravity-free specific force |mean (R_yaw a - g)| become one point (kiss_imu/training/
       reweight.py: motion_features), then the whole warm-up fills the feature plane
    2. the GMM (K by BIC) splits the plane into motion regimes, shown on the plane and along time
    3. per-regime share of the training signal, raw -> after the effective-number weights (histogram morph)
    4. the balanced regimes drive the denoiser training (training loss of the run's last label round)

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
from matplotlib.patches import ConnectionPatch, Ellipse, Rectangle  # noqa: E402

from kiss_imu.config import load_config  # noqa: E402
from kiss_imu.data import Sequence  # noqa: E402
from kiss_imu.training.reweight import G, MotionGMM, _moving_mean, features_fn  # noqa: E402

BG, FG, MUTED, GRID = '#0f1115', '#e8e8ea', '#8a8f98', '#2a2e36'
PAL = ['#3fb8af', '#f2a33a', '#e0607e', '#8f7ce8', '#6cc24a', '#4a90d9', '#d9c84a', '#c9ccd1']
AX3 = ['#e0607e', '#6cc24a', '#4a90d9']          # x, y, z traces
WIN_COL = '#f2d16b'


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7.5)


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
    win_s = rw.get('win_s', 0.2)
    seq = Sequence(d['root'], a.seq, d.get('type', 'generic'), 0.0, gt_file=d.get('gt_file', 'gt_pose.csv'))
    t0 = float(np.loadtxt(os.path.join(a.run, 'online.txt'), max_rows=1)[0])
    res = json.load(open(os.path.join(a.run, 'result.json')))['warmstart']
    t1 = t0 + res['t_train']

    # features exactly as the trainer computes them (whole stream, then the warm-up samples)
    F_all = features_fn(rw)(seq.imu_t, seq.gyr, seq.acc, win_s)
    m = (seq.imu_t >= t0) & (seq.imu_t <= t1)
    F, ts = F_all[m], seq.imu_t[m] - t0
    gyr = seq.gyr[m]
    # the gravity-free specific force the 'norm' feature averages (yaw-only attitude, as in motion_features)
    dt = np.diff(seq.imu_t, prepend=seq.imu_t[0]); dt[dt <= 0] = np.median(dt[dt > 0])
    yaw = np.cumsum(seq.gyr[:, 2] * dt)
    c, s = np.cos(yaw), np.sin(yaw)
    g = G * (1.0 if np.median(seq.acc[:, 2]) >= 0 else -1.0)
    acc_w = np.stack([c * seq.acc[:, 0] - s * seq.acc[:, 1], s * seq.acc[:, 0] + c * seq.acc[:, 1], seq.acc[:, 2]], 1) - g
    acc_w = acc_w[m]
    L = max(1, int(round(win_s / np.median(dt))))
    gyr_mean, acc_mean = _moving_mean(gyr, L), _moving_mean(acc_w, L)

    gm = MotionGMM(rw.get('n_components'), tuple(rw.get('k_range', (2, 5))), win_s,
                   rw.get('method', 'effective'), rw.get('beta', 0.999), tuple(rw.get('clamp', (0.1, 10.0))),
                   seed=rw.get('seed', 0)).fit([F])
    P = gm.proba(F)
    mu = gm.scaler.inverse_transform(gm.gmm.means_)
    order = np.argsort(mu[:, 1] + 0.05 * mu[:, 0])          # regimes calm -> aggressive
    rank = np.empty_like(order); rank[order] = np.arange(gm.K)
    comp = rank[P.argmax(1)]
    counts = P.sum(0)[order]                                  # soft counts, as the trainer
    cw = np.clip(gm.component_weights(P.sum(0)), *gm.clamp)[order]
    share_raw = counts / counts.sum()
    share_bal = counts * cw / (counts * cw).sum()
    tdirs = sorted(x for x in os.listdir(a.run) if x.startswith('train'))
    hp = os.path.join(a.run, tdirs[-1], 'history.json') if tdirs else ''
    hist = json.load(open(hp)) if os.path.exists(hp) else None

    # display decimation (~100 Hz traces)
    dec = max(1, int(round(0.01 / np.median(dt))))
    td, gd, ad = ts[::dec], gyr[::dec], acc_w[::dec]
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(F), min(4000, len(F)), replace=False))
    Fs, cs, tss = F[sub], comp[sub], ts[sub]
    T = ts[-1]

    fig = plt.figure(figsize=(11, 6.2), dpi=100)
    fig.patch.set_facecolor(BG)
    axG = fig.add_axes([0.06, 0.645, 0.40, 0.18]); axA = fig.add_axes([0.06, 0.405, 0.40, 0.18])
    axR = fig.add_axes([0.06, 0.345, 0.40, 0.03])
    axS = fig.add_axes([0.585, 0.40, 0.385, 0.425])
    axH = fig.add_axes([0.06, 0.06, 0.40, 0.165]); axL = fig.add_axes([0.585, 0.06, 0.385, 0.165])
    for ax in (axG, axA, axR, axS, axH, axL):
        style(ax)
    title = fig.text(0.06, 0.945, '', color=FG, fontsize=14, fontweight='bold')
    subt = fig.text(0.06, 0.895, '', color=MUTED, fontsize=9.5)

    # time panels
    gl = [axG.plot(td, gd[:, i], color=AX3[i], lw=0.8, alpha=0.75)[0] for i in range(3)]
    al = [axA.plot(td, ad[:, i], color=AX3[i], lw=0.8, alpha=0.75)[0] for i in range(3)]
    gy = np.percentile(np.abs(gd), 99.5) * 1.15; ay = np.percentile(np.abs(ad), 99.5) * 1.15
    axG.set_ylim(-gy, gy); axA.set_ylim(-ay, ay)
    axG.set_ylabel('ω  [rad/s]', color=MUTED, fontsize=8.5); axA.set_ylabel('R_yaw a − g  [m/s²]', color=MUTED, fontsize=8.5)
    axG.text(0.01, 0.88, 'gyro x y z', transform=axG.transAxes, color=MUTED, fontsize=7.5)
    axA.text(0.01, 0.88, 'gravity-free acc x y z', transform=axA.transAxes, color=MUTED, fontsize=7.5)
    axG.set_xticklabels([]); axA.set_xticklabels([])
    bandG = Rectangle((0, 0), win_s, 1, transform=axG.get_xaxis_transform(), color=WIN_COL, alpha=0.28, lw=0)
    bandA = Rectangle((0, 0), win_s, 1, transform=axA.get_xaxis_transform(), color=WIN_COL, alpha=0.28, lw=0)
    axG.add_patch(bandG); axA.add_patch(bandA)
    mG = [axG.plot([], [], color=AX3[i], lw=2.6)[0] for i in range(3)]       # window means
    mA = [axA.plot([], [], color=AX3[i], lw=2.6)[0] for i in range(3)]
    axR.set_yticks([]); axR.set_xlabel('time in the warm-up [s]', color=MUTED, fontsize=8)
    axR.set_ylim(0, 1)
    strip = axR.imshow(np.zeros((1, len(td), 4)), aspect='auto', extent=(0, T, 0, 1), interpolation='nearest')
    axR.text(-0.01, 0.5, 'regime', transform=axR.transAxes, ha='right', va='center', color=MUTED, fontsize=7.5)

    # feature plane
    xmax, ymax = np.percentile(F[:, 0], 99.5) * 1.1, np.percentile(F[:, 1], 99.5) * 1.1
    axS.set_xlim(0, xmax); axS.set_ylim(0, ymax)
    axS.set_xlabel('|mean gravity-free acc|  [m/s²]', color=MUTED, fontsize=8.5)
    axS.set_ylabel('|mean ω|  [rad/s]', color=MUTED, fontsize=8.5)
    axS.set_title('motion feature plane', color=MUTED, fontsize=9)
    scat = axS.scatter([], [], s=3, linewidths=0)
    cur = axS.scatter([], [], s=70, color=WIN_COL, edgecolors=BG, linewidths=1.2, zorder=5)
    ells = []
    for k in range(gm.K):
        j = order[k]
        C = gm.scaler.scale_[:, None] * gm.gmm.covariances_[j] * gm.scaler.scale_[None]
        ev, V = np.linalg.eigh(C)
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        e = Ellipse(mu[j], 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0]), angle=ang, fill=False, lw=1.8,
                    color=PAL[k % len(PAL)], alpha=0.0)
        axS.add_patch(e); ells.append(e)
    etxt = [axS.text(mu[order[k]][0], mu[order[k]][1], f'M{k + 1}', color=PAL[k % len(PAL)], fontsize=9,
                     fontweight='bold', ha='center', va='center', alpha=0, zorder=6,
                     bbox=dict(boxstyle='round,pad=0.15', fc=BG, ec='none', alpha=0)) for k in range(gm.K)]
    formula = fig.text(0.60, 0.80, '', color=WIN_COL, fontsize=9, ha='left', va='top', linespacing=1.6,
                       bbox=dict(boxstyle='round,pad=0.4', fc=BG, ec=WIN_COL, lw=0.8))
    conG = ConnectionPatch((0, 0), (0, 0), 'data', 'data', axesA=axG, axesB=axS, color=WIN_COL, lw=1.2,
                           arrowstyle='-|>', alpha=0)
    conA = ConnectionPatch((0, 0), (0, 0), 'data', 'data', axesA=axA, axesB=axS, color=WIN_COL, lw=1.2,
                           arrowstyle='-|>', alpha=0)
    fig.add_artist(conG); fig.add_artist(conA)

    # histogram + loss
    x = np.arange(gm.K)
    bars = axH.bar(x, share_raw, color=[PAL[k % len(PAL)] for k in range(gm.K)], width=0.62)
    ghost = axH.bar(x, share_raw, color='none', edgecolor=MUTED, lw=1.0, ls='--', width=0.62)
    axH.set_xticks(x); axH.set_xticklabels([f'M{k + 1}' for k in range(gm.K)], color=MUTED)
    axH.set_ylim(0, max(share_raw.max(), share_bal.max()) * 1.3)
    axH.set_title('share of the training signal per regime', color=MUTED, fontsize=9)
    labs = [axH.text(k, 0, '', ha='center', va='bottom', color=FG, fontsize=8) for k in range(gm.K)]
    axL.set_title('denoiser training loss', color=MUTED, fontsize=9)
    Lh = np.array([h['train_loss'] for h in hist]) if hist else np.zeros(1)
    axL.set_xlim(0, max(1, len(Lh))); axL.set_ylim(np.nanmin(Lh) * 0.9, np.nanpercentile(Lh, 98) * 1.05 + 1e-9)
    axL.set_xlabel('epoch', color=MUTED, fontsize=8)
    loss_ln, = axL.plot([], [], color=FG, lw=1.8)
    for b, gb in zip(bars, ghost):
        b.set_alpha(0); gb.set_alpha(0)
    for ax in (axH, axL, axR):
        ax.set_visible(False)

    N1a, N1b, N2, N3, N4, HOLD = 60, 30, 30, 40, 35, 20
    SLOW_T = 4.0                                   # seconds of the warm-up shown at walking pace
    VIEW = 3.0                                     # visible time span while the window slides
    total = N1a + N1b + N2 + N3 + N4 + HOLD
    reg_rgba = np.array([matplotlib.colors.to_rgba(PAL[c % len(PAL)]) for c in comp[::dec]])
    base = np.array(matplotlib.colors.to_rgba(MUTED))
    cols = np.array([matplotlib.colors.to_rgba(PAL[c % len(PAL)]) for c in cs])

    def set_window(tc):
        lo, hi = tc - win_s / 2, tc + win_s / 2
        for band in (bandG, bandA):
            band.set_x(lo)
        i = min(np.searchsorted(ts, tc), len(ts) - 1)
        for k in range(3):
            mG[k].set_data([lo, hi], [gyr_mean[i, k]] * 2)
            mA[k].set_data([lo, hi], [acc_mean[i, k]] * 2)
        return i

    def frame(f):
        if f < N1a + N1b:                                  # 1. a sliding window makes the features
            if f < N1a:
                q = (f + 1) / N1a
                tc = win_s / 2 + q * (SLOW_T - win_s / 2)
                arrow = 1.0
            else:
                q = ease((f - N1a + 1) / N1b)
                tc = SLOW_T + q * (T - win_s / 2 - SLOW_T)
                arrow = max(0.0, 1 - 3 * q)
            i = set_window(tc)
            lo = max(0.0, tc - VIEW * 0.8)
            axG.set_xlim(lo, lo + VIEW); axA.set_xlim(lo, lo + VIEW)
            k = tss <= tc
            scat.set_offsets(Fs[k] if k.any() else np.empty((0, 2))); scat.set_color(MUTED)
            cur.set_offsets([F[i]])
            conG.xy1, conG.xy2 = (tc + win_s / 2, gy * 0.6), tuple(F[i])
            conA.xy1, conA.xy2 = (tc + win_s / 2, -ay * 0.6), tuple(F[i])
            conG.set_alpha(arrow); conA.set_alpha(arrow)
            formula.set_alpha(arrow)
            formula.set_text(f'0.2 s window  →  one point\n'
                             f'x = |mean(R_yaw a − g)| = {F[i, 0]:.2f}\n'
                             f'y = |mean ω| = {F[i, 1]:.2f}')
            formula.get_bbox_patch().set_alpha(arrow)
            title.set_text('1  Motion features from a sliding window')
            subt.set_text(f'{a.seq}, first {T:.0f} s of the IMU only (GT-free):  t = {tc:4.1f} s, '
                          f'{len(F)} samples → {len(F)} feature points')
        elif f < N1a + N1b + N2:                           # 2. GMM regimes, on the plane and in time
            q = ease((f - N1a - N1b + 1) / N2)
            if f == N1a + N1b:
                cur.set_offsets(np.empty((0, 2)))
                for ln in mG + mA:
                    ln.set_data([], [])
                for band in (bandG, bandA):
                    band.set_alpha(0)
                formula.set_text(''); formula.get_bbox_patch().set_alpha(0)
                axR.set_visible(True)
            scat.set_offsets(Fs)
            scat.set_color((1 - q) * base + q * cols)
            for e in ells:
                e.set_alpha(0.9 * q)
            for t in etxt:
                t.set_alpha(q); t.get_bbox_patch().set_alpha(0.6 * q)
            lo = (1 - q) * max(0.0, T - VIEW); hi = (1 - q) * (max(0.0, T - VIEW) + VIEW) + q * T
            axG.set_xlim(lo, hi); axA.set_xlim(lo, hi); axR.set_xlim(0, T)
            img = reg_rgba.copy(); img[:, 3] = q
            strip.set_data(img[None])
            title.set_text(f'2  GMM motion regimes (K = {gm.K} by BIC)')
            subt.set_text('each feature point gets its most likely regime, M1 (calm) ... M%d (aggressive): '
                          'the same split along time' % gm.K)
        elif f < N1a + N1b + N2 + N3:                      # 3. reweighting
            axH.set_visible(True)
            q = ease((f - N1a - N1b - N2 + 1 - 10) / (N3 - 18))
            qa = ease((f - N1a - N1b - N2 + 1) / 8)
            sh = (1 - q) * share_raw + q * share_bal
            for b, sv in zip(bars, sh):
                b.set_alpha(qa); b.set_height(sv)
            for gb, sv in zip(ghost, share_raw):
                gb.set_alpha(0.8 * min(1, q * 3)); gb.set_height(sv)
            for lb, sv, wk in zip(labs, sh, cw):
                lb.set_text(f'{100 * sv:.0f}%' + (f'  ×{wk:.2f}' if q > 0.5 else '')); lb.set_y(sv); lb.set_alpha(qa)
            title.set_text('3  Motion-balanced reweighting')
            subt.set_text('effective-number weight per regime: rare motion is no longer drowned out (dashed: raw share)')
        else:                                              # 4. training
            axL.set_visible(True)
            q = ease((f - N1a - N1b - N2 - N3 + 1) / N4)
            n = max(1, int(q * len(Lh)))
            loss_ln.set_data(np.arange(n), Lh[:n])
            title.set_text('4  The balanced regimes drive the denoiser training')
            subt.set_text('+ frequency gate, rare-regime boost, per-regime convergence; '
                          'then the running PGO switches to the corrected IMU')
        return []

    anim = FuncAnimation(fig, frame, frames=total, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=a.fps), savefig_kwargs={'facecolor': BG})
    print('saved', a.out, 'K', gm.K, 'share raw', share_raw.round(3), 'bal', share_bal.round(3))


if __name__ == '__main__':
    main()
