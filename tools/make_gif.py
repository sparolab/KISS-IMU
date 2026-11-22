"""Compact animation of one online KISS-IMU run: the map being built on the left, the step on the right, and
running numbers below the step.

    1. Data selection    - the first warmup_s seconds on the raw IMU; their LiDAR odometry poses are the labels
    2. Motion regimes    - a GMM over the sliding-window motion features of those seconds
    3. Reweighting       - per-regime share of the training signal, raw -> motion-balanced
    4. Training          - the denoiser's training loss
    5. Learned IO -> PGO - the IMU flows through the network into the running PGO; the map is built to the end

Best with a run whose hand-over is at warmup_s (a released checkpoint replayed without training):

    python3 scripts/run_warmstart.py configs/diter_os/warmstart.yaml --seqs Forest_new --out runs/ws_pre \
        --set warmstart.adapt=false warmstart.pretrained=checkpoints/diter_os/Forest_new/kiss_imu_ws30.pt
    python3 tools/make_gif.py configs/diter_os/warmstart.yaml Forest_new runs/ws_pre/Forest_new \
        --train checkpoints/diter_os/Forest_new --raw runs/raw/Forest_new --out fig/kiss_imu_online.gif
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from kiss_imu.config import load_config  # noqa: E402
from kiss_imu.data import Sequence  # noqa: E402
from kiss_imu.training.reweight import MotionGMM, features_fn  # noqa: E402

BG, FG, MUTED, LIGHT, MAPC = '#ffffff', '#1f2328', '#59636e', '#d0d7de', '#aab3bd'
C_RAW, C_LRN, C_SEL, C_LOSS = '#3F51B5', '#00897B', '#D81B60', '#F57C00'
PAL = ['#3F51B5', '#00897B', '#F57C00', '#D81B60', '#8f7ce8', '#6cc24a']


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def use_font():
    for d in ('/usr/share/fonts/opentype/inter', '/usr/share/fonts/host/inter', os.path.expanduser('~/.fonts')):
        for f in glob.glob(os.path.join(d, 'Inter-*.otf')) + glob.glob(os.path.join(d, 'Inter*.ttf')):
            font_manager.fontManager.addfont(f)
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family'] = 'Inter' if 'Inter' in names else 'DejaVu Sans'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('seq'); ap.add_argument('run')
    ap.add_argument('--train', help="dir with history.json (default: the run's last train* dir)")
    ap.add_argument('--raw', help='raw-IMU run dir (result.json) for the ATE line')
    ap.add_argument('--out', default='fig/kiss_imu_online.gif')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--map-every', type=int, default=3)
    ap.add_argument('--points', type=int, default=900)
    ap.add_argument('--zmax', type=float, default=3.0)
    ap.add_argument('--set', nargs='*', default=[], dest='overrides')
    a = ap.parse_args()
    use_font()
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
    W = float(w['warmup_s'])
    T = t[-1] - t0
    res = json.load(open(os.path.join(a.run, 'result.json')))
    drift = res.get('imu_drift_after_switch', {})
    att_raw = drift.get('raw', {}).get('att_deg', {}).get('60.0')
    att_lrn = drift.get('used', {}).get('att_deg', {}).get('60.0')
    ate_lrn = res['online']['ate_aligned_rmse']
    ate_raw = json.load(open(os.path.join(a.raw, 'result.json')))['online']['ate_aligned_rmse'] if a.raw else None

    # map points
    k_scan = np.searchsorted(seq.scan_t, t).clip(0, len(seq.scan_t) - 1)
    rng = np.random.default_rng(0)
    mp, mt = [], []
    for i in range(0, len(t), a.map_every):
        x = seq.scan(k_scan[i])
        x = x[rng.choice(len(x), min(a.points, len(x)), replace=False)]
        x = x[x[:, 2] < a.zmax]
        mp.append(Rq[i].apply(x) + P[i]); mt.append(np.full(len(x), t[i] - t0))
    mp, mt = np.concatenate(mp), np.concatenate(mt)
    tt = t - t0

    # motion features + GMM of the warm-up (as the trainer)
    m = (seq.imu_t >= t0) & (seq.imu_t <= t0 + W)
    F = features_fn(rw)(seq.imu_t, seq.gyr, seq.acc, rw.get('win_s', 0.2))[m]
    ti = seq.imu_t[m] - t0
    gm = MotionGMM(rw.get('n_components'), tuple(rw.get('k_range', (2, 5))), rw.get('win_s', 0.2),
                   rw.get('method', 'effective'), rw.get('beta', 0.999), tuple(rw.get('clamp', (0.1, 10.0))),
                   seed=rw.get('seed', 0)).fit([F])
    Pr = gm.proba(F)
    mu = gm.scaler.inverse_transform(gm.gmm.means_)
    order = np.argsort(mu[:, 1] + 0.05 * mu[:, 0])
    rank = np.empty_like(order); rank[order] = np.arange(gm.K)
    comp = rank[Pr.argmax(1)]
    counts = Pr.sum(0)[order]
    cw = np.clip(gm.component_weights(Pr.sum(0)), *gm.clamp)[order]
    s_raw, s_bal = counts / counts.sum(), counts * cw / (counts * cw).sum()
    rare = int(np.argmin(s_raw))
    tdir = a.train
    if tdir is None:
        tdirs = sorted(x for x in os.listdir(a.run) if x.startswith('train'))
        tdir = os.path.join(a.run, tdirs[-1]) if tdirs else None
    hp = os.path.join(tdir, 'history.json') if tdir else ''
    L = np.array([h['train_loss'] for h in json.load(open(hp))]) if os.path.exists(hp) else np.ones(2)

    # ---- figure
    fig = plt.figure(figsize=(9.6, 4.8), dpi=100)
    fig.patch.set_facecolor(BG)
    axO = fig.add_axes([0, 0, 1, 1]); axO.set_xlim(0, 1); axO.set_ylim(0, 1); axO.axis('off'); axO.set_zorder(3)
    axO.patch.set_alpha(0)
    axM = fig.add_axes([0.01, 0.10, 0.535, 0.78]); axM.axis('off')
    lo, hi = P[:, :2].min(0) - 12.0, P[:, :2].max(0) + 12.0          # the whole trajectory + a margin
    r = (0.535 * 9.6) / (0.78 * 4.8)                                    # axes width / height in inches
    c, span = (lo + hi) / 2, hi - lo
    sx = max(span[0], span[1] * r)
    axM.set_xlim(c[0] - sx / 2, c[0] + sx / 2); axM.set_ylim(c[1] - sx / r / 2, c[1] + sx / r / 2)
    sc = axM.scatter(np.zeros(0), np.zeros(0), s=0.15, color=MAPC, linewidths=0, alpha=0.55)
    ln_sel, = axM.plot([], [], color=C_SEL, lw=2.6, solid_capstyle='round', zorder=4)
    ln_lrn, = axM.plot([], [], color=C_LRN, lw=2.0, solid_capstyle='round', zorder=3)
    head = axM.scatter([], [], s=40, color=FG, zorder=5, edgecolors=BG, linewidths=1.2)
    # legend
    for k, (c, s) in enumerate([(C_SEL, 'first 30 s, raw IMU (training data)'), (C_LRN, 'learned IO in the PGO')]):
        x0 = 0.02 + k * 0.30
        axO.scatter([x0], [0.955], s=40, color=c)
        axO.text(x0 + 0.012, 0.955, s, color=c, fontsize=8.5, va='center', fontweight='bold')
    clock = axO.text(0.02, 0.905, '', color=MUTED, fontsize=8.5, va='center')
    axO.plot([0.56, 0.56], [0.03, 0.97], color=LIGHT, lw=0.8)

    # sidebar
    X0 = 0.585
    BW = 0.39
    title = axO.text(X0, 0.935, '', color=FG, fontsize=15, fontweight='bold', va='center')
    subt = axO.text(X0, 0.865, '', color=MUTED, fontsize=9.2, va='center', linespacing=1.4)
    plab = axO.text(X0, 0.795, '', color=FG, fontsize=8.5, va='center')
    axP = fig.add_axes([X0 + 0.04, 0.47, BW - 0.04, 0.30]); axP.set_zorder(2)

    def clear_panel():
        axP.clear(); axP.set_facecolor(BG); axP.axis('on')
        for s in axP.spines.values():
            s.set_color(LIGHT)
        axP.tick_params(colors=MUTED, labelsize=7.5, length=2)

    # running numbers (bars)
    rows = []
    for k in range(4):
        y = 0.375 - k * 0.088
        lab = axO.text(X0, y, '', color=FG, fontsize=8.3, va='bottom')
        val = axO.text(X0 + BW, y, '', color=FG, fontsize=8.3, fontweight='bold', va='bottom', ha='right')
        ghost = FancyBboxPatch((X0, y - 0.028), 0, 0.017, boxstyle='square,pad=0', fc=LIGHT, ec='none')
        bar = FancyBboxPatch((X0, y - 0.028), 0, 0.017, boxstyle='square,pad=0', fc=LIGHT, ec='none')
        axO.add_patch(ghost); axO.add_patch(bar)
        rows.append((lab, val, ghost, bar))

    def row(k, label, value, frac, color, ghost=None):
        lab, val, g, b = rows[k]
        lab.set_text(label); val.set_text(value)
        b.set_width(BW * float(np.clip(frac, 0, 1))); b.set_facecolor(color)
        g.set_width(BW * float(np.clip(ghost, 0, 1)) if ghost is not None else 0)

    pill = FancyBboxPatch((0.015, 0.03), 0.53, 0.06, boxstyle='round,pad=0,rounding_size=0.012', fc='#fce4ec',
                          ec='none', alpha=0)
    axO.add_patch(pill)
    pill_t = axO.text(0.025, 0.06, '', color=C_SEL, fontsize=9, fontweight='bold', va='center')

    N1, N2, N3, N4, N5, HOLD = 80, 75, 75, 60, 130, 45
    total = N1 + N2 + N3 + N4 + N5 + HOLD
    i_sel = np.searchsorted(tt, W)
    rng2 = np.random.default_rng(1)
    sub = np.sort(rng2.choice(len(F), min(1500, len(F)), replace=False))
    Fs, cs = F[sub], comp[sub]
    xmax, ymax = np.percentile(F[:, 0], 99.5) * 1.1, np.percentile(F[:, 1], 99.5) * 1.1
    gnorm = np.linalg.norm(seq.gyr[m], axis=1)
    dec = max(1, len(ti) // 600)
    state = {'stage': -1}
    rare_lab = f'rarest motion regime M{rare + 1}: share of the signal'

    def draw_map(tc):
        k = mt <= tc
        sc.set_offsets(mp[k, :2])
        j = int(np.searchsorted(tt, tc))
        js = min(j, i_sel)
        ln_sel.set_data(P[:js + 1, 0], P[:js + 1, 1])
        if j > i_sel:
            ln_lrn.set_data(P[i_sel:j + 1, 0], P[i_sel:j + 1, 1])
        head.set_offsets([P[min(j, len(P) - 1), :2]])
        clock.set_text(f'{a.seq}   t = {tc:5.1f} s / {T:.0f} s')

    def stage(sidx, ttl, st, pl):
        title.set_text(ttl); subt.set_text(st); plab.set_text(pl)
        if state['stage'] != sidx:
            state['stage'] = sidx
            clear_panel()
            return True
        return False

    def frame(f):
        if f >= N1:
            row(0, 'training data', f'first {W:.0f} s of {T:.0f} s', W / T, C_SEL)
        if f >= N1 + N2 + N3:
            row(1, rare_lab, f'{100 * s_raw[rare]:.0f} % → {100 * s_bal[rare]:.0f} %', s_bal[rare],
                PAL[rare % len(PAL)], ghost=s_raw[rare])
        if f >= N1 + N2 + N3 + N4:
            row(2, 'training loss', f'{L[0]:.2f} → {L[-1]:.2f}', L[-1] / L[0], C_LOSS, ghost=1.0)

        if f < N1:                                                     # 1. data selection
            q = min(1.0, (f + 1) / (0.75 * N1))
            tc = q * W
            new = stage(0, 'Data selection', 'the first 30 s run on the raw IMU;\nLiDAR odometry poses become the labels',
                        '|ω| of the IMU stream  [rad/s]')
            if new:
                axP.set_xlim(0, W); axP.set_ylim(0, np.percentile(gnorm, 99.5) * 1.1)
                axP._ln, = axP.plot([], [], color=C_SEL, lw=1.0)
                axP.set_xlabel('t [s]', color=MUTED, fontsize=7)
            k = ti[::dec] <= tc
            axP._ln.set_data(ti[::dec][k], gnorm[::dec][k])
            draw_map(tc)
            row(0, 'training data', f'{tc:4.1f} s of {T:.0f} s', tc / T, C_SEL)
        elif f < N1 + N2:                                              # 2. motion regimes
            q = ease((f - N1 + 1 - 10) / (0.55 * N2))
            new = stage(1, 'Motion regimes', 'a 0.2 s window turns the IMU into motion\nfeatures; a GMM splits them (K by BIC)',
                        f'motion features, K = {gm.K} regimes')
            if new:
                axP.set_xlim(0, xmax); axP.set_ylim(0, ymax)
                axP.set_xlabel('|mean gravity-free acc|', color=MUTED, fontsize=7)
                axP.set_ylabel('|mean ω|', color=MUTED, fontsize=7)
                axP._sc = axP.scatter(Fs[:, 0], Fs[:, 1], s=5, linewidths=0, color=MAPC)
                axP._el = []
                for kk in range(gm.K):
                    j = order[kk]
                    C = gm.scaler.scale_[:, None] * gm.gmm.covariances_[j] * gm.scaler.scale_[None]
                    ev, V = np.linalg.eigh(C)
                    e = Ellipse(mu[j], 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0]),
                                angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])), fill=False, lw=2.0,
                                color=PAL[kk % len(PAL)], alpha=0)
                    axP.add_patch(e); axP._el.append(e)
                axP._mt = [axP.text(*mu[order[kk]], f'M{kk + 1}', color='white', fontsize=8.5, fontweight='bold', ha='center',
                                    va='center', alpha=0, zorder=6, bbox=dict(boxstyle='round,pad=0.2', fc=PAL[kk % len(PAL)], ec='none', alpha=0))
                           for kk in range(gm.K)]
            base = np.array(to_rgba(MAPC))
            cols = np.array([to_rgba(PAL[c % len(PAL)]) for c in cs])
            axP._sc.set_color((1 - q) * base + q * cols)
            for e in axP._el:
                e.set_alpha(q)
            for tx in axP._mt:
                tx.set_alpha(q); tx.get_bbox_patch().set_alpha(q)
            row(1, rare_lab, f'{100 * s_raw[rare]:.0f} %', q * s_raw[rare], PAL[rare % len(PAL)])
        elif f < N1 + N2 + N3:                                         # 3. reweighting
            q = ease((f - N1 - N2 + 1 - 15) / (0.5 * N3))
            new = stage(2, 'Motion-balanced reweighting', 'effective-number weights per regime:\nrare motion is not drowned out',
                        'share of the training signal per regime')
            x = np.arange(gm.K)
            if new:
                axP._b = axP.bar(x, s_raw, color=[PAL[k % len(PAL)] for k in range(gm.K)], width=0.6)
                axP.bar(x, s_raw, color='none', edgecolor=MUTED, lw=0.9, ls='--', width=0.6)
                axP.set_xticks(x); axP.set_xticklabels([f'M{k + 1}' for k in range(gm.K)])
                axP.set_ylim(0, max(s_raw.max(), s_bal.max()) * 1.25)
                axP._t = [axP.text(k, 0, '', ha='center', va='bottom', color=FG, fontsize=8.5, fontweight='bold') for k in range(gm.K)]
                axP._w = [axP.text(k, 0.004, f'×{cw[k]:.2f}', ha='center', va='bottom', color='white', fontsize=8, alpha=0)
                          for k in range(gm.K)]
            sh = (1 - q) * s_raw + q * s_bal
            for b, s, tx in zip(axP._b, sh, axP._t):
                b.set_height(s); tx.set_y(s); tx.set_text(f'{100 * s:.0f}%')
            for tw in axP._w:
                tw.set_alpha(min(1.0, 2 * q))
            row(1, rare_lab, f'{100 * s_raw[rare]:.0f} % → {100 * sh[rare]:.0f} %', sh[rare], PAL[rare % len(PAL)],
                ghost=s_raw[rare])
        elif f < N1 + N2 + N3 + N4:                                    # 4. training
            q = ease((f - N1 - N2 - N3 + 1) / (0.75 * N4))
            new = stage(3, 'Training', 'the denoiser learns gyro + acc corrections\nfrom the balanced labels (multi-horizon)',
                        'training loss')
            if new:
                axP.set_xlim(0, len(L)); axP.set_ylim(min(L) * 0.9, max(L) * 1.05)
                axP.set_xlabel('epoch', color=MUTED, fontsize=7)
                axP._ln, = axP.plot([], [], color=C_LOSS, lw=2.2)
            n = max(1, int(q * len(L)))
            axP._ln.set_data(np.arange(n), L[:n])
            row(2, 'training loss', f'{L[0]:.2f} → {L[n - 1]:.2f}', L[n - 1] / L[0], C_LOSS, ghost=1.0)
        else:                                                          # 5. learned IO -> PGO
            q = min(1.0, (f - N1 - N2 - N3 - N4 + 1) / (0.85 * N5))
            tc = W + q * (T - W)
            new = stage(4, 'Learned IO → PGO', 'the running PGO switches to the corrected\nIMU, no restart; the map is built',
                        'data flow')
            if new:
                axP.set_xlim(0, 1); axP.set_ylim(0, 1); axP.axis('off')
                bx = {'IMU': (0.12, 0.72, C_SEL), 'KISS-IMU': (0.48, 0.72, C_LRN), 'PGO': (0.86, 0.46, C_RAW),
                      'LiDAR': (0.12, 0.20, MUTED)}
                for name, (x, y, c) in bx.items():
                    wd = 0.26 if name == 'KISS-IMU' else 0.2
                    axP.add_patch(FancyBboxPatch((x - wd / 2, y - 0.09), wd, 0.18,
                                                 boxstyle='round,pad=0,rounding_size=0.04', fc=BG, ec=c, lw=1.6))
                    axP.text(x, y, name, ha='center', va='center', color=c, fontsize=9.5, fontweight='bold')
                paths = [((0.22, 0.72), (0.35, 0.72), C_SEL), ((0.61, 0.72), (0.86, 0.72), C_LRN),
                         ((0.86, 0.72), (0.86, 0.55), C_LRN), ((0.22, 0.20), (0.86, 0.20), MUTED),
                         ((0.86, 0.20), (0.86, 0.37), MUTED)]
                axP._paths = []
                for p0, p1, c in paths:
                    axP.plot([p0[0], p1[0]], [p0[1], p1[1]], color=c, lw=1.2, alpha=0.35)
                    axP._paths.append((np.array(p0), np.array(p1), axP.scatter([], [], s=14, color=c, zorder=5)))
                axP.text(0.74, 0.78, 'learned IO', color=C_LRN, fontsize=7, ha='center')
                axP.text(0.29, 0.78, 'raw', color=C_SEL, fontsize=7, ha='center')
                axP.text(0.50, 0.25, 'scans', color=MUTED, fontsize=7, ha='center')
            u = (np.arange(3) / 3 + f * 0.04) % 1.0
            for p0, p1, s in axP._paths:
                s.set_offsets(p0[None] + u[:, None] * (p1 - p0)[None])
            draw_map(tc)
            if att_raw is not None:
                e = ease(q * 3)
                row(3, 'IMU-only attitude drift over 60 s', f'{att_raw:.0f}° → {att_lrn:.0f}°',
                    e * att_lrn / att_raw + (1 - e), C_LRN, ghost=1.0)
            k0 = N1 + N2 + N3 + N4 + N5 - 25
            if f >= k0:
                al = min(1, (f - k0) / 10)
                pill.set_alpha(al); pill_t.set_alpha(al)
                pill_t.set_text(f'trained on {W:.0f} s of its own run · no GT' +
                                (f' · ATE {ate_raw:.3f} → {ate_lrn:.3f} m' if ate_raw is not None else ''))
        return []

    anim = FuncAnimation(fig, frame, frames=total, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=a.fps), savefig_kwargs={'facecolor': BG})
    print('saved', a.out, 'K', gm.K, 'rare', rare, s_raw.round(3), s_bal.round(3), att_raw, att_lrn, ate_raw, ate_lrn)


if __name__ == '__main__':
    main()
