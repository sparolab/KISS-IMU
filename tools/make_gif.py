"""Story of one online KISS-IMU run in one animation (map on the left, the step on the right):

    1. Collect   - the first warmup_s seconds on the raw IMU: every 0.2 s IMU window leaves the robot's path
                   and lands in the motion-feature plane (|gravity-free acc|, |gyro|)
    2. Group     - a GMM (K by BIC) splits the plane into motion regimes; the path takes their colours
    3. Balance   - each regime becomes a bubble sized by its share of the training signal, then resized by
                   the effective-number weights (rare motion grows)
    4. Learn     - the balanced regimes stream into the denoiser
    5. Run       - the running PGO takes the denoised IMU (learned IO); the map is built to the end, the path
                   coloured by where the PGO leans on LiDAR (GICP Hessian) vs the IMU (its sample variance)
    6. Denoised  - gyro-only dead reckoning (60 s, LiDAR speed) from a few points of the run: raw vs denoised,
                   and the IMU-only attitude drift

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
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402

from kiss_imu.config import load_config  # noqa: E402
from kiss_imu.data import Sequence  # noqa: E402
from kiss_imu.eval.drift import integrate_gyro  # noqa: E402
from kiss_imu.training.reweight import MotionGMM, features_fn  # noqa: E402

BG, FG, MUTED, LIGHT, MAPC = '#ffffff', '#1f2328', '#59636e', '#d0d7de', '#b4bcc5'
C_RAW, C_LRN, C_SEL, C_NET = '#D81B60', '#00897B', '#D81B60', '#3F51B5'
PAL = ['#3F51B5', '#00897B', '#F57C00', '#8E24AA', '#43A047', '#6D4C41']


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def use_font():
    for d in ('/usr/share/fonts/opentype/inter', '/usr/share/fonts/host/inter', os.path.expanduser('~/.fonts')):
        for f in glob.glob(os.path.join(d, 'Inter-*.otf')) + glob.glob(os.path.join(d, 'Inter*.ttf')):
            font_manager.fontManager.addfont(f)
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family'] = 'Inter' if 'Inter' in names else 'DejaVu Sans'


def corrected_gyro(ckpt, seq):
    from kiss_imu.models import load_model
    from kiss_imu.pgo.modules.io_module import causal_correction
    m, ck = load_model(ckpt, 'cpu')
    return causal_correction(m, seq.gyr, seq.acc, ck['window']['L'], ck['window']['K'], 'cpu')


def dead_reckon(seq, gyr, ref, t0, H):
    """Gyro-only dead reckoning: the attitude from integrating gyr from the reference attitude at t0, the
    body-frame velocity from the reference (LiDAR PGO) trajectory. Returns reference and dead-reckoned xy."""
    ts, P, sl = ref
    k = (seq.imu_t >= t0) & (seq.imu_t <= t0 + H)
    t = seq.imu_t[k][::2]
    Rr = sl(t)
    p = np.stack([np.interp(t, ts, P[:, i]) for i in range(3)], 1)
    vb = Rr.inv().apply(np.gradient(p, t, axis=0))
    Re = Rr[0].as_matrix() @ integrate_gyro(t, gyr[k][::2])
    pe = p[0] + np.cumsum(np.einsum('nij,nj->ni', Re, vb) * np.r_[np.diff(t), 0][:, None], 0)
    return t - t0, p[:, :2], pe[:, :2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('seq'); ap.add_argument('run')
    ap.add_argument('--train', required=True, help='dir with the checkpoint (*.pt) and history.json')
    ap.add_argument('--raw', help='raw-IMU run dir (result.json) for the ATE line')
    ap.add_argument('--out', default='fig/kiss_imu_online.gif')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--dr-starts', type=float, nargs='+', help='default: the three spots of step 5')
    ap.add_argument('--dr-horizon', type=float, default=60.0)
    ap.add_argument('--map-every', type=int, default=3)
    ap.add_argument('--points', type=int, default=900)
    ap.add_argument('--zmax', type=float, default=3.0)
    ap.add_argument('--png', type=int, nargs='*', help='only save these frames as PNGs (layout check)')
    ap.add_argument('--set', nargs='*', default=[], dest='overrides')
    a = ap.parse_args()
    use_font()
    cfg = load_config(a.config, a.overrides)
    d, w = cfg['data'], cfg['warmstart']
    tpath = w['train_config'] if os.path.exists(w['train_config']) else os.path.join(os.path.dirname(a.config), w['train_config'])
    rw = load_config(tpath).get('reweight', {})
    win_s = rw.get('win_s', 0.2)
    seq = Sequence(d['root'], a.seq, d.get('type', 'generic'), 0.0, min_range=d.get('min_range', 1.0),
                   max_range=d.get('max_range', 100.0), scan_time_offset=d.get('scan_time_offset', 0.0),
                   lidar_rotvec_deg=d.get('lidar_rotvec_deg'))
    on = np.loadtxt(os.path.join(a.run, 'online.txt'))
    t, Pw, Rq = on[:, 0], on[:, 1:4], Rotation.from_quat(on[:, 4:8])
    t0 = t[0]
    origin = Pw[0].copy()
    P = Pw - origin
    W = float(w['warmup_s'])
    tt = t - t0
    T = tt[-1]
    res = json.load(open(os.path.join(a.run, 'result.json')))
    drift = res.get('imu_drift_after_switch', {})
    att_raw = drift.get('raw', {}).get('att_deg', {}).get('60.0', np.nan)
    att_lrn = drift.get('used', {}).get('att_deg', {}).get('60.0', np.nan)
    ate_lrn = res['online']['ate_aligned_rmse']
    ate_raw = json.load(open(os.path.join(a.raw, 'result.json')))['online']['ate_aligned_rmse'] if a.raw else None
    ckpt = sorted(glob.glob(os.path.join(a.train, '*.pt')))[0]
    hp = os.path.join(a.train, 'history.json')
    L = np.array([h['train_loss'] for h in json.load(open(hp))]) if os.path.exists(hp) else np.ones(2)

    # map points
    k_scan = np.searchsorted(seq.scan_t, t).clip(0, len(seq.scan_t) - 1)
    rng = np.random.default_rng(0)
    mp, mt = [], []
    for i in range(0, len(t), a.map_every):
        x = seq.scan(k_scan[i])
        x = x[rng.choice(len(x), min(a.points, len(x)), replace=False)]
        x = x[x[:, 2] < a.zmax]
        mp.append(Rq[i].apply(x) + P[i]); mt.append(np.full(len(x), tt[i]))
    mp, mt = np.concatenate(mp)[:, :2], np.concatenate(mt)

    # motion features + GMM of the warm-up (as the trainer)
    m = (seq.imu_t >= t0) & (seq.imu_t <= t0 + W)
    F = features_fn(rw)(seq.imu_t, seq.gyr, seq.acc, win_s)[m]
    ti = seq.imu_t[m] - t0
    gm = MotionGMM(rw.get('n_components'), tuple(rw.get('k_range', (2, 5))), win_s,
                   rw.get('method', 'effective'), rw.get('beta', 0.999), tuple(rw.get('clamp', (0.1, 10.0))),
                   seed=rw.get('seed', 0)).fit([F])
    Pr = gm.proba(F)
    mu = gm.scaler.inverse_transform(gm.gmm.means_)
    order = np.argsort(mu[:, 1] + 0.05 * mu[:, 0])                    # calm -> fast turns
    rank = np.empty_like(order); rank[order] = np.arange(gm.K)
    comp = rank[Pr.argmax(1)]
    K = gm.K
    counts = Pr.sum(0)[order]
    cw = np.clip(gm.component_weights(Pr.sum(0)), *gm.clamp)[order]
    s_raw, s_bal = counts / counts.sum(), counts * cw / (counts * cw).sum()
    rare = int(np.argmin(s_raw))
    mu_o = mu[order]
    names = ['calm'] * K
    names[K - 1] = 'fast turns'
    if K > 2:
        j = int(np.argmax(mu_o[1:K - 1, 0])) + 1
        names[j] = 'bumpy'
        for k in range(1, K - 1):
            if k != j:
                names[k] = 'cruising'
    # 0.2 s windows of the warm-up
    wc = np.arange(win_s / 2, W, win_s)
    iw = np.searchsorted(ti, wc).clip(0, len(ti) - 1)
    Fw, cwin = F[iw], comp[iw]
    pw = np.stack([np.interp(wc, tt, P[:, i]) for i in range(2)], 1)

    # denoised gyro + dead reckoning against the (GT-free) smoothed PGO trajectory
    g_c, a_c = corrected_gyro(ckpt, seq)
    # where the PGO leans on LiDAR vs IMU: per-scan position information of the two factors
    #   LiDAR: the logged LO information (GICP Hessian), its weakest translation direction
    #   IMU:   white specific-force noise over the scan interval from the stream's own sample variance
    #          (io.sample_var) on the stream the PGO used (denoised after the hand-over)
    from kiss_imu.imu.noise import diff_var
    io_c = cfg['io']
    t_sw = t0 + float(res.get('warmstart', {}).get('t_switch') or W)     # t_switch is relative to the first scan
    acc_used = np.where((seq.imu_t >= t_sw)[:, None], a_c, seq.acc)
    sv = io_c.get('sample_var') or {}
    av = np.asarray(diff_var(seq.imu_t, acc_used, io_c['acc_noise'], float(sv.get('win_s', 0.1)), int(sv.get('order', 1))))
    av = av.mean(1) if av.ndim > 1 else av
    steps = [json.loads(x) for x in open(os.path.join(a.run, 'steps.jsonl'))]
    st_t = np.array([s_['t'] for s_ in steps])
    cum = np.r_[0, np.cumsum(av)]
    ii = np.searchsorted(seq.imu_t, st_t)
    trust = np.full(len(steps), np.nan)
    for i in range(1, len(steps)):
        info = steps[i].get('lo_info')
        if info is None or ii[i] <= ii[i - 1]:
            continue
        wl = np.linalg.eigvalsh(np.array(info).reshape(6, 6)[3:, 3:]).min()
        dt = st_t[i] - st_t[i - 1]
        var_p = (cum[ii[i]] - cum[ii[i - 1]]) / (ii[i] - ii[i - 1]) * dt * dt ** 3 / 3
        trust[i] = np.log10(wl * var_p)                                  # > 0: LiDAR weighs more
    ok = np.isfinite(trust)
    trust = np.interp(np.arange(len(trust)), np.where(ok)[0], trust[ok])
    trust = np.convolve(np.pad(trust, 15, mode='edge'), np.ones(31) / 31, mode='valid')   # ~3 s
    tr_t = st_t - t0
    tr_xy = np.stack([np.interp(tr_t, tt, P[:, i]) for i in range(2)], 1)
    # the absolute ratio leans on the IMU's white-noise model; show where the PGO leans on LiDAR more or less
    # than it usually does on this run (relative to the median), scaled to the run's spread
    lo5, hi95 = np.percentile(trust, [5, 95])
    spread = float(10 ** (hi95 - lo5))
    share_l = 1.0 / (1.0 + 10 ** (-(trust - np.median(trust))))           # LiDAR share vs the run's typical mix
    trust = (trust - np.median(trust)) / max(1e-6, (hi95 - lo5) / 2)
    print(f'trust: LiDAR/IMU weight ratio spread x{spread:.1f} (5-95 %)')
    # three spots to zoom into: where the PGO leans most on LiDAR, most on the IMU, and a typical one
    cand = np.where((tr_t > W + 15) & (tr_t < T - a.dr_horizon - 5))[0]
    i_hi, i_lo = cand[np.argmax(share_l[cand])], cand[np.argmin(share_l[cand])]
    far = cand[(np.abs(tr_t[cand] - tr_t[i_hi]) > 40) & (np.abs(tr_t[cand] - tr_t[i_lo]) > 40)]
    i_md = far[np.argmin(np.abs(share_l[far] - 0.5))] if len(far) else cand[len(cand) // 2]
    spots = sorted([int(i_hi), int(i_lo), int(i_md)], key=lambda i: tr_t[i])
    if not a.dr_starts:
        a.dr_starts = [float(tr_t[i]) for i in spots]
    print('spots', [(round(float(tr_t[i])), round(float(share_l[i]), 2)) for i in spots])
    sm = np.loadtxt(os.path.join(a.run, 'smoothed.txt'))
    ref = (sm[:, 0], sm[:, 1:4] - origin, Slerp(sm[:, 0], Rotation.from_quat(sm[:, 4:8])))
    DR = []
    for s0 in a.dr_starts:
        tr, pr, prw = dead_reckon(seq, seq.gyr, ref, t0 + s0, a.dr_horizon)
        _, _, pcd = dead_reckon(seq, g_c, ref, t0 + s0, a.dr_horizon)
        DR.append((tr, pr, prw, pcd))
        print(f'DR {s0:.0f}s: raw end err {np.linalg.norm(prw[-1] - pr[-1]):.1f} m, '
              f'denoised {np.linalg.norm(pcd[-1] - pr[-1]):.1f} m')

    # ---- figure
    FW, FH = 9.6, 5.4
    fig = plt.figure(figsize=(FW, FH), dpi=100)
    fig.patch.set_facecolor(BG)
    MAP = [0.01, 0.07, 0.535, 0.80]
    axM = fig.add_axes(MAP); axM.axis('off')
    r = (MAP[2] * FW) / (MAP[3] * FH)

    def fit(lo, hi):
        c, span = (lo + hi) / 2, hi - lo
        sx = max(span[0], span[1] * r)
        return np.array([c[0] - sx / 2, c[0] + sx / 2, c[1] - sx / r / 2, c[1] + sx / r / 2])

    i_w = int(np.searchsorted(tt, W))
    LIM_FULL = fit(P[:, :2].min(0) - 10.0, P[:, :2].max(0) + 10.0)
    LIM_WARM = fit(P[:i_w + 1, :2].min(0) - 12.0, P[:i_w + 1, :2].max(0) + 12.0)

    def set_lim(L4):
        axM.set_xlim(L4[0], L4[1]); axM.set_ylim(L4[2], L4[3])

    set_lim(LIM_WARM)
    WS = [0.625, 0.25, 0.345, 0.52]                                      # workspace (feature plane)
    axW = fig.add_axes(WS)
    axO = fig.add_axes([0, 0, 1, 1]); axO.set_xlim(0, 1); axO.set_ylim(0, 1); axO.axis('off'); axO.patch.set_alpha(0)
    axO.set_zorder(5)
    xmax, ymax = np.percentile(F[:, 0], 99.5) * 1.08, np.percentile(F[:, 1], 99.5) * 1.08
    axW.set_xlim(0, xmax); axW.set_ylim(0, ymax)
    axW.set_facecolor(BG)
    for s in axW.spines.values():
        s.set_color(LIGHT)
    axW.spines['top'].set_visible(False); axW.spines['right'].set_visible(False)
    axW.tick_params(colors=MUTED, labelsize=7, length=2)
    axW.set_xlabel('moving  |a − g|  [m/s²]', color=MUTED, fontsize=8.5)
    axW.set_ylabel('turning  |ω|  [rad/s]', color=MUTED, fontsize=8.5)
    to_fig = fig.transFigure.inverted()

    def fpt(ax, xy):
        return to_fig.transform(ax.transData.transform(np.atleast_2d(xy)))

    # map artists
    sc = axM.scatter(np.zeros(0), np.zeros(0), s=0.15, color=MAPC, linewidths=0, alpha=0.6)
    seg_sel = LineCollection([], linewidths=3.0, capstyle='round', zorder=4)
    axM.add_collection(seg_sel)
    ln_lrn, = axM.plot([], [], color=C_LRN, lw=2.2, solid_capstyle='round', zorder=3)
    head = axM.scatter([], [], s=46, color=FG, zorder=6, edgecolors=BG, linewidths=1.4)
    dr_art = []
    for _ in DR:
        r0, = axM.plot([], [], color=FG, lw=1.4, alpha=0.6, zorder=7)
        r1, = axM.plot([], [], color=C_RAW, lw=2.0, ls=(0, (3, 1.5)), zorder=8)
        r2, = axM.plot([], [], color=C_LRN, lw=2.4, zorder=9)
        st = axM.scatter([], [], s=50, color=FG, zorder=10, edgecolors=BG, linewidths=1.4)
        dr_art.append((r0, r1, r2, st))
    clock = axO.text(0.02, 0.905, '', color=MUTED, fontsize=8.5, va='center')
    leg = axO.text(0.02, 0.955, '', color=FG, fontsize=9, va='center', fontweight='bold')
    axO.plot([0.56, 0.56], [0.04, 0.96], color=LIGHT, lw=0.8)

    # sidebar text
    X0 = 0.59
    step = axO.text(X0, 0.955, '', color=MUTED, fontsize=8, va='center', fontweight='bold')
    title = axO.text(X0, 0.905, '', color=FG, fontsize=16, fontweight='bold', va='center')
    subt = axO.text(X0, 0.845, '', color=MUTED, fontsize=9, va='center', linespacing=1.35)
    big = axO.text(X0, 0.10, '', color=FG, fontsize=22, fontweight='bold', va='center')
    bigl = axO.text(X0, 0.045, '', color=MUTED, fontsize=8.5, va='center')

    # feature plane artists
    pts = axW.scatter(np.zeros(0), np.zeros(0), s=9, linewidths=0, color=MUTED, zorder=3)
    ells = []
    for k in range(K):
        j = order[k]
        C = gm.scaler.scale_[:, None] * gm.gmm.covariances_[j] * gm.scaler.scale_[None]
        ev, V = np.linalg.eigh(C)
        e = Ellipse(mu[j], 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0]), angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])),
                    fc=to_rgba(PAL[k], 0), ec=to_rgba(PAL[k], 0), lw=1.8, zorder=2)
        axW.add_patch(e); ells.append(e)
    elab = [axW.text(*mu_o[k], f'M{k + 1}', color='white', fontsize=8.5, fontweight='bold', ha='center', va='center',
                     zorder=6, alpha=0, bbox=dict(boxstyle='round,pad=0.25', fc=PAL[k], ec='none', alpha=0))
            for k in range(K)]
    chips = axO.scatter(np.zeros(0), np.zeros(0), s=22, marker='s', linewidths=0, zorder=8)
    ctr = fpt(axW, mu_o)                                                # regime centres in figure coords

    # bubbles / net (figure coords)
    bub = axO.scatter(np.zeros(K), np.zeros(K), s=np.zeros(K), color=[PAL[k] for k in range(K)], zorder=6,
                      edgecolors=BG, linewidths=1.5, alpha=0)
    bub_t = [axO.text(0, 0, '', color=FG, fontsize=8.5, fontweight='bold', ha='center', va='top', zorder=7) for _ in range(K)]
    bub_n = [axO.text(0, 0, '', color=MUTED, fontsize=7.5, ha='center', va='top', zorder=7) for _ in range(K)]
    layers = [4, 6, 6, 3]
    NX = np.linspace(0.745, 0.925, len(layers))
    nodes = [np.stack([np.full(n, x), 0.52 + (np.arange(n) - (n - 1) / 2) * 0.062], 1) for x, n in zip(NX, layers)]
    edges, ecol = [], []
    for li in range(len(layers) - 1):
        for p0 in nodes[li]:
            for p1 in nodes[li + 1]:
                edges.append([p0, p1]); ecol.append(li)
    ecol = np.array(ecol)
    net_e = LineCollection(edges, colors=[to_rgba(LIGHT, 0)] * len(edges), linewidths=0.9, zorder=4)
    axO.add_collection(net_e)
    allnodes = np.concatenate(nodes)
    ncol = np.concatenate([np.full(n, i) for i, n in enumerate(layers)])
    net_n = axO.scatter(allnodes[:, 0], allnodes[:, 1], s=60, color=BG, edgecolors=C_NET, linewidths=1.6, zorder=5, alpha=0)
    net_box = FancyBboxPatch((NX[0] - 0.03, 0.30), NX[-1] - NX[0] + 0.06, 0.44, boxstyle='round,pad=0,rounding_size=0.02',
                             fc='none', ec=C_NET, lw=1.2, ls='--', alpha=0, zorder=3)
    axO.add_patch(net_box)
    net_lab = axO.text((NX[0] + NX[-1]) / 2, 0.76, 'KISS-IMU denoiser', color=C_NET, fontsize=9, fontweight='bold',
                       ha='center', alpha=0)
    io_in = axO.text(0.625, 0.52, 'raw\nIMU', color=C_RAW, fontsize=9, fontweight='bold', ha='center', va='center', alpha=0)
    io_out = axO.text(0.975, 0.52, 'PGO', color=C_LRN, fontsize=9, fontweight='bold', ha='center', va='center', alpha=0,
                      bbox=dict(boxstyle='round,pad=0.35', fc=BG, ec=C_LRN, lw=1.4, alpha=0))
    parts = axO.scatter(np.zeros(0), np.zeros(0), s=16, linewidths=0, zorder=8)
    # loss sparkline
    axL = fig.add_axes([0.80, 0.045, 0.17, 0.11]); axL.axis('off'); axL.set_xlim(0, len(L)); axL.set_ylim(min(L) * 0.9, max(L) * 1.05)
    loss_ln, = axL.plot([], [], color=C_NET, lw=1.8)
    # attitude dials (step 6)
    axG = fig.add_axes([0.60, 0.25, 0.38, 0.52]); axG.set_xlim(0, 2.2); axG.set_ylim(-0.25, 1.25); axG.set_aspect('equal')
    axG.axis('off'); axG.set_visible(False)
    dials = []
    for k, (cx, col, nm) in enumerate([(0.5, C_RAW, 'raw IMU'), (1.7, C_LRN, 'KISS-IMU')]):
        axG.add_patch(Wedge((cx, 0.15), 0.5, 0, 180, width=0.02, color=LIGHT))
        axG.plot([cx, cx], [0.15, 0.65], color=FG, lw=1.0, ls=':', alpha=0.6)
        wd = Wedge((cx, 0.15), 0.5, 90, 90, width=0.5, color=to_rgba(col, 0.15))
        axG.add_patch(wd)
        ar = FancyArrowPatch((cx, 0.15), (cx, 0.65), arrowstyle='-|>', mutation_scale=16, color=col, lw=2.6)
        axG.add_patch(ar)
        val = axG.text(cx, -0.08, '', color=col, fontsize=15, fontweight='bold', ha='center', va='center')
        axG.text(cx, -0.22, nm, color=MUTED, fontsize=8.5, ha='center', va='center')
        dials.append((cx, wd, ar, val))
    axG.text(1.1, 0.9, 'IMU-only attitude drift over 60 s (RMSE)', color=MUTED, fontsize=8.5, ha='center')

    # step 5 spots: zoom into the map, the local pose graph (nodes 1 s apart; LiDAR factor on one side, IMU
    # factor on the other, width = its share of the weight) and the share as numbers
    C_LID, C_IMU = '#3F51B5', '#F57C00'
    ZW = 15.0

    def zoom_lim(cxy):
        h = np.array([ZW / 2, ZW / 2 / r])
        return fit(cxy - h, cxy + h)

    loc_lc = LineCollection([], capstyle='round', zorder=8)
    axM.add_collection(loc_lc)
    loc_nodes = axM.scatter(np.zeros(0), np.zeros(0), s=70, color=BG, edgecolors=FG, linewidths=1.4, zorder=9)
    badges = [axM.text(tr_xy[i, 0], tr_xy[i, 1] + 4.0, f'LiDAR {100 * share_l[i]:.0f}%', color=C_LID, fontsize=7.5,
                       fontweight='bold', ha='center', va='bottom', alpha=0, zorder=11,
                       clip_on=True, bbox=dict(boxstyle='round,pad=0.25', fc=BG, ec=C_LID, lw=0.8, alpha=0)) for i in spots]
    sp_title = axO.text(0.59, 0.255, '', color=FG, fontsize=9, fontweight='bold', va='center', alpha=0)
    sp_l = axO.text(0.59, 0.19, '', color=C_LID, fontsize=16, fontweight='bold', va='center', alpha=0)
    sp_i = axO.text(0.97, 0.19, '', color=C_IMU, fontsize=16, fontweight='bold', va='center', ha='right', alpha=0)
    sp_bl = Rectangle((0.59, 0.115), 0, 0.03, color=C_LID, alpha=0)
    sp_bi = Rectangle((0.59, 0.115), 0, 0.03, color=C_IMU, alpha=0)
    axO.add_patch(sp_bl); axO.add_patch(sp_bi)
    sp_c = axO.text(0.59, 0.07, "PGO weighting here, relative to this run's typical LiDAR : IMU mix", color=MUTED,
                    fontsize=7.6, va='center', alpha=0)
    leg_l = axO.text(0.02, 0.955, '— LiDAR factor', color=C_LID, fontsize=9, fontweight='bold', va='center', alpha=0)
    leg_i = axO.text(0.14, 0.955, '— IMU factor    (width = weight)', color=C_IMU, fontsize=9, fontweight='bold',
                     va='center', alpha=0)

    def local_graph(i0, g):
        tn = tr_t[i0] + np.arange(-4, 5) * 1.0
        k = np.searchsorted(tr_t, tn).clip(0, len(tr_t) - 1)
        xy = tr_xy[k]
        d = np.gradient(xy, axis=0); d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) * 0.45
        ne = max(0, int(g * (len(xy) - 1)))
        segs, lw, col = [], [], []
        for e in range(ne):
            sh = float(share_l[(k[e] + k[e + 1]) // 2])
            segs.append([xy[e] + nrm[e], xy[e + 1] + nrm[e + 1]]); lw.append(1.0 + 12.0 * sh); col.append(to_rgba(C_LID, 0.9))
            segs.append([xy[e] - nrm[e], xy[e + 1] - nrm[e + 1]]); lw.append(1.0 + 12.0 * (1 - sh)); col.append(to_rgba(C_IMU, 0.9))
        loc_lc.set_segments(segs); loc_lc.set_linewidths(lw if lw else [1]); loc_lc.set_color(col if col else [to_rgba(BG, 0)])
        loc_nodes.set_offsets(xy[:ne + 1] if g > 0 else np.zeros((0, 2)))

    def spot_card(al, k=None):
        for x in (sp_title, sp_l, sp_i, sp_bl, sp_bi, sp_c):
            x.set_alpha(al)
        if k is not None:
            i = spots[k]; sh = float(share_l[i])
            sp_title.set_text(f'spot {k + 1} / {len(spots)}   t = {tr_t[i]:.0f} s')
            sp_l.set_text(f'LiDAR {100 * sh:.0f}%'); sp_i.set_text(f'IMU {100 * (1 - sh):.0f}%')
            sp_bl.set_width(0.38 * sh); sp_bi.set_x(0.59 + 0.38 * sh); sp_bi.set_width(0.38 * (1 - sh))

    pill = FancyBboxPatch((0.015, 0.025), 0.53, 0.055, boxstyle='round,pad=0,rounding_size=0.012', fc='#e0f2f1',
                          ec='none', alpha=0, zorder=9)
    axO.add_patch(pill)
    pill_t = axO.text(0.03, 0.0525, '', color=C_LRN, fontsize=9, fontweight='bold', va='center', zorder=10)

    NBLD, NSP = 95, 40                                                    # step 5: build, then each spot
    NA, NB, NC, ND, NE, NF, HOLD = 85, 55, 60, 65, NBLD + 3 * NSP, 95, 40
    S = np.cumsum([0, NA, NB, NC, ND, NE, NF])
    total = int(S[-1] + HOLD)
    FLY = 14
    i_sel = int(np.searchsorted(tt, W))
    src = fpt(axM, pw)
    dst = fpt(axW, Fw)
    regime_col = np.array([to_rgba(PAL[c]) for c in cwin])
    grey = np.array(to_rgba('#8a939d'))
    row_y = np.linspace(0.70, 0.32, K)                                    # bubble column in step 4
    row_x = np.linspace(0.66, 0.94, K)                                    # bubble row in step 3
    SZ = 5200.0                                                            # marker area (pt^2) for share 1
    prt = {'p0': np.zeros((0, 2)), 'p1': np.zeros((0, 2)), 'age': np.zeros(0), 'col': np.zeros((0, 4)), 'dur': np.zeros(0)}
    prng = np.random.default_rng(3)
    comp_scan = comp[np.searchsorted(ti, tt[:i_sel + 1]).clip(0, len(ti) - 1)]
    path_reg = np.array([to_rgba(PAL[c]) for c in comp_scan])

    def map_to(tc, path_cols=None):
        sc.set_offsets(mp[mt <= tc])
        j = int(np.searchsorted(tt, tc))
        js = min(j, i_sel)
        pts_ = P[:js + 1, :2]
        segs = np.stack([pts_[:-1], pts_[1:]], 1) if js > 0 else np.zeros((0, 2, 2))
        seg_sel.set_segments(segs)
        seg_sel.set_color(path_cols[:max(js, 0)] if path_cols is not None else C_SEL)
        if j > i_sel:
            ln_lrn.set_data(P[i_sel:j + 1, 0], P[i_sel:j + 1, 1])
        head.set_offsets([P[min(j, len(P) - 1), :2]])
        clock.set_text(f'{a.seq}    t = {tc:5.1f} s / {T:.0f} s')

    def text(n, ttl, st):
        step.set_text(f'STEP {n} / 6'); title.set_text(ttl); subt.set_text(st)

    def spawn(p0, p1, col, dur):
        prt['p0'] = np.vstack([prt['p0'], p0]); prt['p1'] = np.vstack([prt['p1'], p1])
        prt['col'] = np.vstack([prt['col'], col]); prt['age'] = np.r_[prt['age'], np.zeros(len(p0))]
        prt['dur'] = np.r_[prt['dur'], np.full(len(p0), dur)]

    def step_particles():
        prt['age'] += 1
        keep = prt['age'] <= prt['dur']
        for k in prt:
            prt[k] = prt[k][keep]
        u = ease(prt['age'] / np.maximum(prt['dur'], 1))[:, None]
        pos = (1 - u) * prt['p0'] + u * prt['p1']
        parts.set_offsets(pos if len(pos) else np.zeros((0, 2)))
        if len(pos):
            parts.set_color(prt['col'])

    def net_wave(f, alpha=1.0):
        ph = (f * 0.12) % (len(layers) + 1)
        lit = np.exp(-((ecol + 0.5 - ph) ** 2) / 0.4)
        net_e.set_color(np.array([to_rgba(C_NET, alpha * (0.12 + 0.75 * v)) for v in lit]))
        nl = np.exp(-((ncol - ph) ** 2) / 0.4)
        net_n.set_facecolor([to_rgba(C_NET, 0.15 + 0.85 * v) for v in nl])

    def show_net(al):
        net_n.set_alpha(al); net_box.set_alpha(al); net_lab.set_alpha(al)
        if al == 0:
            net_e.set_color([to_rgba(LIGHT, 0)] * len(edges))

    def frame(f):
        if f < S[1]:                                                       # 1. collect
            q = min(1.0, (f + 1) / (0.8 * NA))
            tc = q * W
            text(1, 'Collect', f'the robot runs the first {W:.0f} s on the raw IMU;\n'
                               f'every {win_s:.1f} s IMU window becomes one motion point')
            leg.set_text('— first 30 s on the raw IMU (training data)'); leg.set_color(C_SEL)
            map_to(tc)
            ft = int(np.searchsorted(wc, tc))                                # windows emitted so far
            age = f - (wc[:ft] / W) * 0.8 * NA
            fly = age < FLY
            landed = ~fly
            u = ease(age[fly] / FLY)[:, None]
            pos = (1 - u) * src[:ft][fly] + u * dst[:ft][fly]
            pos[:, 1] += 0.10 * np.sin(np.pi * u[:, 0])
            chips.set_offsets(pos if len(pos) else np.zeros((0, 2))); chips.set_color('#8a939d')
            pts.set_offsets(Fw[:ft][landed] if landed.any() else np.zeros((0, 2))); pts.set_color('#8a939d')
            big.set_text(f'{int(landed.sum())}'); big.set_color(FG)
            bigl.set_text(f'motion points from {min(tc, W):4.1f} s of IMU  (GT-free)')
        elif f < S[2]:                                                     # 2. group
            q = ease((f - S[1] + 1 - 5) / (0.55 * NB))
            chips.set_offsets(np.zeros((0, 2)))
            text(2, 'Group', 'a GMM finds the motion regimes (K by BIC);\nthe path takes their colours')
            pts.set_offsets(Fw); pts.set_color((1 - q) * grey + q * regime_col)
            for k, e in enumerate(ells):
                e.set_facecolor(to_rgba(PAL[k], 0.10 * q)); e.set_edgecolor(to_rgba(PAL[k], q))
            for k, tx in enumerate(elab):
                tx.set_alpha(q); tx.get_bbox_patch().set_alpha(q)
                tx.set_text(f'M{k + 1}  {names[k]}')
            leg.set_text('— path coloured by motion regime'); leg.set_color(FG)
            map_to(W, (1 - q) * np.array(to_rgba(C_SEL)) + q * path_reg)
            big.set_text(f'K = {K}'); bigl.set_text(f'motion regimes in the first {W:.0f} s')
        elif f < S[3]:                                                     # 3. balance
            fl = f - S[2]
            q1 = ease((fl + 1) / 20)                                         # plane -> bubbles
            q2 = ease((fl + 1 - 26) / 22)                                    # raw -> balanced
            text(3, 'Balance', 'each regime\'s share of the training signal:\nrare motion is weighted up (effective number)')
            map_to(W, path_reg)
            pts.set_alpha(1 - q1)
            for k, e in enumerate(ells):
                e.set_facecolor(to_rgba(PAL[k], 0.10 * (1 - q1))); e.set_edgecolor(to_rgba(PAL[k], 1 - q1))
            for tx in elab:
                tx.set_alpha(1 - q1); tx.get_bbox_patch().set_alpha(1 - q1)
            pos = (1 - q1) * ctr + q1 * np.stack([row_x, np.full(K, 0.55)], 1)
            sh = (1 - q2) * s_raw + q2 * s_bal
            bub.set_offsets(pos); bub.set_sizes(SZ * sh * q1 + 1); bub.set_alpha(min(1, q1 * 1.5))
            for k in range(K):
                rr = np.sqrt(SZ * sh[k] / np.pi) / 72 / FH                   # marker radius in figure height
                bub_t[k].set_position((pos[k, 0], 0.55 - rr - 0.02)); bub_t[k].set_text(f'{100 * sh[k]:.0f}%')
                bub_t[k].set_alpha(q1)
                bub_n[k].set_position((pos[k, 0], 0.55 - rr - 0.065))
                bub_n[k].set_text(f'M{k + 1} {names[k]}' + (f'\n×{cw[k]:.2f}' if q2 > 0.3 else ''))
                bub_n[k].set_alpha(q1)
            axW.set_visible(q1 < 0.99)
            big.set_text(f'{100 * s_raw[rare]:.0f}% → {100 * sh[rare]:.0f}%'); big.set_color(PAL[rare])
            bigl.set_text(f'share of the rarest regime (M{rare + 1}, {names[rare]})')
        elif f < S[4]:                                                     # 4. learn
            fl = f - S[3]
            q1 = ease((fl + 1) / 15)
            text(4, 'Learn', 'the balanced regimes train the denoiser\n(gyro + acc corrections, LiDAR labels)')
            axW.set_visible(False)
            pos = (1 - q1) * np.stack([row_x, np.full(K, 0.55)], 1) + q1 * np.stack([np.full(K, 0.635), row_y], 1)
            bub.set_offsets(pos); bub.set_sizes(SZ * (1 - 0.7 * q1) * s_bal + 1)
            for k in range(K):
                bub_t[k].set_alpha(1 - q1); bub_n[k].set_alpha(1 - q1)
            show_net(q1)
            net_wave(f, q1)
            if 10 < fl < ND - 8:
                ks = prng.choice(K, 3, p=s_bal)
                tgt = nodes[0][prng.integers(0, layers[0], 3)]
                spawn(pos[ks], tgt, np.array([to_rgba(PAL[k]) for k in ks]), 12)
            step_particles()
            qn = ease((fl + 1 - 10) / (ND - 18))
            n = max(1, int(qn * len(L)))
            loss_ln.set_data(np.arange(n), L[:n])
            big.set_text(f'{L[0]:.2f} → {L[n - 1]:.2f}'); big.set_color(C_NET); bigl.set_text('training loss')
        elif f < S[5]:                                                     # 5. run with the learned IO
            fl = f - S[4]
            q1 = ease((fl + 1) / 12)
            text(5, 'Run', 'the PGO switches to the denoised IMU, no restart;\nLiDAR weighed by its GICP Hessian, IMU by its own noise')
            bub.set_alpha(1 - q1)
            loss_ln.set_data([], [])
            big.set_text(''); bigl.set_text('')
            if fl < NBLD:
                q = min(1.0, (fl + 1) / (0.9 * NBLD))
                io_in.set_alpha(q1); io_out.set_alpha(q1); io_out.get_bbox_patch().set_alpha(q1)
                show_net(1.0); net_wave(f)
                if fl < NBLD - 6:
                    spawn(np.array([[0.65, 0.52]]), nodes[0][prng.integers(0, layers[0], 1)], np.array([to_rgba(C_RAW)]), 9)
                    spawn(nodes[-1][prng.integers(0, layers[-1], 1)], np.array([[0.955, 0.52]]), np.array([to_rgba(C_LRN)]), 9)
                step_particles()
                set_lim((1 - ease(fl / 18)) * LIM_WARM + ease(fl / 18) * LIM_FULL)
                tc = W + q * (T - W)
                map_to(tc, path_reg)
                leg.set_text('— learned IO in the PGO'); leg.set_color(C_LRN)
            else:
                k, u = divmod(fl - NBLD, NSP)
                i = spots[k]
                map_to(T, path_reg); head.set_alpha(0)
                show_net(0.25); net_wave(f, 0.25); step_particles()
                z = ease(u / 9) * (1 - ease((u - (NSP - 7)) / 7))
                set_lim((1 - z) * LIM_FULL + z * zoom_lim(tr_xy[i]))
                local_graph(i, ease((u - 7) / 16) if u < NSP - 7 else 0.0)
                spot_card(float(ease(u / 6)) if k == 0 else 1.0, k)
                leg.set_text(''); leg_l.set_alpha(z); leg_i.set_alpha(z)
                clock.set_text(f'{a.seq}    t = {tr_t[i]:5.1f} s')
                if u > NSP - 10:
                    badges[k].set_alpha(1); badges[k].get_bbox_patch().set_alpha(1)
        else:                                                              # 6. denoised
            fl = f - S[5]
            q1 = ease((fl + 1) / 12)
            q = ease((fl + 1 - 10) / (0.7 * NF))
            text(6, 'Denoised', f'gyro-only dead reckoning for {a.dr_horizon:.0f} s from a few points\n'
                                f'(speed from LiDAR): raw IMU vs KISS-IMU')
            show_net(1 - q1); net_wave(f, 1 - q1)
            io_in.set_alpha(1 - q1); io_out.set_alpha(1 - q1); io_out.get_bbox_patch().set_alpha(1 - q1)
            parts.set_offsets(np.zeros((0, 2)))
            if q1 >= 1:
                show_net(0)
            axG.set_visible(True)
            leg_l.set_alpha(0); leg_i.set_alpha(0); spot_card(1 - q1); local_graph(spots[0], 0.0)
            ln_lrn.set_color((1 - q1) * np.array(to_rgba(C_LRN)) + q1 * np.array(to_rgba('#c5ccd3')))
            seg_sel.set_alpha(1 - 0.7 * q1); head.set_alpha(1 - q1)
            leg.set_text('— reference (LiDAR)    - - raw IMU    — KISS-IMU'); leg.set_color(FG)
            clock.set_text(f'{a.seq}    dead reckoning  {q * a.dr_horizon:4.1f} s / {a.dr_horizon:.0f} s')
            for (tr, pr, prw, pcd), (r0, r1, r2, st) in zip(DR, dr_art):
                n = max(2, int(q * len(tr)))
                r0.set_data(pr[:n, 0], pr[:n, 1]); r1.set_data(prw[:n, 0], prw[:n, 1]); r2.set_data(pcd[:n, 0], pcd[:n, 1])
                st.set_offsets([pr[0]])
            for (cx, wd, ar, val), ang in zip(dials, [att_raw, att_lrn]):
                aq = q * ang
                wd.set_theta1(90 - aq); wd.set_theta2(90)
                th = np.radians(90 - aq)
                ar.set_positions((cx, 0.15), (cx + 0.5 * np.cos(th), 0.15 + 0.5 * np.sin(th)))
                val.set_text(f'{aq:.0f}°')
            big.set_text(''); bigl.set_text('')
            if fl > 0.75 * NF:
                al = min(1, (fl - 0.75 * NF) / 10)
                pill.set_alpha(al); pill_t.set_alpha(al)
                pill_t.set_text(f'trained on {W:.0f} s of its own run · no GT' +
                                (f' · ATE {ate_raw:.3f} → {ate_lrn:.3f} m' if ate_raw is not None else ''))
        return []

    if a.png:
        for f in range(max(a.png) + 1):
            frame(f)
            if f in a.png:
                fig.savefig(os.path.splitext(a.out)[0] + f'_{f:04d}.png', facecolor=BG)
        return
    anim = FuncAnimation(fig, frame, frames=total, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=a.fps), savefig_kwargs={'facecolor': BG})
    print('saved', a.out, 'K', K, names, s_raw.round(3), s_bal.round(3), att_raw, att_lrn)


if __name__ == '__main__':
    main()
