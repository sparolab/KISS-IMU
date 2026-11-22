"""Animated explainer of KISS-IMU's motion-balanced learning on a run's warm-up (GT-free, as in training):

    1. Windows  - the warm-up is cut into 0.2 s IMU windows (tiles on a timeline); each window becomes one
                  motion point (|mean gravity-free acc|, |mean gyro|) in the feature plane
    2. Regimes  - a GMM (K by BIC) groups the points; the timeline shows when each regime happened
    3. Imbalance- the tiles sort themselves into one stack per regime: calm motion dominates
    4. Balance  - effective-number weights rescale every tile of a regime; rare motion grows
    5. Batches  - what a training batch sees, uniform vs motion-balanced sampling

    python3 tools/make_gmm_gif.py configs/diter_os/warmstart.yaml Forest_new runs/warmstart/Forest_new \
        --out fig/gmm_balance.gif
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
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle  # noqa: E402

from kiss_imu.config import load_config  # noqa: E402
from kiss_imu.data import Sequence  # noqa: E402
from kiss_imu.training.reweight import MotionGMM, features_fn  # noqa: E402

BG, FG, MUTED, LIGHT = '#ffffff', '#1f2328', '#59636e', '#d0d7de'
GREY = '#8a939d'
PAL = ['#3F51B5', '#00897B', '#F57C00', '#8E24AA', '#43A047', '#6D4C41']
C_HI = '#D81B60'


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
    ap.add_argument('--out', default='fig/gmm_balance.gif')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--batch', type=int, default=20)
    ap.add_argument('--png', type=int, nargs='*', help='only save these frames as PNGs (layout check)')
    ap.add_argument('--set', nargs='*', default=[], dest='overrides')
    a = ap.parse_args()
    use_font()
    cfg = load_config(a.config, a.overrides)
    d, w = cfg['data'], cfg['warmstart']
    tpath = w['train_config'] if os.path.exists(w['train_config']) else os.path.join(os.path.dirname(a.config), w['train_config'])
    rw = load_config(tpath).get('reweight', {})
    win_s = rw.get('win_s', 0.2)
    seq = Sequence(d['root'], a.seq, d.get('type', 'generic'), 0.0, gt_file=d.get('gt_file', 'gt_pose.csv'))
    t0 = float(np.loadtxt(os.path.join(a.run, 'online.txt'), max_rows=1)[0])
    W = float(w['warmup_s'])

    m = (seq.imu_t >= t0) & (seq.imu_t <= t0 + W)
    F = features_fn(rw)(seq.imu_t, seq.gyr, seq.acc, win_s)[m]          # as the trainer
    ti = seq.imu_t[m] - t0
    gm = MotionGMM(rw.get('n_components'), tuple(rw.get('k_range', (2, 5))), win_s,
                   rw.get('method', 'effective'), rw.get('beta', 0.999), tuple(rw.get('clamp', (0.1, 10.0))),
                   seed=rw.get('seed', 0)).fit([F])
    Pr = gm.proba(F)
    mu = gm.scaler.inverse_transform(gm.gmm.means_)
    order = np.argsort(mu[:, 1] + 0.05 * mu[:, 0])
    rank = np.empty_like(order); rank[order] = np.arange(gm.K)
    K = gm.K
    cw = np.clip(gm.component_weights(Pr.sum(0)), *gm.clamp)[order]      # the trainer's regime weights
    mu_o = mu[order]
    names = ['calm'] * K
    names[K - 1] = 'fast turns'
    if K > 2:
        j = int(np.argmax(mu_o[1:K - 1, 0])) + 1
        names[j] = 'bumpy'
        for k in range(1, K - 1):
            if k != j:
                names[k] = 'cruising'
    # 0.2 s windows (tiles)
    wc = np.arange(win_s / 2, W, win_s)
    NW = len(wc)
    iw = np.searchsorted(ti, wc).clip(0, len(ti) - 1)
    Fw = F[iw]
    cwin = rank[Pr[iw].argmax(1)]
    n_k = np.bincount(cwin, minlength=K).astype(float)
    s_raw = n_k / n_k.sum()
    s_bal = n_k * cw / (n_k * cw).sum()
    rare = int(np.argmin(s_raw))
    hist = None
    tdirs = sorted(x for x in os.listdir(a.run) if x.startswith('train'))
    hp = os.path.join(a.run, tdirs[-1], 'history.json') if tdirs else ''
    if os.path.exists(hp):
        hist = np.array([h['train_loss'] for h in json.load(open(hp))])

    # ---- figure
    FW, FH = 9.6, 5.4
    fig = plt.figure(figsize=(FW, FH), dpi=100)
    fig.patch.set_facecolor(BG)
    axO = fig.add_axes([0, 0, 1, 1]); axO.set_xlim(0, 1); axO.set_ylim(0, 1); axO.axis('off'); axO.patch.set_alpha(0)
    axO.set_zorder(5)
    to_fig = fig.transFigure.inverted()
    # timeline of tiles
    TL = (0.05, 0.62, 0.83)                                              # x0, x1, y
    tw = (TL[1] - TL[0]) / NW
    th = 0.05
    tile_x = TL[0] + (np.arange(NW) + 0.5) * tw
    tl = PatchCollection([Rectangle((x - tw * 0.42, TL[2] - th / 2), tw * 0.84, th) for x in tile_x],
                         facecolors=[to_rgba(LIGHT)] * NW, edgecolors='none', zorder=3)
    axO.add_collection(tl)
    axO.text(TL[0], TL[2] + 0.045, f'{a.seq}: first {W:.0f} s of IMU, cut into {NW} windows of {win_s:.1f} s',
             color=MUTED, fontsize=8.5, va='center')
    for s in range(0, int(W) + 1, 5):
        axO.text(TL[0] + s / W * (TL[1] - TL[0]), TL[2] - 0.05, f'{s} s', color=MUTED, fontsize=7, ha='center')
    cursor = Rectangle((0, TL[2] - th / 2 - 0.008), tw, th + 0.016, fc='none', ec=C_HI, lw=1.6, zorder=6, alpha=0)
    axO.add_patch(cursor)
    # feature plane
    PL = [0.09, 0.13, 0.50, 0.55]
    axS = fig.add_axes(PL)
    xmax, ymax = np.percentile(F[:, 0], 99.5) * 1.08, np.percentile(F[:, 1], 99.5) * 1.08
    axS.set_xlim(0, xmax); axS.set_ylim(0, ymax)
    for s in axS.spines.values():
        s.set_color(LIGHT)
    axS.spines['top'].set_visible(False); axS.spines['right'].set_visible(False)
    axS.tick_params(colors=MUTED, labelsize=7, length=2)
    axS.set_xlabel('moving  |mean (a − g)|  [m/s²]', color=MUTED, fontsize=8.5)
    axS.set_ylabel('turning  |mean ω|  [rad/s]', color=MUTED, fontsize=8.5)
    pts = axS.scatter(np.zeros(0), np.zeros(0), s=16, linewidths=0, color=GREY, zorder=3)
    ells = []
    for k in range(K):
        j = order[k]
        C = gm.scaler.scale_[:, None] * gm.gmm.covariances_[j] * gm.scaler.scale_[None]
        ev, V = np.linalg.eigh(C)
        e = Ellipse(mu[j], 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0]), angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])),
                    fc=to_rgba(PAL[k], 0), ec=to_rgba(PAL[k], 0), lw=1.8, zorder=2)
        axS.add_patch(e); ells.append(e)
    elab = [axS.text(*mu_o[k], f'M{k + 1}  {names[k]}', color='white', fontsize=8.5, fontweight='bold', ha='center',
                     va='center', zorder=6, alpha=0, bbox=dict(boxstyle='round,pad=0.25', fc=PAL[k], ec='none', alpha=0))
            for k in range(K)]
    dst = to_fig.transform(axS.transData.transform(Fw))
    chips = axO.scatter(np.zeros(0), np.zeros(0), s=30, marker='s', linewidths=0, color=GREY, zorder=8)

    # sidebar
    X0 = 0.68
    axO.plot([0.655, 0.655], [0.04, 0.96], color=LIGHT, lw=0.8)
    step = axO.text(X0, 0.955, '', color=MUTED, fontsize=8, va='center', fontweight='bold')
    title = axO.text(X0, 0.905, '', color=FG, fontsize=16, fontweight='bold', va='center')
    subt = axO.text(X0, 0.835, '', color=MUTED, fontsize=8.8, va='center', linespacing=1.4)
    big = axO.text(X0, 0.13, '', color=FG, fontsize=22, fontweight='bold', va='center')
    bigl = axO.text(X0, 0.07, '', color=MUTED, fontsize=8.5, va='center')
    # feature card (step 1)
    card = FancyBboxPatch((X0, 0.42), 0.29, 0.28, boxstyle='round,pad=0,rounding_size=0.015', fc='#f6f8fa', ec=LIGHT,
                          lw=1, alpha=0, zorder=2)
    axO.add_patch(card)
    card_t = axO.text(X0 + 0.015, 0.665, '', color=FG, fontsize=9, fontweight='bold', va='center', zorder=3)
    card_l = [axO.text(X0 + 0.015, y, s, color=MUTED, fontsize=8, va='center', zorder=3, alpha=0)
              for y, s in [(0.605, 'moving  |mean (a − g)|'), (0.505, 'turning  |mean ω|')]]
    card_v = [axO.text(X0 + 0.275, y, '', color=FG, fontsize=8.5, fontweight='bold', va='center', ha='right', zorder=3)
              for y in (0.605, 0.505)]
    card_b = [Rectangle((X0 + 0.015, y), 0, 0.025, color=C_HI, zorder=3) for y in (0.555, 0.455)]
    card_bg = [Rectangle((X0 + 0.015, y), 0.26, 0.025, color=LIGHT, zorder=2, alpha=0) for y in (0.555, 0.455)]
    for r in card_bg + card_b:
        axO.add_patch(r)
    formula = axO.text(X0, 0.55, '', color=FG, fontsize=15, va='center', linespacing=1.7)

    # stacks (steps 3-4): tiles as squares in figure coords
    COLS = 4
    sq_w = 0.016
    sq_h = sq_w * FW / FH
    gap = 0.002
    col_x = np.linspace(0.13, 0.53, K)
    base_y = 0.17
    slot = {}
    cnt = np.zeros(K, int)
    for i in np.argsort(wc):
        k = cwin[i]
        r_, c_ = divmod(cnt[k], COLS)
        slot[i] = (col_x[k] + (c_ - (COLS - 1) / 2) * (sq_w + gap), base_y + r_ * (sq_h + gap) + sq_h / 2)
        cnt[k] += 1
    slot_xy = np.array([slot[i] for i in range(NW)])
    rows_k = np.ceil(n_k / COLS)
    stack = {'pc': None}

    def draw_tiles(xy, h, cols, alpha=1.0):
        if stack['pc'] is not None:
            stack['pc'].remove()
        rects = [Rectangle((x - sq_w / 2, y - hh / 2), sq_w, hh) for (x, y), hh in zip(xy, h)]
        cols = np.array(cols, float).reshape(-1, 4).copy()
        cols[:, 3] *= alpha
        stack['pc'] = PatchCollection(rects, facecolors=cols, edgecolors='none', zorder=7)
        axO.add_collection(stack['pc'])
    msz = (sq_w * FW * 72) ** 2                                          # marker area for one tile
    col_lab = [axO.text(col_x[k], base_y - 0.035, '', color=PAL[k], fontsize=8.5, fontweight='bold', ha='center',
                        va='center', alpha=0) for k in range(K)]
    col_pct = [axO.text(col_x[k], 0, '', color=FG, fontsize=9, fontweight='bold', ha='center', va='bottom', alpha=0)
               for k in range(K)]
    col_w = [axO.text(col_x[k], base_y - 0.075, '', color=MUTED, fontsize=8, ha='center', va='center', alpha=0)
             for k in range(K)]
    # batches (step 5)
    NB = a.batch
    bx = 0.08 + np.arange(NB) * 0.026
    by = np.array([0.62, 0.40])
    tray_lab = [axO.text(0.08 - 0.012, by[i] + 0.06, s, color=c, fontsize=9, fontweight='bold', va='center', alpha=0)
                for i, (s, c) in enumerate([('uniform sampling', MUTED), ('motion-balanced sampling', C_HI)])]
    tray_cnt = [axO.text(bx[-1] + 0.02, by[i], '', color=FG, fontsize=9, fontweight='bold', va='center', alpha=0)
                for i in range(2)]

    def quota(p):
        q = np.floor(p * NB).astype(int)
        for k in np.argsort(-(p * NB - q))[:NB - q.sum()]:
            q[k] += 1
        return np.concatenate([np.full(q[k], k) for k in range(K)])

    batch_k = [quota(s_raw), quota(s_bal)]
    rngb = np.random.default_rng(5)
    batch_k = [rngb.permutation(b) for b in batch_k]
    bslots = axO.scatter(np.tile(bx, 2), np.repeat(by, NB), s=msz * 1.3, marker='s', color=to_rgba(LIGHT, 0), linewidths=0, zorder=6)
    bfill = axO.scatter(np.zeros(0), np.zeros(0), s=msz * 1.3, marker='s', linewidths=0, zorder=7)

    N1, N2, N3, N4, N5, HOLD = 95, 55, 60, 65, 85, 35
    S = np.cumsum([0, N1, N2, N3, N4, N5])
    total = int(S[-1] + HOLD)
    SLOW = 6
    emit = np.r_[np.arange(SLOW) * 7.0, np.linspace(SLOW * 7.0, 0.82 * N1, NW - SLOW)]
    fly = np.r_[np.full(SLOW, 14.0), np.full(NW - SLOW, 11.0)]
    reg_rgba = np.array([to_rgba(PAL[k]) for k in cwin])
    state = {'ys': slot_xy[:, 1]}
    grey_rgba = np.array(to_rgba(GREY))

    def txt(n, ttl, st):
        step.set_text(f'STEP {n} / 5'); title.set_text(ttl); subt.set_text(st)

    def card_alpha(al):
        card.set_alpha(al)
        for x in card_l + card_bg:
            x.set_alpha(al)
        for x in card_v + card_b + [card_t]:
            x.set_alpha(al)

    def frame(f):
        if f < S[1]:                                                       # 1. windows -> points
            txt(1, 'Windows → motion points', f'each {win_s:.1f} s IMU window is summarised\nby how much it moves and turns')
            n_em = int((emit <= f).sum())
            age = f - emit[:n_em]
            fl_ = age < fly[:n_em]
            u = ease(age[fl_] / fly[:n_em][fl_])[:, None]
            src = np.stack([tile_x[:n_em][fl_], np.full(fl_.sum(), TL[2])], 1)
            pos = (1 - u) * src + u * dst[:n_em][fl_]
            pos[:, 0] += 0.0
            chips.set_offsets(pos if len(pos) else np.zeros((0, 2)))
            landed = np.where(~fl_)[0]
            pts.set_offsets(Fw[landed] if len(landed) else np.zeros((0, 2)))
            fc = np.array([to_rgba(LIGHT)] * NW); fc[:n_em] = to_rgba(GREY)
            tl.set_facecolor(fc)
            cur = max(0, n_em - 1)
            cursor.set_alpha(1 if n_em else 0); cursor.set_x(tile_x[cur] - tw / 2)
            card_alpha(1.0 if n_em else 0.0)
            card_t.set_text(f'window #{cur + 1}   t = {wc[cur] - win_s / 2:4.1f}–{wc[cur] + win_s / 2:4.1f} s')
            for v, b, val, mx in zip(card_v, card_b, Fw[cur], (xmax, ymax)):
                v.set_text(f'{val:.2f}'); b.set_width(0.26 * min(1, val / mx))
            big.set_text(f'{len(landed)}'); bigl.set_text(f'motion points of {NW}')
        elif f < S[2]:                                                     # 2. regimes
            q = ease((f - S[1] + 1 - 5) / (0.55 * N2))
            txt(2, 'Motion regimes', 'a GMM (K by BIC) groups the points;\nthe timeline shows when each regime happened')
            chips.set_offsets(np.zeros((0, 2))); cursor.set_alpha(0); card_alpha(1 - q)
            pts.set_offsets(Fw); pts.set_color((1 - q) * grey_rgba + q * reg_rgba)
            tl.set_facecolor((1 - q) * grey_rgba + q * reg_rgba)
            for k, e in enumerate(ells):
                e.set_facecolor(to_rgba(PAL[k], 0.10 * q)); e.set_edgecolor(to_rgba(PAL[k], q))
            for tx in elab:
                tx.set_alpha(q); tx.get_bbox_patch().set_alpha(q)
            big.set_text(f'K = {K}'); bigl.set_text('motion regimes in the warm-up')
        elif f < S[3]:                                                     # 3. sort into stacks
            fl = f - S[2]
            q = ease((fl + 1 - 3 - np.arange(NW) * 0.12) / 22)                  # staggered flights
            txt(3, 'Imbalance', 'sorted by regime, the windows stack up:\ncalm motion dominates, fast turns are rare')
            qa = ease((fl + 1) / 12)
            axS.set_visible(qa < 0.99)
            pts.set_alpha(1 - qa)
            for k, e in enumerate(ells):
                e.set_facecolor(to_rgba(PAL[k], 0.10 * (1 - qa))); e.set_edgecolor(to_rgba(PAL[k], 1 - qa))
            for tx in elab:
                tx.set_alpha(1 - qa); tx.get_bbox_patch().set_alpha(1 - qa)
            src = np.stack([tile_x, np.full(NW, TL[2])], 1)
            pos = (1 - q)[:, None] * src + q[:, None] * slot_xy
            pos[:, 1] += 0.08 * np.sin(np.pi * q)
            moving = q > 0
            draw_tiles(pos[moving], np.full(moving.sum(), sq_h), reg_rgba[moving])
            fc = reg_rgba.copy(); fc[moving, 3] = 0.25
            tl.set_facecolor(fc)
            done = ease((fl + 1 - 40) / 12)
            for k in range(K):
                top = base_y + rows_k[k] * (sq_h + gap)
                col_lab[k].set_text(f'M{k + 1} {names[k]}'); col_lab[k].set_alpha(done)
                col_pct[k].set_text(f'{100 * s_raw[k]:.0f}%'); col_pct[k].set_position((col_x[k], top + 0.01))
                col_pct[k].set_alpha(done)
            big.set_text(f'{int(n_k[np.argmax(n_k)])} vs {int(n_k[rare])}'); big.set_color(FG)
            bigl.set_text(f'windows: most common regime vs M{rare + 1} {names[rare]}')
        elif f < S[4]:                                                     # 4. balance
            fl = f - S[3]
            q = ease((fl + 1 - 18) / 26)
            txt(4, 'Balance', 'effective-number weights per regime\n(+ frequency gate, rare-regime boost)')
            formula.set_text(r'$w_k \propto \frac{1-\beta}{1-\beta^{\,n_k}}$' + f'    β = {rw.get("beta", 0.999)}')
            formula.set_alpha(ease((fl + 1) / 10)); formula.set_position((X0, 0.62))
            scale = (1 - q) + q * cw / cw.max() * (n_k.max() * cw.max() / (n_k * cw).max())
            ys = np.empty(NW)
            for i in range(NW):
                k = cwin[i]
                r_ = (slot_xy[i, 1] - base_y - sq_h / 2) / (sq_h + gap)
                ys[i] = base_y + (r_ * (sq_h + gap) + sq_h / 2) * scale[k]
            draw_tiles(np.stack([slot_xy[:, 0], ys], 1), sq_h * scale[cwin], reg_rgba)
            state['ys'] = ys
            sh = (1 - q) * s_raw + q * s_bal
            for k in range(K):
                top = base_y + rows_k[k] * (sq_h + gap) * scale[k]
                col_pct[k].set_text(f'{100 * sh[k]:.0f}%'); col_pct[k].set_position((col_x[k], top + 0.01))
                col_w[k].set_text(f'× {cw[k]:.2f}'); col_w[k].set_alpha(q)
            big.set_text(f'{100 * s_raw[rare]:.0f}% → {100 * sh[rare]:.0f}%'); big.set_color(PAL[rare])
            bigl.set_text(f'share of M{rare + 1} {names[rare]} in the training signal')
        else:                                                              # 5. batches
            fl = f - S[4]
            qa = ease((fl + 1) / 12)
            txt(5, 'What a batch sees', f'{NB} training windows drawn per batch:\nuniform vs motion-balanced')
            formula.set_alpha(1 - qa)
            scale = cw / cw.max() * (n_k.max() * cw.max() / (n_k * cw).max())
            draw_tiles(np.stack([slot_xy[:, 0], state['ys']], 1), sq_h * scale[cwin], reg_rgba, 1 - qa)
            for k in range(K):
                col_lab[k].set_alpha(1 - qa); col_pct[k].set_alpha(1 - qa); col_w[k].set_alpha(1 - qa)
            bslots.set_color(to_rgba(LIGHT, qa))
            for x in tray_lab + tray_cnt:
                x.set_alpha(qa)
            nfill = int(np.clip((fl - 12) / 2.0, 0, NB))
            xy, col = [], []
            for r_, bk in enumerate(batch_k):
                for i in range(nfill):
                    xy.append((bx[i], by[r_])); col.append(to_rgba(PAL[bk[i]]))
            bfill.set_offsets(np.array(xy) if xy else np.zeros((0, 2)))
            if col:
                bfill.set_color(col)
            for r_, bk in enumerate(batch_k):
                nr = int((bk[:nfill] == rare).sum())
                tray_cnt[r_].set_text(f'{nr} × M{rare + 1}')
                tray_cnt[r_].set_color(PAL[rare])
            nr0, nr1 = int((batch_k[0] == rare).sum()), int((batch_k[1] == rare).sum())
            big.set_text(f'{nr0} → {nr1}'); big.set_color(PAL[rare])
            bigl.set_text(f'M{rare + 1} {names[rare]} windows per batch of {NB}')
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
    print('saved', a.out, 'K', K, names, 'n', n_k, 'w', cw.round(2), 'raw', s_raw.round(3), 'bal', s_bal.round(3),
          'batch', [np.bincount(b, minlength=K).tolist() for b in batch_k])


if __name__ == '__main__':
    main()
