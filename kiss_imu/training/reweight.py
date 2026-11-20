"""Motion-frequency reweighting of training windows (GMM on IMU motion features).

Each IMU sample gets a 2-d motion feature [linear speed proxy, angular speed] averaged over a
short window (ported from branch icra2026: src/models/gmm.py). A GMM is fit on the training
features; every component gets a weight that grows as it gets rarer, and every training window
gets the responsibility-weighted mean of its samples' component weights (soft-mode). Weights are
normalised to mean 1 over the training windows, so the loss scale is unchanged.

Only the IMU is used, so the same weighting works without GT.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

G = np.array([0.0, 0.0, 9.81])


def _moving_mean(x, L):
    k = np.ones(L) / L
    if x.ndim == 1:
        return np.convolve(x, k, mode='same')
    return np.stack([np.convolve(x[:, i], k, mode='same') for i in range(x.shape[1])], 1)


def motion_features(ts, gyr, acc, win_s=0.2):
    """(N, 2) per-sample [|mean gravity-free acc|, |mean gyro|] over a centred win_s window.
    Gravity is removed with a yaw-only attitude, as in the original module, along the IMU z axis with the sign
    the data shows (z-up: +g as before; z-down IMUs such as the ANYmal built-in one: -g, which would otherwise
    double gravity into a constant ~2 g feature)."""
    dt = np.diff(ts, prepend=ts[0])
    dt[dt <= 0] = np.median(dt[dt > 0])
    yaw = np.cumsum(gyr[:, 2] * dt)
    c, s = np.cos(yaw), np.sin(yaw)
    g = G * (1.0 if np.median(acc[:, 2]) >= 0 else -1.0)
    acc_w = np.stack([c * acc[:, 0] - s * acc[:, 1], s * acc[:, 0] + c * acc[:, 1], acc[:, 2]], 1) - g
    L = max(1, int(round(win_s / np.median(dt))))
    return np.stack([np.linalg.norm(_moving_mean(acc_w, L), axis=1),
                     np.linalg.norm(_moving_mean(gyr, L), axis=1)], 1)


def motion_features_rich(ts, gyr, acc, win_s=0.2):
    """(N, 9) per-sample motion description over a centred win_s window, richer than motion_features:
    mean body rate (3, signed: which axis turns and which way), mean specific-force direction (3: the tilt
    relative to gravity), rate and specific-force spread (std of |gyr|, std of |acc|: gait / impacts) and
    the dynamic specific force | |mean acc| - g |."""
    dt = np.diff(ts, prepend=ts[0])
    dt[dt <= 0] = np.median(dt[dt > 0])
    L = max(1, int(round(win_s / np.median(dt))))
    w_m, f_m = _moving_mean(gyr, L), _moving_mean(acc, L)
    gn, an = np.linalg.norm(gyr, axis=1, keepdims=True), np.linalg.norm(acc, axis=1, keepdims=True)
    g_sd = np.sqrt(np.clip(_moving_mean(gn ** 2, L) - _moving_mean(gn, L) ** 2, 0, None))
    a_sd = np.sqrt(np.clip(_moving_mean(an ** 2, L) - _moving_mean(an, L) ** 2, 0, None))
    fn = np.linalg.norm(f_m, axis=1, keepdims=True)
    return np.concatenate([w_m, f_m / np.clip(fn, 1e-6, None), g_sd, a_sd, np.abs(fn - G)], 1)


def features_fn(cfg):
    """reweight.features: 'norm' (default; |gravity-free acc|, |gyr| as the original module) or 'rich'."""
    kind = (cfg or {}).get('features', 'norm')
    if kind == 'norm':
        return motion_features
    if kind == 'rich':
        return motion_features_rich
    raise ValueError(f'unknown reweight.features {kind}')


class MotionGMM:
    def __init__(self, n_components=None, k_range=(2, 8), win_s=0.2, method='sqrt_inv', beta=0.999,
                 clamp=(0.1, 10.0), max_fit_samples=200000, seed=0):
        self.K, self.k_range, self.win_s = n_components, k_range, win_s
        self.method, self.beta, self.clamp = method, beta, clamp
        self.max_fit, self.seed = max_fit_samples, seed

    def fit(self, feats):
        X = np.vstack(feats)
        rng = np.random.default_rng(self.seed)
        Xf = X[rng.choice(len(X), self.max_fit, replace=False)] if len(X) > self.max_fit else X
        self.scaler = StandardScaler().fit(Xf)
        Xn = self.scaler.transform(Xf)
        ks = [self.K] if self.K else range(*self.k_range)
        fits = [GaussianMixture(k, covariance_type='full', random_state=self.seed, n_init=3).fit(Xn) for k in ks]
        self.gmm = min(fits, key=lambda g: g.bic(Xn))
        self.K = self.gmm.n_components
        return self

    def proba(self, X):
        return self.gmm.predict_proba(self.scaler.transform(X))

    def component_weights(self, counts):
        n = np.clip(np.asarray(counts, float), 1e-9, None)
        if self.method == 'inverse':
            w = 1.0 / n
        elif self.method == 'sqrt_inv':
            w = 1.0 / np.sqrt(n)
        elif self.method == 'effective':     # class-balanced loss, Cui et al. 2019
            w = (1 - self.beta) / (1 - self.beta ** np.clip(n, 1, None))
        else:
            raise ValueError(f'unknown reweight method {self.method}')
        return w / w.mean()


def window_weights(dataset, sequences, cfg):
    """Per-window weights (mean 1) for a GTWindows dataset, plus a summary dict."""
    gm = MotionGMM(cfg.get('n_components'), tuple(cfg.get('k_range', (2, 8))), cfg.get('win_s', 0.2),
                   cfg.get('method', 'sqrt_inv'), cfg.get('beta', 0.999), tuple(cfg.get('clamp', (0.1, 10.0))),
                   seed=cfg.get('seed', 0))
    feats = [features_fn(cfg)(s.imu_t, s.gyr, s.acc, gm.win_s) for s in sequences]
    used = [np.zeros(len(f), bool) for f in feats]          # fit only on samples the training windows cover
    for mi, s in dataset.items:
        used[mi][s:s + dataset.L] = True
    gm.fit([f[u] for f, u in zip(feats, used) if u.any()])
    probs = [gm.proba(f) for f in feats]
    # window-level soft responsibilities: mean over the window's samples
    P = np.stack([probs[mi][s:s + dataset.L].mean(0) for mi, s in dataset.items])
    counts = P.sum(0)
    cw = gm.component_weights(counts)
    w = P @ cw
    w = np.clip(w / w.mean(), *gm.clamp)
    w = w / w.mean()
    order = np.argsort(counts)
    summary = {'K': gm.K, 'share': (counts / counts.sum())[order].round(4).tolist(),
               'comp_weight': cw[order].round(3).tolist(),
               'means_[acc,gyr]': gm.scaler.inverse_transform(gm.gmm.means_)[order].round(3).tolist(),
               'window_w': {'min': float(w.min()), 'p50': float(np.median(w)), 'p95': float(np.quantile(w, .95)),
                            'max': float(w.max())}}
    return w.astype(np.float32), summary


def sample_weights(sequences, used, cfg):
    """Per-sample soft weights (mean 1 over the used samples) for losses whose units are not fixed windows
    (the multi-horizon loss: a pair's weight is the mean over the samples it spans, see HorizonWindows).
    used: per-sequence boolean masks of the samples the training covers (the GMM is fit on those only)."""
    gm = MotionGMM(cfg.get('n_components'), tuple(cfg.get('k_range', (2, 8))), cfg.get('win_s', 0.2),
                   cfg.get('method', 'sqrt_inv'), cfg.get('beta', 0.999), tuple(cfg.get('clamp', (0.1, 10.0))),
                   seed=cfg.get('seed', 0))
    feats = [features_fn(cfg)(s.imu_t, s.gyr, s.acc, gm.win_s) for s in sequences]
    gm.fit([f[u] for f, u in zip(feats, used) if u.any()])
    probs = [gm.proba(f) for f in feats]
    counts = sum(p[u].sum(0) for p, u in zip(probs, used))                # soft sample counts
    cw = gm.component_weights(counts)
    ws = [p @ cw for p in probs]
    mean = np.concatenate([w[u] for w, u in zip(ws, used)]).mean()
    ws = [np.clip(w / mean, *gm.clamp) for w in ws]
    mean = np.concatenate([w[u] for w, u in zip(ws, used)]).mean()
    ws = [(w / mean).astype(np.float32) for w in ws]
    allw = np.concatenate([w[u] for w, u in zip(ws, used)])
    order = np.argsort(counts)
    summary = {'K': gm.K, 'share': (counts / counts.sum())[order].round(4).tolist(),
               'comp_weight': cw[order].round(3).tolist(),
               'means_[acc,gyr]': gm.scaler.inverse_transform(gm.gmm.means_)[order].round(3).tolist(),
               'sample_w': {'min': float(allw.min()), 'p50': float(np.median(allw)),
                            'p95': float(np.quantile(allw, .95)), 'max': float(allw.max())}}
    return ws, summary


# ---------------------------------------------------------------------------------------------------
# KISS-IMU's full motion-balancing recipe (the ICRA 2026 code, branch icra2026: src/train.py), applied per batch:
#   (1) class weights      effective-number weights of the window's hard component, batch-normalised to mean 1
#   (2) FreqGate           drop windows of components used more often than 1/K (EMA of usage), keep prob
#                          clamp((1/K / freq)^alpha, p_min, p_max)
#   (3) rare boost         + boost * mean loss of the top-k rarest windows of the batch
#   (4) convergence        a component whose loss EMA stops improving (patience steps, tol) leaves the loss;
#                          a batch made only of converged components is skipped
# ---------------------------------------------------------------------------------------------------

def window_components(dataset, sequences, cfg):
    """Hard component id per window (soft-mode: argmax of the window-mean responsibilities), the
    effective-number component weights from the soft sample counts, and a summary."""
    gm = MotionGMM(cfg.get('n_components'), tuple(cfg.get('k_range', (2, 8))), cfg.get('win_s', 0.2),
                   cfg.get('method', 'effective'), cfg.get('beta', 0.999), tuple(cfg.get('clamp', (0.1, 10.0))),
                   seed=cfg.get('seed', 0))
    feats = [features_fn(cfg)(s.imu_t, s.gyr, s.acc, gm.win_s) for s in sequences]
    used = [np.zeros(len(f), bool) for f in feats]
    for mi, s in dataset.items:
        used[mi][s:s + dataset.L] = True
    gm.fit([f[u] for f, u in zip(feats, used) if u.any()])
    probs = [gm.proba(f) for f in feats]
    counts = sum(p[u].sum(0) for p, u in zip(probs, used))                # soft sample counts
    cw = np.clip(gm.component_weights(counts), *gm.clamp)
    comp = np.array([int(probs[mi][s:s + dataset.L].mean(0).argmax()) for mi, s in dataset.items])
    win_share = np.bincount(comp, minlength=gm.K) / len(comp)
    order = np.argsort(counts)
    summary = {'K': gm.K, 'sample_share': (counts / counts.sum())[order].round(4).tolist(),
               'window_share': win_share[order].round(4).tolist(), 'comp_weight': cw[order].round(3).tolist(),
               'means_[acc,gyr]': gm.scaler.inverse_transform(gm.gmm.means_)[order].round(3).tolist()}
    return comp, cw.astype(np.float32), gm.K, summary


def pair_responsibilities(sequences, used, cfg):
    """KISS-IMU recipe for the multi-horizon loss, whose units are label pairs, not fixed windows: per-sample
    GMM responsibilities (a label pair's hard component is the argmax of their mean over the samples it spans,
    HorizonLoss.pair_components), the effective-number component weights from the soft sample counts, a summary.
    used: per-sequence boolean masks of the samples the training covers (the GMM is fit on those only)."""
    gm = MotionGMM(cfg.get('n_components'), tuple(cfg.get('k_range', (2, 8))), cfg.get('win_s', 0.2),
                   cfg.get('method', 'effective'), cfg.get('beta', 0.999), tuple(cfg.get('clamp', (0.1, 10.0))),
                   seed=cfg.get('seed', 0))
    feats = [features_fn(cfg)(s.imu_t, s.gyr, s.acc, gm.win_s) for s in sequences]
    gm.fit([f[u] for f, u in zip(feats, used) if u.any()])
    probs = [gm.proba(f) for f in feats]
    counts = sum(p[u].sum(0) for p, u in zip(probs, used))                # soft sample counts
    cw = np.clip(gm.component_weights(counts), *gm.clamp)
    order = np.argsort(counts)
    summary = {'K': gm.K, 'sample_share': (counts / counts.sum())[order].round(4).tolist(),
               'comp_weight': cw[order].round(3).tolist(),
               'means_[acc,gyr]': gm.scaler.inverse_transform(gm.gmm.means_)[order].round(3).tolist()}
    return probs, cw.astype(np.float32), gm.K, summary


class FreqGate:
    def __init__(self, K, alpha=0.7, p_min=0.15, p_max=1.0, prior=10.0, ema=0.99):
        self.K, self.alpha, self.p_min, self.p_max, self.prior, self.beta = K, alpha, p_min, p_max, prior, ema
        self.rate = np.zeros(K)

    def rarity(self):
        freq = (self.rate + self.prior) / (self.rate.sum() + self.prior * self.K)
        return np.clip((1.0 / self.K / freq) ** self.alpha, self.p_min, self.p_max)

    def keep(self, comp, rng):
        p = self.rarity()[comp]
        keep = rng.random(len(comp)) < p
        if not keep.any():
            keep[np.argmax(p)] = True
        return keep

    def update(self, comp, used):
        if used.sum() > 0:
            h = np.bincount(comp, weights=used.astype(float), minlength=self.K) / used.sum()
            self.rate = self.beta * self.rate + (1 - self.beta) * h


class ConvergenceManager:
    """A component whose loss EMA stops improving leaves the loss (KISS-IMU CompConvergenceManager).
    unit 'step' (the original): EMA over the per-batch component means, patience in batches, absolute tol.
    unit 'epoch': EMA over the per-epoch component means (end_epoch), patience in epochs, tol relative to the
    loss level (rel_tol). The per-step rule with an absolute tol fires on minibatch noise as soon as the EMA
    happens to sit within tol of its best for `patience` batches, which on a loss of order 1 drops every
    component within a few epochs (runs/t10/gt/ga_bal_full: all 7 converged, 7197 batches skipped)."""

    def __init__(self, K, ema=0.9, tol=1e-3, patience=5, min_seen=100, unit='step', rel_tol=None):
        self.K, self.beta, self.tol, self.patience, self.min_seen = K, ema, tol, patience, min_seen
        self.unit, self.rel_tol = unit, rel_tol
        self.ema, self.best = np.full(K, np.inf), np.full(K, np.inf)
        self.count, self.bad, self.done = np.zeros(K, int), np.zeros(K, int), np.zeros(K, bool)
        self._sum, self._n = np.zeros(K), np.zeros(K)

    def _observe(self, k, cur):
        self.ema[k] = cur if np.isinf(self.ema[k]) else self.beta * self.ema[k] + (1 - self.beta) * cur
        improved = self.best[k] - self.ema[k]
        if self.ema[k] < self.best[k]:
            self.best[k], self.bad[k] = self.ema[k], 0
        else:
            self.bad[k] += 1
        tol = self.tol if self.rel_tol is None else self.rel_tol * max(abs(self.best[k]), 1e-12)
        if (self.count[k] >= self.min_seen and self.bad[k] >= self.patience and abs(improved) < tol
                and not self.done[k]):
            self.done[k] = True
            print(f'  [balance] component {k} converged (loss ema {self.ema[k]:.4f})', flush=True)

    def update(self, comp, loss):
        for k in np.unique(comp):
            m = comp == k
            self.count[k] += int(m.sum())
            if self.unit == 'epoch':
                self._sum[k] += float(loss[m].sum()); self._n[k] += int(m.sum())
            else:
                self._observe(k, float(loss[m].mean()))

    def end_epoch(self):
        if self.unit != 'epoch':
            return
        for k in np.flatnonzero(self._n):
            self._observe(k, self._sum[k] / self._n[k])
        self._sum[:], self._n[:] = 0.0, 0.0

    def reset(self):
        self.__init__(self.K, self.beta, self.tol, self.patience, self.min_seen, self.unit, self.rel_tol)


class MotionBalancer:
    """Turns per-window losses + component ids into the training loss (None = skip the batch)."""

    def __init__(self, K, comp_weights, cfg, seed=0):
        self.K, self.cw = K, comp_weights if cfg.get('class_weight', True) else np.ones(K, np.float32)
        fg, rb, cv = cfg.get('freq_gate'), cfg.get('rare_boost'), cfg.get('convergence')
        self.gate = FreqGate(K, **{k: v for k, v in fg.items() if k != 'enabled'}) if fg and fg.get('enabled') else None
        self.boost = (float(rb.get('weight', 0.3)), float(rb.get('topk', 0.25))) if rb and rb.get('enabled') else None
        self.conv = (ConvergenceManager(K, **{k: v for k, v in cv.items() if k != 'enabled'})
                     if cv and cv.get('enabled') else None)
        self.rng = np.random.default_rng(seed)
        self.skipped = 0
        ad = cfg.get('additive')     # {weight}: plain mean over ALL windows + weight * component-balanced mean
        self.additive = float(ad.get('weight', 1.0)) if ad and ad.get('enabled', True) else None

    def __call__(self, comp_t, loss_w):
        import torch
        comp = comp_t.cpu().numpy()
        if self.additive is not None:
            # no window is down-weighted or dropped (a weight that depends only on the input leaves the
            # Bayes-optimal correction unchanged, so down-weighting common motion only costs samples);
            # the rare components get extra gradient through a term that averages per-component means
            ks = np.unique(comp)
            cm = torch.stack([loss_w[torch.as_tensor(comp == k, device=loss_w.device)].mean() for k in ks])
            return loss_w.mean() + self.additive * cm.mean()
        w = torch.as_tensor(self.cw[comp], device=loss_w.device, dtype=loss_w.dtype)
        w = w / w.mean()
        active = np.ones(len(comp), bool)
        if self.conv is not None:
            active = ~self.conv.done[comp]
            if not active.any():
                self.skipped += 1
                return None
        keep = np.ones(len(comp), bool)
        if self.gate is not None:
            keep = self.gate.keep(comp, self.rng)
            if not (keep & active).any():
                a = np.flatnonzero(active)
                keep[a[np.argmax(self.gate.rarity()[comp[a]])]] = True
        eff = w * torch.as_tensor((active & keep).astype(np.float32), device=loss_w.device, dtype=loss_w.dtype)
        loss = (eff * loss_w).mean()
        rare = np.zeros(len(comp), bool)
        if self.boost is not None and self.gate is not None:
            k = max(1, int(self.boost[1] * len(comp)))
            rare[np.argsort(-self.gate.rarity()[comp], kind='stable')[:k]] = True
            m = torch.as_tensor(rare.astype(np.float32), device=loss_w.device, dtype=loss_w.dtype)
            loss = loss + self.boost[0] * (m * loss_w).sum() / (m.sum() + 1e-6)
        if self.gate is not None:
            self.gate.update(comp, (active & keep) | rare)
        if self.conv is not None:
            self.conv.update(comp, loss_w.detach().cpu().numpy())
        return loss

    def end_epoch(self):
        if self.conv is not None:
            self.conv.end_epoch()
