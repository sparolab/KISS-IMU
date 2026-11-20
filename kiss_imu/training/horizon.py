"""Multi-horizon supervision on long windows (HorizonWindows), for pseudo-labels.

Over 0.1-1 s a LiDAR-odometry / PGO trajectory is *less* accurate than a good gyro (scan-registration error
enters every scan), so fitting the IMU to the labels there teaches label noise. The labels win only over
longer horizons. This loss compares the corrected IMU with the labels between label stamps i and j = i + h
for several horizons h, with relative preintegration over the whole span (rotation and velocity).

Weighting (loss.horizon.weighting):
  fixed  horizons >= a few seconds only (loss.horizon.horizons), each pair's residual normalised by the
         expected raw-IMU error at that horizon
  cov    all horizons; every pair is weighted by how much more the label can be trusted than the IMU over that
         pair, from a noise model fitted without GT:
             disagreement(label, raw IMU) per-axis variance  D(h) = a + b h + c h^2   (typical-quality pairs)
         a = label error of a pair (two endpoints, s_l^2 = a / 2 each, constant in h), b h + c h^2 = raw-IMU
         error growing with h. An endpoint's label variance is s_l^2 / q, q = LO Hessian min eig / median. Per pair: s_m^2 = b h + c h^2, s_lab^2 = s_l^2 (1/q_i + 1/q_j),
          k = s_m^2 / (s_m^2 + s_lab^2) (trust in the label), weight = max(0, 2k - 1) / s_m^2
         -> 0 where the label is no better than the IMU, 1 / s_m^2 where it is much better.
"""

import numpy as np
import torch

from kiss_imu.imu.preintegration_torch import gt_deltas, preintegrate, residual
from kiss_imu.geometry import so3_torch as so3

G = (0.0, 0.0, -9.81)


def correct_span(model, gyr, acc, L, K, mask=None):
    """Corrected samples after the L - K context, every K-block from its own L-sample window (as at run time).
    gyr, acc: (B, ctx + S, 3) -> (B, S, 3) each. mask: optional {ratio, p} input masking (training only,
    augment.mask_inputs): the network sees the masked window, its correction is applied to the raw samples."""
    B, N, _ = gyr.shape
    n = (N - L) // K + 1
    Gw = gyr.unfold(1, L, K).permute(0, 1, 3, 2).reshape(B * n, L, 3)
    Aw = acc.unfold(1, L, K).permute(0, 1, 3, 2).reshape(B * n, L, 3)
    if mask:
        from .augment import mask_inputs
        out = model(*mask_inputs(Gw, Aw, float(mask['ratio']), float(mask.get('p', 1.0))))
        g, a = Gw + out['d_gyr'], Aw + out['d_acc']
    else:
        out = model(Gw, Aw)
        g, a = out['gyr'], out['acc']
    return g[:, -K:].reshape(B, n * K, 3), a[:, -K:].reshape(B, n * K, 3)


def segment_deltas(gyr, acc, dt, b, kmax):
    """Preintegrate between consecutive boundaries. gyr/acc/dt: (B, S, ...) after the context; b: (B, NB).
    Returns dR (B, NB-1, 3, 3), dV, dP (B, NB-1, 3), T (B, NB-1)."""
    B, S, _ = gyr.shape
    nb = b.shape[1]
    pos = b[:, :-1, None] + torch.arange(kmax, device=b.device)          # (B, NB-1, kmax)
    mask = pos < b[:, 1:, None]
    pos = pos.clamp(max=S - 1)
    gat = lambda x: torch.gather(x, 1, pos.reshape(B, -1, 1).expand(-1, -1, 3)).reshape(B * (nb - 1), kmax, 3)  # noqa
    h = torch.gather(dt, 1, pos.reshape(B, -1)).reshape(B * (nb - 1), kmax) * mask.reshape(-1, kmax)
    dR, dV, dP, T = preintegrate(gat(gyr.double()), gat(acc.double()), h)
    return dR.reshape(B, nb - 1, 3, 3), dV.reshape(B, nb - 1, 3), dP.reshape(B, nb - 1, 3), T.reshape(B, nb - 1)


def cumulative(dR, dV, dP, T):
    """Preintegration from boundary 0 to every boundary: R (B, NB, 3, 3), V, P (B, NB, 3), T (B, NB)."""
    B, n = dR.shape[:2]
    R = [torch.eye(3, dtype=dR.dtype, device=dR.device).expand(B, 3, 3)]
    V, P = [dV.new_zeros(B, 3)], [dV.new_zeros(B, 3)]
    Tc = torch.cat([T.new_zeros(B, 1), T.cumsum(1)], 1)
    for m in range(n):
        P.append(P[-1] + V[-1] * T[:, m, None] + (R[-1] @ dP[:, m, :, None])[..., 0])
        V.append(V[-1] + (R[-1] @ dV[:, m, :, None])[..., 0])
        R.append(R[-1] @ dR[:, m])
    return torch.stack(R, 1), torch.stack(V, 1), torch.stack(P, 1), Tc


def pair_index(NB, steps, stride):
    """(i, j, horizon id) for every horizon (in boundary steps) and start every `stride` boundaries."""
    I, J, H = [], [], []
    for hid, h in enumerate(steps):
        i = np.arange(0, NB - h, stride)
        I.append(i), J.append(i + h), H.append(np.full(len(i), hid))
    return np.concatenate(I), np.concatenate(J), np.concatenate(H)


def pair_residuals(cum, batch, I, J):
    """(B, P, 9) [rot, vel, pos] residual of the IMU between boundaries I and J against the labels."""
    R, V, P, Tc = cum
    Ri, Rj = R[:, I], R[:, J]
    RiT = Ri.transpose(-1, -2)
    T = Tc[:, J] - Tc[:, I]
    dR = RiT @ Rj
    dV = (RiT @ (V[:, J] - V[:, I])[..., None])[..., 0]
    dP = (RiT @ (P[:, J] - P[:, I] - V[:, I] * T[..., None])[..., None])[..., 0]
    B, n = T.shape
    f = lambda x: x.reshape(B * n, *x.shape[2:])                          # noqa: E731
    lab = gt_deltas(f(batch['R'][:, I]), f(batch['p'][:, I]), f(batch['v'][:, I]),
                    f(batch['R'][:, J]), f(batch['p'][:, J]), f(batch['v'][:, J]), f(T))
    return residual((f(dR), f(dV), f(dP)), lab).reshape(B, n, 9)


class HorizonLoss:
    """See the module docstring. cfg = loss.horizon."""

    def __init__(self, cfg, period, NB, parts):
        self.mode = cfg.get('weighting', 'fixed')
        hs = cfg.get('horizons', [5.0, 10.0, 20.0] if self.mode == 'fixed' else [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        self.h_s = np.array([float(h) for h in hs if h <= (NB - 1) * period + 1e-6])
        self.steps = np.maximum(1, np.round(self.h_s / period).astype(int))
        stride = max(1, int(round(float(cfg.get('start_stride_s', 0.5)) / period)))
        I, J, H = pair_index(NB, self.steps, stride)
        self.I, self.J, self.H = (torch.as_tensor(x) for x in (I, J, H))
        self.parts = {k: float(v) for k, v in parts.items() if float(v) > 0}          # rot, vel
        self.noise = None                                                             # fitted in fit_noise
        self.NB, self.period, self.stride = NB, period, stride

    def residuals(self, model_out, batch, kmax):
        g, a = model_out
        dt = batch['dt'][:, -g.shape[1]:]
        seg = segment_deltas(g, a, dt, batch['b'], kmax)
        return pair_residuals(cumulative(*seg), batch, self.I, self.J)

    FIT_H = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)

    @torch.no_grad()
    def fit_noise(self, loader, ctx, kmax, device, max_batches=200):
        """GT-free noise model from the raw IMU against the labels (module docstring), over FIT_H."""
        keep = (self.h_s, self.steps, self.I, self.J, self.H)
        self.h_s = np.array([h for h in self.FIT_H if h <= (self.NB - 1) * self.period + 1e-6])
        self.steps = np.maximum(1, np.round(self.h_s / self.period).astype(int))
        self.I, self.J, self.H = (torch.as_tensor(x) for x in pair_index(self.NB, self.steps, self.stride))
        acc = {k: [[] for _ in self.h_s] for k in ('rot', 'vel')}
        Q = [[] for _ in self.h_s]
        for nb_, b in enumerate(loader):
            if nb_ >= max_batches:
                break
            b = {k: v.to(device) for k, v in b.items()}
            r = self.residuals((b['gyr'][:, ctx:], b['acc'][:, ctx:]), b, kmax)
            ok = (b['valid'][:, self.I] & b['valid'][:, self.J]).cpu()
            qi, qj = b['q'][:, self.I].cpu(), b['q'][:, self.J].cpu()
            inv = 0.5 * (1 / qi.clamp(min=1e-3) + 1 / qj.clamp(min=1e-3))
            for hid in range(len(self.h_s)):
                m = ok & (self.H == hid)[None]
                for k, sl in (('rot', slice(0, 3)), ('vel', slice(3, 6))):
                    acc[k][hid].append(r[..., sl].pow(2).sum(-1).cpu()[m] / 3.0)
                Q[hid].append(inv[m])
        self.noise = {}
        h = self.h_s
        for k in ('rot', 'vel'):
            # robust per-axis variance per horizon (median of chi2_3 / 3 is 0.789) for typical-quality pairs
            D = np.array([float(torch.cat(acc[k][i])[torch.cat(Q[i]) < 1.5].median()) / 0.789 for i in range(len(h))])
            A = np.stack([np.ones_like(h), h, h ** 2], 1)
            from scipy.optimize import nnls
            coef = nnls(A / D[:, None], np.ones_like(h))[0]                       # relative least squares
            a0, b0, c0 = coef
            self.noise[k] = {'s_l2': a0 / 2.0, 'b': b0, 'c': c0, 'D': D.tolist(), 'h': h.tolist(),
                             'fit': (A @ coef).tolist()}
            # crossover: label endpoint variance (a0) = IMU variance (b h + c h^2)
            hh = np.linspace(0.01, 120, 12000)
            cross = hh[np.argmax(b0 * hh + c0 * hh ** 2 > a0)] if (b0 * hh + c0 * hh ** 2 > a0).any() else float('inf')
            self.noise[k]['crossover_s'] = float(cross)
        self.h_s, self.steps, self.I, self.J, self.H = keep
        return self.noise

    def weights(self, batch, kind):
        """(B, P) weights for the 'rot' or 'vel' residual of every pair."""
        n = self.noise[kind]
        h = torch.as_tensor(self.h_s, device=batch['q'].device)[self.H.to(batch['q'].device)]
        s_m2 = (n['b'] * h + n['c'] * h ** 2).clamp(min=1e-12)[None]
        ok = (batch['valid'][:, self.I] & batch['valid'][:, self.J]).double()
        if 'swc' in batch:               # motion reweighting: mean per-sample weight over the pair's span (1 if off)
            I, J = self.I.to(ok.device), self.J.to(ok.device)
            cnt = (batch['b'][:, J] - batch['b'][:, I]).double().clamp(min=1)
            ok = ok * (batch['swc'][:, J] - batch['swc'][:, I]) / cnt
        if self.mode == 'fixed':
            return ok / s_m2
        qi, qj = batch['q'][:, self.I].clamp(min=1e-3), batch['q'][:, self.J].clamp(min=1e-3)
        s_lab2 = n['s_l2'] * (1 / qi + 1 / qj)
        k = s_m2 / (s_m2 + s_lab2)
        return ok * (2 * k - 1).clamp(min=0) / s_m2

    def __call__(self, r, batch):
        total, parts = 0.0, {}
        for kind, sl in (('rot', slice(0, 3)), ('vel', slice(3, 6))):
            if kind not in self.parts:
                continue
            w = self.weights(batch, kind)
            e = r[..., sl].pow(2).sum(-1) / 3.0
            v = (w * e).sum() / (w > 0).sum().clamp(min=1)
            parts[kind] = float(v)
            total = total + self.parts[kind] * v
        return total, parts

    def pair_losses(self, r, batch):
        """(B, P) per-pair loss (the weighted rot + vel terms of __call__) and (B, P) mask of the pairs that
        count; the mean over the mask equals __call__'s total."""
        L, n = 0.0, 0.0
        for kind, sl in (('rot', slice(0, 3)), ('vel', slice(3, 6))):
            if kind not in self.parts:
                continue
            w = self.weights(batch, kind)
            L = L + self.parts[kind] * w * r[..., sl].pow(2).sum(-1) / 3.0
            n = n + (w > 0).double()
        return L, n > 0

    def pair_components(self, batch):
        """(B, P) hard GMM component of every pair: argmax of the mean responsibility over the pair's span."""
        pc = batch['pc']
        I, J = self.I.to(pc.device), self.J.to(pc.device)
        return (pc[:, J] - pc[:, I]).argmax(-1)

    def summary(self):
        return {'mode': self.mode, 'horizons_s': self.h_s.tolist(), 'pairs_per_window': int(len(self.I)),
                'noise': self.noise}
