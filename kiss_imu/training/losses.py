"""GT supervision for the denoiser: preintegrate the corrected samples per segment and
compare with the GT deltas. Loss = sum over [rot, vel, pos] of w * mean(|r / scale|^2),
optionally a weighted mean over segments (motion reweighting, see reweight.py)."""

import torch

from kiss_imu.imu.preintegration_torch import gt_deltas, preintegrate, residual


def segment_residuals(batch, gyr, acc, K):
    """(B * M, 9) residual of every K-sample segment of the window."""
    B, L, _ = gyr.shape
    M = L // K
    seg = lambda x: x.reshape(B * M, K, *x.shape[2:])            # noqa: E731
    dR, dV, dP, T = preintegrate(seg(gyr.double()), seg(acc.double()), seg(batch['dt']))
    Ri, pi, vi = (batch[k][:, :-1].reshape(B * M, *batch[k].shape[2:]) for k in ('R', 'p', 'v'))
    Rj, pj, vj = (batch[k][:, 1:].reshape(B * M, *batch[k].shape[2:]) for k in ('R', 'p', 'v'))
    return residual((dR, dV, dP), gt_deltas(Ri, pi, vi, Rj, pj, vj, T))


class SegmentLoss:
    PARTS = ('rot', 'vel', 'pos')

    def __init__(self, scale, weight):
        self.scale = {k: float(scale[k]) for k in self.PARTS}
        self.weight = {k: float(weight.get(k, 1.0)) for k in self.PARTS}

    def __call__(self, r, w=None):
        """r: (N, 9) residuals; w: optional (N,) per-row weights (mean ~1)."""
        red = (lambda x: x.mean()) if w is None else (lambda x: (w * x).sum() / w.sum())   # noqa: E731
        parts = {k: red((r[:, 3 * i:3 * i + 3] / self.scale[k]).pow(2).sum(-1)) for i, k in enumerate(self.PARTS)}
        total = sum(self.weight[k] * parts[k] for k in self.PARTS)
        return total, {k: float(v) for k, v in parts.items()}


def per_row(loss, r):
    """(N,) weighted loss of every residual row (same terms as SegmentLoss)."""
    return sum(loss.weight[k] * (r[:, 3 * i:3 * i + 3] / loss.scale[k]).pow(2).sum(-1) for i, k in enumerate(loss.PARTS))


def rmse(r):
    """Per-part RMSE of (N, 9) residuals: rot in deg, vel in m/s, pos in m."""
    rms = lambda x: float(x.pow(2).sum(-1).mean().sqrt())        # noqa: E731
    return {'rot_deg': rms(r[:, :3]) * 57.29577951308232, 'vel': rms(r[:, 3:6]), 'pos': rms(r[:, 6:])}
