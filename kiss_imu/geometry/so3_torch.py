"""SO(3) helpers in torch, batched over leading dims (training-side preintegration)."""

import torch


def hat(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([
        torch.stack([z, -w[..., 2], w[..., 1]], -1),
        torch.stack([w[..., 2], z, -w[..., 0]], -1),
        torch.stack([-w[..., 1], w[..., 0], z], -1),
    ], -2)


def vee(W):
    return torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], -1)


def _theta(w):
    return torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-24)   # finite gradient at 0


def exp(w):
    th = _theta(w)[..., None]
    small = th < 1e-4
    th2 = th * th
    A = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    B = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2)
    K = hat(w)
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    return I + A * K + B * (K @ K)


def log(Rm):
    """Valid for rotation angles below pi."""
    tr = Rm[..., 0, 0] + Rm[..., 1, 1] + Rm[..., 2, 2]
    c = ((tr - 1) / 2).clamp(-1 + 1e-12, 1 - 1e-12)
    th = torch.acos(c)[..., None]
    small = th < 1e-4
    s = torch.where(small, 1 - th * th / 6, torch.sin(th) / th)
    return vee(Rm - Rm.transpose(-1, -2)) / (2 * s)
