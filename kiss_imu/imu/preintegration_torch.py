"""Batched, differentiable IMU preintegration in torch (training).

Same algebra as kiss_imu/imu/preintegration.py, over a batch of segments at once, without
covariance (the denoising head has no uncertainty output).
"""

import torch

from kiss_imu.geometry import so3_torch as so3

GRAVITY = (0.0, 0.0, -9.81)


def preintegrate(gyr, acc, dt):
    """gyr, acc: (B, K, 3); dt: (B, K). Returns dR (B,3,3), dV (B,3), dP (B,3), T (B,)."""
    B = gyr.shape[0]
    dR = torch.eye(3, dtype=gyr.dtype, device=gyr.device).expand(B, 3, 3).clone()
    dV = gyr.new_zeros(B, 3)
    dP = gyr.new_zeros(B, 3)
    for k in range(gyr.shape[1]):
        h = dt[:, k, None]
        Ra = (dR @ acc[:, k, :, None])[..., 0]
        dP = dP + dV * h + 0.5 * Ra * h * h
        dV = dV + Ra * h
        dR = dR @ so3.exp(gyr[:, k] * h)
    return dR, dV, dP, dt.sum(1)


def gt_deltas(R_i, p_i, v_i, R_j, p_j, v_j, T):
    """What an ideal IMU would preintegrate between two GT states."""
    g = torch.tensor(GRAVITY, dtype=R_i.dtype, device=R_i.device)
    RiT = R_i.transpose(-1, -2)
    T = T[:, None]
    dR = RiT @ R_j
    dV = (RiT @ (v_j - v_i - g * T)[..., None])[..., 0]
    dP = (RiT @ (p_j - p_i - v_i * T - 0.5 * g * T * T)[..., None])[..., 0]
    return dR, dV, dP


def residual(pred, gt):
    """(N, 9) [rot, vel, pos] residual of preintegrated (dR, dV, dP) against GT deltas."""
    (dR, dV, dP), (gR, gV, gP) = pred, gt
    return torch.cat([so3.log(dR.transpose(-1, -2) @ gR), gV - dV, gP - dP], -1)
