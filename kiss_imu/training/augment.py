"""Augmentations for training on little data (the warm-start adaptation), after the TartanIMU challenge recipe.

time_scaled: replay a labelled sequence s times faster. The motion's angular rate scales by s and its
gravity-free specific force by s^2; the label poses stay where they are and only their stamps move. The
sensor's own errors must not scale with the motion, so the current bias estimate (the PGO's, at the time
the warm-up ends) is held out before scaling and put back after: a bias that the time-scaling multiplied by
s would teach the denoiser a bias that is not there. Gravity comes from the label attitude, as in the
challenge code (a window mean leaks the platform's own acceleration and s^2 then amplifies it).
Scale factor / misalignment errors scale with the motion, as they should; white noise scales too
(a mild augmentation of its own).

mask_inputs: one contiguous span per input window replaced by a fill (zero rate for the gyro, the window
mean for the accelerometer); the correction is still applied to the unmasked samples, so the network has
to infer the correction of a masked span from its context (regularises towards slowly varying errors).
"""

import copy

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

G_W = np.array([0.0, 0.0, 9.81])          # specific force of a body at rest, world z-up


def time_scaled(seq, s, bg=None, ba=None):
    """A copy of a labelled sequence replayed s times faster (s > 1 faster, s < 1 slower), at the same IMU rate.
    bg, ba: bias estimates of seq's IMU (held out of the scaling); None -> zero."""
    if seq.gt_t is None:
        raise ValueError('time_scaled needs labels (gt or pseudo-labels)')
    s = float(s)
    bg = np.zeros(3) if bg is None else np.asarray(bg, float)
    ba = np.zeros(3) if ba is None else np.asarray(ba, float)
    t0, dt = seq.imu_t[0], float(np.median(np.diff(seq.imu_t)))
    n = max(int(np.floor((seq.imu_t[-1] - t0) / (s * dt))) + 1, 2)
    tau = t0 + np.arange(n) * s * dt                       # original clock, sampled s times faster
    tau = tau[tau <= seq.imu_t[-1]]
    gyr = np.stack([np.interp(tau, seq.imu_t, seq.gyr[:, k]) for k in range(3)], 1)
    acc = np.stack([np.interp(tau, seq.imu_t, seq.acc[:, k]) for k in range(3)], 1)
    R = seq._slerp(np.clip(tau, seq.gt_t[0], seq.gt_t[-1])).as_matrix()
    g_b = np.einsum('nji,j->ni', R, G_W)                    # R^T g: gravity's specific force in the body
    c = copy.copy(seq)
    c.name = f'{seq.name}@x{s:g}'
    c.imu_t = t0 + (tau - t0) / s
    c.gyr = s * (gyr - bg) + bg
    c.acc = s * s * (acc - ba - g_b) + g_b + ba
    warp = lambda t: t0 + (np.asarray(t, float) - t0) / s   # noqa: E731
    c.gt_t = warp(seq.gt_t)
    c._slerp = Slerp(c.gt_t, Rotation.from_matrix(seq.gt_R))
    c._gt_v = None
    c.anchor_t = None if seq.anchor_t is None else warp(seq.anchor_t)
    c.invalid_t = warp(seq.invalid_t)
    if seq.label_q is not None:
        c.label_q = seq.label_q.copy()
        c.label_q[:, 0] = warp(seq.label_q[:, 0])
    c.scan_files, c.scan_t = [], np.zeros(0)
    return c


def mask_inputs(gyr, acc, ratio, p=1.0, generator=None):
    """(B, L, 3) each -> masked copies: in a fraction p of the windows, one contiguous span of ratio * L
    samples is set to zero rate (gyro) and to the window's mean specific force (accelerometer)."""
    B, L, _ = gyr.shape
    w = max(1, int(round(L * ratio)))
    dev = gyr.device
    start = torch.randint(0, L - w + 1, (B,), device=dev, generator=generator)
    on = torch.rand(B, device=dev, generator=generator) < p
    pos = torch.arange(L, device=dev)[None]
    m = ((pos >= start[:, None]) & (pos < start[:, None] + w) & on[:, None])[..., None]
    g = torch.where(m, torch.zeros_like(gyr), gyr)
    a = torch.where(m, acc.mean(1, keepdim=True).expand_as(acc), acc)
    return g, a
