"""Output heads on encoder features. Register new ones in HEADS."""

import math

import torch.nn as nn

ACC_SCALE, GYR_SCALE = 0.1, math.pi / 180       # m/s^2, rad/s per unit of head output


class DenoisingHead(nn.Module):
    """2-layer MLP giving an additive correction to the gyro and/or accelerometer.

    targets: which sensors it corrects ('gyr', 'acc'); the others pass through unchanged.
    The last layer starts at zero, so an untrained net returns the raw IMU.
    Output units: 1 = 1 deg/s (gyro), 0.1 m/s^2 (accelerometer).
    """
    SCALE = {'gyr': GYR_SCALE, 'acc': ACC_SCALE}

    def __init__(self, in_dim, hidden=64, targets=('gyr', 'acc')):
        super().__init__()
        self.targets = tuple(targets)
        assert self.targets and set(self.targets) <= {'gyr', 'acc'}, targets
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 3 * len(self.targets)))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, f):
        out = self.mlp(f)
        res = {f'd_{k}': f.new_zeros(*f.shape[:-1], 3) for k in ('gyr', 'acc')}
        for n, k in enumerate(self.targets):
            res[f'd_{k}'] = out[..., 3 * n:3 * n + 3] * self.SCALE[k]
        return res


HEADS = {'denoising': DenoisingHead}
