"""Encoder + head(s). The denoiser returns corrected gyro / accelerometer samples."""

import torch
import torch.nn as nn

from .encoders import ENCODERS
from .heads import HEADS

G = 9.81


class Affine(nn.Module):
    """Sensor calibration as a correction: d = M x + b per sensor (M: scale factors and axis misalignment, b: bias),
    one set for the whole run, starting at zero (the raw IMU). 24 parameters, so a few seconds of labels pin it."""

    def __init__(self, targets=('gyr', 'acc')):
        super().__init__()
        self.targets = tuple(targets)
        self.M = nn.ParameterDict({k: nn.Parameter(torch.zeros(3, 3)) for k in self.targets})
        self.b = nn.ParameterDict({k: nn.Parameter(torch.zeros(3)) for k in self.targets})

    def forward(self, gyr, acc):
        x = {'gyr': gyr, 'acc': acc}
        return {f'd_{k}': (x[k] @ self.M[k].T + self.b[k]) if k in self.targets else torch.zeros_like(x[k])
                for k in ('gyr', 'acc')}


class IMUDenoiser(nn.Module):
    """heads: learned corrections from the encoder features (may be empty); affine: an Affine calibration added
    to them (model.affine: true or {targets: [...]})."""

    def __init__(self, encoder='cnn_gru', encoder_args=None, heads=('denoising',), head_args=None, affine=None):
        super().__init__()
        self.encoder = ENCODERS[encoder](**(encoder_args or {}))
        self.heads = nn.ModuleDict({h: HEADS[h](self.encoder.out_dim, **(head_args or {}).get(h, {})) for h in heads})
        self.affine = None
        if affine:
            self.affine = Affine(**(affine if isinstance(affine, dict) else {}))

    @property
    def stride(self):
        return self.encoder.stride

    def forward(self, gyr, acc):
        """gyr, acc: (B, L, 3), L a multiple of the stride. Returns a dict with 'gyr', 'acc'
        (corrected samples) and every head's raw outputs."""
        L = gyr.shape[1]
        assert L % self.stride == 0, f'window length {L} is not a multiple of the stride {self.stride}'
        out = {'d_gyr': torch.zeros_like(gyr), 'd_acc': torch.zeros_like(acc)}
        if len(self.heads):
            f = self.encoder(torch.cat([gyr, acc / G], -1))
            f = f.repeat_interleave(self.stride, dim=1)         # one feature per sample
            for name, head in self.heads.items():
                o = head(f)
                out.update({k: v for k, v in o.items() if not k.startswith('d_')})
                out['d_gyr'], out['d_acc'] = out['d_gyr'] + o['d_gyr'], out['d_acc'] + o['d_acc']
        if self.affine is not None:
            o = self.affine(gyr, acc)
            out['d_gyr'], out['d_acc'] = out['d_gyr'] + o['d_gyr'], out['d_acc'] + o['d_acc']
        out['gyr'] = gyr + out['d_gyr']
        out['acc'] = acc + out['d_acc']
        return out


def build_model(cfg):
    """cfg: the `model` section of a training config."""
    return IMUDenoiser(cfg['encoder'], cfg.get('encoder_args'), tuple(cfg.get('heads', ['denoising'])),
                       cfg.get('head_args'), cfg.get('affine'))


def load_model(path, device='cpu'):
    """Model and its checkpoint dict (with 'model_cfg' and 'window' metadata)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(ck['model_cfg'])
    model.load_state_dict(ck['state_dict'])
    return model.to(device).eval(), ck
