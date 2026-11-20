"""Inertial odometry module: preintegrated IMU between two node times.

    mode 'raw'        the recorded samples
    mode 'corrected'  the samples after a trained denoiser (models/), computed causally: the
                      sequence is cut into windows of the model's length L that end every `hop`
                      samples, and each window keeps only its last `hop` corrected samples, so a
                      sample's correction never sees the future
"""

import numpy as np
import torch

from kiss_imu.imu.noise import diff_var

from kiss_imu.imu.preintegration import preintegrate
from kiss_imu.models import load_model


@torch.no_grad()
def causal_correction(model, gyr, acc, L, hop, device, batch=256):
    N = len(gyr)
    if N < L:
        raise ValueError(f'sequence shorter ({N}) than the model window ({L})')
    ends = list(range(L, N + 1, hop))
    if ends[-1] != N:
        ends.append(N)
    g_out, a_out = gyr.copy(), acc.copy()
    g_t = torch.tensor(gyr, dtype=torch.float32)
    a_t = torch.tensor(acc, dtype=torch.float32)
    for b0 in range(0, len(ends), batch):
        e = ends[b0:b0 + batch]
        G = torch.stack([g_t[x - L:x] for x in e]).to(device)
        A = torch.stack([a_t[x - L:x] for x in e]).to(device)
        out = model(G, A)
        gc, ac = out['gyr'].double().cpu().numpy(), out['acc'].double().cpu().numpy()
        for n, x in enumerate(e):
            keep = L if x == L else min(hop, x - (ends[b0 + n - 1] if b0 + n > 0 else 0))
            g_out[x - keep:x] = gc[n, L - keep:]
            a_out[x - keep:x] = ac[n, L - keep:]
    return g_out, a_out


class IOModule:
    def __init__(self, seq, cfg, device=None):
        self.mode = cfg.get('mode', 'raw')
        self.ts = seq.imu_t
        self.gyr_noise, self.acc_noise = cfg['gyr_noise'], cfg['acc_noise']
        self.gyr_rw, self.acc_rw = cfg['gyr_bias_rw'], cfg['acc_bias_rw']
        self.cov_scale = cfg.get('cov_scale', 1.0)
        if self.mode == 'raw':
            self.gyr, self.acc = seq.gyr, seq.acc
        elif self.mode == 'corrected':
            device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            model, ck = load_model(cfg['checkpoint'], device)
            w = ck['window']
            if abs(w['rate'] - seq.imu_rate) > 0.05 * w['rate']:
                raise ValueError(f"model trained at {w['rate']:.0f} Hz, sequence is {seq.imu_rate:.0f} Hz")
            hop = int(cfg.get('hop') or w['K'])
            self.gyr, self.acc = causal_correction(model, seq.gyr, seq.acc, w['L'], hop, device)
            noise = cfg.get('corrected_noise') or {}
            self.gyr_noise = noise.get('gyr', self.gyr_noise)
            self.acc_noise = noise.get('acc', self.acc_noise)
        else:
            raise ValueError(f'unknown io.mode {self.mode!r} (raw | corrected)')
        self.raw_gyr, self.raw_acc = seq.gyr, seq.acc
        self.t_switch = None
        self._sv = cfg.get('sample_var') or {}
        self._update_sample_var()

    def _update_sample_var(self):
        # io.sample_var: per-sample IMU variance from the stream's own differences (kiss_imu.imu.noise),
        # e.g. {sensors: [gyr, acc], win_s: 0.1, order: 1}; computed on the stream actually used (raw or corrected)
        self.gyr_var = self.acc_var = None
        sv = self._sv
        if sv.get('enabled'):
            sensors = sv.get('sensors', ['gyr', 'acc'])
            win, order = float(sv.get('win_s', 0.1)), int(sv.get('order', 1))
            if 'gyr' in sensors:
                self.gyr_var = diff_var(self.ts, self.gyr, self.gyr_noise, win, order)
            if 'acc' in sensors:
                self.acc_var = diff_var(self.ts, self.acc, self.acc_noise, win, order)

    def swap(self, model, window, t_switch, device=None, hop=None, noise=None, acc_model=None):
        """Switch to a denoiser's corrected stream from t_switch on (online hand-over, e.g. after warm-start
        training). Samples before t_switch are left as they were; the correction is the causal one of
        io.mode=corrected, so no sample uses the future. window: the checkpoint's {'K', 'L', 'rate'};
        noise: optional {gyr, acc} densities of the corrected stream (as io.corrected_noise).
        acc_model: optional second denoiser (same window) whose accelerometer output is used instead, e.g. the
        gyro from a pretrained model and the accelerometer from its warm-start adaptation."""
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if abs(window['rate'] - 1.0 / np.median(np.diff(self.ts))) > 0.05 * window['rate']:
            raise ValueError(f"model trained at {window['rate']:.0f} Hz, stream differs")
        model = model.to(device).eval()
        g, a = causal_correction(model, self.raw_gyr, self.raw_acc, window['L'], int(hop or window['K']), device)
        if acc_model is not None:
            _, a = causal_correction(acc_model.to(device).eval(), self.raw_gyr, self.raw_acc, window['L'],
                                     int(hop or window['K']), device)
        k = int(np.searchsorted(self.ts, t_switch))
        # the part of the bias the correction now removes: mean (current - corrected) over the last seconds
        # before the switch (causal: past samples only); the PGO biases jump by -that to stay consistent
        k0 = int(np.searchsorted(self.ts, t_switch - 10.0))
        self.switch_offset = np.r_[(self.gyr[k0:k] - g[k0:k]).mean(0), (self.acc[k0:k] - a[k0:k]).mean(0)] \
            if k > k0 else np.zeros(6)
        self.gyr = np.concatenate([self.gyr[:k], g[k:]])
        self.acc = np.concatenate([self.acc[:k], a[k:]])
        if noise:
            self.gyr_noise = noise.get('gyr', self.gyr_noise)
            self.acc_noise = noise.get('acc', self.acc_noise)
        self.mode, self.t_switch = 'switched', float(t_switch)
        self._update_sample_var()                 # trailing windows: values before t_switch are unchanged

    def preintegrate(self, t0, t1, bg, ba):
        return preintegrate(self.ts, self.gyr, self.acc, t0, t1, bg, ba, self.gyr_noise, self.acc_noise,
                            self.gyr_var, self.acc_var)
