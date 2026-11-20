"""Training windows for GT-supervised denoising.

A window is L = K * M consecutive IMU samples, cut into M segments of K samples
(K = segment_s * IMU rate, e.g. 0.1 s, about one LiDAR period: what one PGO edge
preintegrates). For each window the dataset returns the raw samples, their dt and
the GT rotation / position / velocity at the M + 1 segment boundaries.

Windows that leave the GT span or straddle an IMU gap are dropped.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def samples_per_segment(rate, segment_s, stride):
    """K: segment length in samples, rounded to a multiple of the encoder stride."""
    return max(stride, int(round(rate * segment_s / stride)) * stride)


class GTWindows(Dataset):
    def __init__(self, sequences, K, M, hop, gap_factor=3.0, span=(0.0, 1.0), anchor_tol=0.004):
        """span: keep only windows inside this fraction of each sequence's GT time span (low-data runs).

        Sequences with label anchors (seq.anchor_t, e.g. pseudo-labels that exist only at scan stamps) get
        windows that start at an anchor and keep only windows whose every segment boundary lies within
        anchor_tol of an anchor, so the boundary poses are not interpolated across a scan period (slerp
        over 0.1 s low-passes the legged gait and the labels under-rotate)."""
        assert hop % K == 0, 'hop must be a multiple of the segment length'
        self.K, self.M, self.L = K, M, K * M
        self.data, self.items = [], []
        self.weights = None          # optional per-window loss weights (set by the trainer)
        self.comp = None             # optional per-window motion component id (MotionBalancer)
        self.own = {}                # item index -> (R, p, v) boundary labels (anchored windows)
        for seq in sequences:
            if getattr(seq, 'anchor_t', None) is not None:
                self._add_anchored(seq, hop // K, gap_factor, span, anchor_tol)
                continue
            ts = seq.imu_t
            dt = np.diff(ts)
            bad = np.r_[dt > gap_factor * np.median(dt), False]
            bidx = np.arange(0, len(ts), K)                     # boundaries at every K-th sample
            ok = seq.in_gt_span(ts[bidx], margin=0.05)
            Rb, pb, vb = seq.gt_states(ts[bidx])
            mi = len(self.data)
            self.data.append(dict(
                gyr=torch.tensor(seq.gyr[:-1], dtype=torch.float32),
                acc=torch.tensor(seq.acc[:-1], dtype=torch.float32),
                dt=torch.tensor(dt, dtype=torch.float64),
                R=torch.tensor(Rb), p=torch.tensor(pb), v=torch.tensor(vb)))
            cum_bad = np.r_[0, np.cumsum(bad)]
            T0, T1 = seq.gt_t[0], seq.gt_t[-1]
            lo_t, hi_t = T0 + span[0] * (T1 - T0), T0 + span[1] * (T1 - T0)
            for s in range(0, len(ts) - self.L - 1, hop):
                b0 = s // K
                if b0 + self.M >= len(bidx) or not ok[b0:b0 + self.M + 1].all():
                    continue
                if cum_bad[s + self.L] - cum_bad[s] > 0:
                    continue
                if ts[s] < lo_t or ts[s + self.L] > hi_t:
                    continue
                self.items.append((mi, s))

    def _add_anchored(self, seq, hop_segments, gap_factor, span, tol):
        ts = seq.imu_t
        dt = np.diff(ts)
        bad = np.r_[dt > gap_factor * np.median(dt), False]
        cum_bad = np.r_[0, np.cumsum(bad)]
        mi = len(self.data)
        self.data.append(dict(gyr=torch.tensor(seq.gyr[:-1], dtype=torch.float32),
                              acc=torch.tensor(seq.acc[:-1], dtype=torch.float32),
                              dt=torch.tensor(dt, dtype=torch.float64)))
        a = seq.anchor_t
        T0, T1 = seq.gt_t[0], seq.gt_t[-1]
        lo_t, hi_t = T0 + span[0] * (T1 - T0), T0 + span[1] * (T1 - T0)
        for j in range(0, len(a) - self.M - 1, max(1, hop_segments)):
            s = int(np.searchsorted(ts, a[j]))
            if s + self.L >= len(ts) - 1:
                break
            bt = ts[s + np.arange(self.M + 1) * self.K]
            k = np.searchsorted(a, bt).clip(1, len(a) - 1)
            near = np.minimum(np.abs(bt - a[k - 1]), np.abs(bt - a[k]))
            if near.max() > tol or not seq.in_gt_span(bt, margin=0.05).all():
                continue
            if cum_bad[s + self.L] - cum_bad[s] > 0 or bt[0] < lo_t or bt[-1] > hi_t:
                continue
            R, p, v = seq.gt_states(bt)
            self.own[len(self.items)] = (torch.tensor(R), torch.tensor(p), torch.tensor(v))
            self.items.append((mi, s))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        mi, s = self.items[i]
        d = self.data[mi]
        if i in self.own:
            R, p, v = self.own[i]
        else:
            b = slice(s // self.K, s // self.K + self.M + 1)
            R, p, v = d['R'][b], d['p'][b], d['v'][b]
        return dict(gyr=d['gyr'][s:s + self.L], acc=d['acc'][s:s + self.L], dt=d['dt'][s:s + self.L],
                    R=R, p=p, v=v,
                    w=torch.tensor(1.0 if self.weights is None else float(self.weights[i])),
                    comp=torch.tensor(-1 if self.comp is None else int(self.comp[i])))


class HorizonWindows(Dataset):
    """Long windows for multi-horizon supervision (pseudo-labels or GT).

    A window holds NB label boundaries (label stamps for pseudo-labels, every K-th sample for GT) spanning
    ~span_s seconds, the IMU samples between them and L - K samples of context before the first boundary, so
    the model can correct every K-sample block from a full L-sample window as at run time (causal_correction).
    Segments between boundaries have variable length (<= kmax samples) and are preintegrated padded with dt = 0.

    Item: gyr / acc / dt (ctx + S_pad samples, zero-padded), b (NB,) boundary sample offsets after the context,
    R / p / v (NB) labels at the boundaries, valid (NB,) boundary usable, q (NB,) LO quality (1 = typical),
    swc (NB,) cumulative per-sample loss weight at the boundaries (sample count without reweighting).
    """

    def __init__(self, sequences, K, L, span_s=20.0, hop_s=10.0, gap_factor=3.0, span=(0.0, 1.0), tol=0.003,
                 kmax_factor=2.1):
        self.K, self.L, self.ctx = K, L, L - K
        self.kmax = int(np.ceil(kmax_factor * K))
        self.data, self.items = [], []
        self.NB = None
        for seq in sequences:
            ts = seq.imu_t
            dt = np.diff(ts)
            bad = np.r_[dt > gap_factor * np.median(dt), False]
            cum_bad = np.r_[0, np.cumsum(bad)]
            if getattr(seq, 'anchor_t', None) is not None:
                bt = seq.anchor_t
                bt = bt[(bt > ts[0]) & (bt < ts[-1])]
                i = np.searchsorted(ts, bt).clip(1, len(ts) - 1)
                idx = np.where(np.abs(ts[i - 1] - bt) < np.abs(ts[i] - bt), i - 1, i)
                # a label within tol of an IMU sample, or the nearest sample at a coarse IMU rate (100 Hz: up to
                # 5 ms off; a scan clock offset can put every stamp between samples)
                tol_s = max(tol, 0.5 / seq.imu_rate + 1e-6)
                keep = (np.abs(ts[idx] - bt) < tol_s) & np.r_[True, np.diff(idx) > 0]
                bt, idx = bt[keep], idx[keep]
            else:
                idx = np.arange(0, len(ts), K)
                bt = ts[idx]
            if len(bt) < 3:
                continue
            period = np.median(np.diff(bt))
            if self.NB is None:
                self.NB = int(round(span_s / period)) + 1
                self.S_pad = int(np.ceil((self.NB - 1) * period * seq.imu_rate * 1.05 / K + 2)) * K
            NB, S_pad = self.NB, self.S_pad
            valid = seq.in_gt_span(bt, margin=0.05)
            q = seq.label_quality(bt)
            R, p, v = seq.gt_states(bt)
            mi = len(self.data)
            self.data.append(dict(
                gyr=torch.tensor(seq.gyr, dtype=torch.float32), acc=torch.tensor(seq.acc, dtype=torch.float32),
                dt=torch.tensor(np.r_[dt, 0.0], dtype=torch.float64), idx=idx, R=torch.tensor(R), p=torch.tensor(p),
                v=torch.tensor(v), valid=torch.tensor(valid), q=torch.tensor(q),
                swc=torch.arange(len(ts) + 1, dtype=torch.float64)))     # cumulative sample weight (1 each)
            seg = np.diff(idx)
            T0, T1 = bt[0], bt[-1]
            lo_t, hi_t = T0 + span[0] * (T1 - T0), T0 + span[1] * (T1 - T0)
            hop = max(1, int(round(hop_s / period)))
            for m0 in range(0, len(bt) - NB + 1, hop):
                m1 = m0 + NB - 1
                s0 = idx[m0] - self.ctx
                if s0 < 0 or idx[m1] - idx[m0] > S_pad or s0 + self.ctx + S_pad > len(ts):
                    continue
                if seg[m0:m1].max() > self.kmax or cum_bad[idx[m1]] - cum_bad[s0] > 0:
                    continue
                if bt[m0] < lo_t or bt[m1] > hi_t or valid[m0:m1 + 1].mean() < 0.5:
                    continue
                self.items.append((mi, m0))

    def used_masks(self):
        """Per-sequence masks of the IMU samples the windows cover (context included)."""
        used = [np.zeros(len(d['gyr']), bool) for d in self.data]
        for mi, m0 in self.items:
            idx = self.data[mi]['idx']
            used[mi][idx[m0] - self.ctx:idx[m0 + self.NB - 1] + 1] = True
        return used

    def set_sample_weights(self, ws):
        """Per-sample loss weights (reweight.sample_weights): a label pair (i, j) is then weighted by the mean
        weight of the samples between its boundaries (HorizonLoss.weights)."""
        for d, w in zip(self.data, ws):
            d['swc'] = torch.tensor(np.r_[0.0, np.cumsum(np.asarray(w, np.float64))])

    def set_sample_probs(self, probs):
        """Per-sample GMM responsibilities (N, Kc) (reweight.pair_responsibilities), kept cumulatively so a
        label pair's mean responsibility over its span is one difference (HorizonLoss.pair_components)."""
        for d, p in zip(self.data, probs):
            d['pc'] = torch.tensor(np.r_[np.zeros((1, p.shape[1])), np.cumsum(np.asarray(p, np.float64), 0)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        mi, m0 = self.items[i]
        d = self.data[mi]
        sl = slice(m0, m0 + self.NB)
        idx = d['idx'][sl]
        s0 = idx[0] - self.ctx
        n = self.ctx + self.S_pad
        dt = d['dt'][s0:s0 + n].clone()
        dt[self.ctx + idx[-1] - idx[0]:] = 0.0                  # nothing after the last boundary
        return dict(gyr=d['gyr'][s0:s0 + n], acc=d['acc'][s0:s0 + n], dt=dt,
                    b=torch.tensor(idx - idx[0], dtype=torch.long),
                    R=d['R'][sl], p=d['p'][sl], v=d['v'][sl], valid=d['valid'][sl], q=d['q'][sl],
                    swc=d['swc'][torch.as_tensor(idx)],
                    **({'pc': d['pc'][torch.as_tensor(idx)]} if 'pc' in d else {}))
