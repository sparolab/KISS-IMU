"""Warm-start self-supervision: learn the IMU denoiser from the first seconds of a run, then use it in that run.

Per sequence, one PGO run (offline, deterministic):

    1. raw IMU PGO from the first scan, logging the LO pose + information of every scan
    2. at t0 + warmup_s: pseudo-labels from the log so far (LO poses, invalid = rejected or weakly constrained scans),
       the sequence truncated at that time (no future IMU or scans), denoiser training on it (no GT anywhere)
    3. hand-over at t0 + warmup_s + latency (latency = measured training wall time, or fixed): the IO module
       switches to the causally corrected stream (IOModule.swap); the PGO is never restarted
    4. continue to the end; score against GT, over the whole run and from the hand-over on (PGO and IMU-only
       drift of the raw / pretrained / used streams)

warmstart.pretrained: a checkpoint to start from (trained without GT on other runs); the warm-up then
fine-tunes it (train config: init_checkpoint) instead of training from scratch. warmstart.adapt: false uses
the pretrained model as is, handed over at t0 + warmup_s (0 = from the first step) with no training.
warmstart.streams: which model corrects each sensor after the hand-over, {gyr: adapted|pretrained,
acc: adapted|pretrained} (default both adapted), e.g. the pretrained gyro with the adapted accelerometer.
warmstart.bias_jump: at the hand-over, shift the PGO biases by the mean correction of the last 10 s (the
corrected stream already removes that part; without it the bias is subtracted twice until it re-converges).
warmstart.bias_rw_scale: scales the bias random walk after the hand-over (the PGO then only tracks a residual).

    python3 scripts/run_warmstart.py configs/diter_os/warmstart.yaml --seqs Forest_new --out runs/warmstart
    # baseline (same PGO, raw IMU throughout)
    python3 scripts/run_pgo.py configs/diter_os/pgo.yaml --seqs Forest_new --out runs/raw
"""

import argparse
import copy
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from kiss_imu.config import load_config  # noqa: E402
from kiss_imu.data import Sequence  # noqa: E402
from kiss_imu.eval.drift import attitude_drift, drift_rmse, vel_rmse, velocity_drift  # noqa: E402
from kiss_imu.eval.metrics import evaluate  # noqa: E402
from kiss_imu.pgo.modules.io_module import causal_correction  # noqa: E402
from kiss_imu.models import load_model  # noqa: E402
from kiss_imu.pgo.pipeline import run_sequence  # noqa: E402
from kiss_imu.training.labels import labels_from_steps  # noqa: E402
from kiss_imu.training.label_select import select_labels  # noqa: E402
from kiss_imu.training.trainer import Trainer  # noqa: E402


class WarmStart:
    """run_sequence hook: trains once the warm-up has been seen, swaps the IO stream at the hand-over."""

    def __init__(self, cfg, cfg_path, seq, out_dir, device=None):
        self.cfg, self.w, self.seq, self.out, self.device = cfg, cfg['warmstart'], seq, out_dir, device
        tpath = os.path.join(os.path.dirname(cfg_path), self.w['train_config'])
        if not os.path.exists(tpath):                 # else relative to the repository root
            tpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.w['train_config'])
        over = list(self.w.get('train_overrides') or [])
        self.pretrained = self.w.get('pretrained')
        if self.pretrained:
            over.append(f'train.init_checkpoint={self.pretrained}')
        self.tcfg = load_config(tpath, over)
        self.adapt = bool(self.w.get('adapt', True))
        if not self.adapt and not self.pretrained:
            raise ValueError('warmstart.adapt: false needs warmstart.pretrained')
        self.t0 = self.t_train = self.t_switch = None
        self.model = self.window = None
        self.info = {}

    def __call__(self, pgo, t):
        if self.t0 is None:
            self.t0 = pgo.online[0][0]
        if self.model is None and t >= self.t0 + float(self.w['warmup_s']):
            if self.adapt:
                self._train(pgo, t)
            else:                                     # pretrained model as is, no training: no latency
                self.model, ck = load_model(self.pretrained, self.device or 'cpu')
                self.window, self.t_hand = ck['window'], float(t)
                self.info.update(adapt=False, pretrained=self.pretrained)
        if self.model is not None and self.t_switch is None and t >= self.t_hand:
            gm, am = self._stream_models()
            pgo.io.swap(gm, self.window, t, self.device, noise=self.w.get('corrected_noise'),
                        acc_model=am if am is not gm else None)
            self.t_switch = float(t)
            self.info['t_switch'] = self.t_switch - self.t0
            self.info['switch_offset'] = pgo.io.switch_offset.tolist()
            if self.w.get('bias_jump', False):         # the PGO biases drop what the correction now removes
                pgo.bias_jump = -pgo.io.switch_offset
            s = float(self.w.get('bias_rw_scale', 1.0))  # residual bias only: a tighter random walk
            pgo.io.gyr_rw, pgo.io.acc_rw = pgo.io.gyr_rw * s, pgo.io.acc_rw * s
            print(f'  hand-over to the corrected IMU at +{self.info["t_switch"]:.1f} s', flush=True)

    def _stream_models(self):
        st = self.w.get('streams') or {}
        pick = {}
        for k in ('gyr', 'acc'):
            src = st.get(k, 'adapted')
            if src == 'pretrained' and self.adapt:
                if getattr(self, '_pre_model', None) is None:
                    self._pre_model, ck = load_model(self.pretrained, self.device or 'cpu')
                    if (ck['window']['K'], ck['window']['L']) != (self.window['K'], self.window['L']):
                        raise ValueError('pretrained and adapted windows differ')
                pick[k] = self._pre_model
            elif src in ('adapted', 'pretrained'):
                pick[k] = self.model
            else:
                raise ValueError(f'warmstart.streams.{k}: {src!r} (adapted | pretrained)')
        self.info['streams'] = {k: st.get(k, 'adapted') for k in ('gyr', 'acc')}
        return pick['gyr'], pick['acc']

    def _train(self, pgo, t):
        """Everything here is what the system has at time t: its own log and the data up to t.
        warmstart.label_rounds R > 0: after each training, re-run the PGO over the warm-up with the corrected IMU
        and re-pick every scan pair's label among the current labels and the re-run's LO / smoothed poses by
        point-cloud overlap (KISS-IMU's ICP-vs-PGO label vote, kiss_imu/training/label_select.py), then retrain
        on the new labels; R + 1 trainings in all. warmstart.label_init: scratch (new model each round) or continue
        (fine-tune the previous round's model, warmstart.label_epochs epochs if set)."""
        lc = self.w.get('labels') or {}
        mef = float(lc.get('min_eig_frac', 0.05))
        lab = labels_from_steps(pgo.log, lc.get('source', 'lo'), mef,
                                poses=np.array(pgo.smoothed) if lc.get('source') == 'smoothed' else None)
        P, inv, q = lab['P'], lab['invalid_t'], lab['quality']
        x = pgo.graph.states[pgo.graph.newest_key]          # the PGO's current bias estimate (time-scaling)
        bias_hat = (x.bg.copy(), x.ba.copy()) if pgo.estimate_bias else None
        rounds = int(self.w.get('label_rounds', 0))
        print(f'  warm-up done at +{t - self.t0:.1f} s: {len(P)} labels '
              f'({lab["invalid_frac"]:.1%} invalid), training ...', flush=True)
        tic, rinfo = time.time(), []
        for r in range(rounds + 1):
            seen = self.seq.until(t).set_labels(P, inv, q)
            seen.bias_hat = bias_hat
            tdir = os.path.join(self.out, 'train' if r == 0 else f'train_r{r}')
            tcfg = self.tcfg
            if r > 0 and self.w.get('label_init', 'scratch') == 'continue':   # fine-tune the previous round's model
                tcfg = copy.deepcopy(self.tcfg)
                tcfg['train']['init_checkpoint'] = os.path.join(prev_dir, f"{self.w.get('checkpoint', 'last')}.pt")
                if self.w.get('label_epochs'):
                    tcfg['train']['epochs'] = int(self.w['label_epochs'])
            trainer = Trainer(tcfg, tdir, train_seqs=[seen], val_seqs=[])
            prev_dir = tdir
            hist = trainer.fit()
            dev = self.device or trainer.dev
            self.model, ck = load_model(os.path.join(tdir, f"{self.w.get('checkpoint', 'last')}.pt"), dev)
            self.window = ck['window']
            if r == rounds:
                break
            P, inv, info = self._relabel(t, P, inv, mef, dev)
            info['round'] = r + 1
            rinfo.append(info)
            print(f'  label round {r + 1}: {info}, {len(inv)} invalid', flush=True)
        self.final_labels, self.ckpt, self.bias_hat = (P, inv, q), os.path.join(tdir, f"{self.w.get('checkpoint', 'last')}.pt"), bias_hat
        dt = time.time() - tic
        lat = self.w.get('latency', 'measured')
        latency = dt if lat == 'measured' else float(lat)
        self.t_train, self.t_hand = float(t), float(t) + latency
        self.info.update(adapt=True, pretrained=self.pretrained, t_train=self.t_train - self.t0, train_wall_s=dt, latency_s=latency, n_labels=len(lab['P']),
                         label_invalid_frac=lab['invalid_frac'], train_windows=len(trainer.train_set),
                         final_train_loss=hist[-1]['train_loss'] if hist else None, label_rounds=rinfo)
        print(f'  trained in {dt:.1f} s ({len(trainer.train_set)} windows, final loss '
              f'{self.info["final_train_loss"]:.4g}); hand-over at +{self.t_hand - self.t0:.1f} s', flush=True)

    def _relabel(self, t, P, inv, mef, dev):
        """Re-run the PGO over the warm-up (data up to t only, no GT/labels) with the corrected IMU from the
        first sample, then the per-pair overlap vote between the current labels and the re-run's poses."""
        c = self.seq.until(t)
        c.gt_t = None                                   # nothing to stop at, nothing to read: GT-free re-run
        done = []

        def hook(p, ts):
            if not done:
                p.io.swap(self.model, self.window, float(c.imu_t[0]), dev)
                done.append(True)
        p2 = run_sequence(self.cfg, c, device=self.device, progress=False, hook=hook)
        lo = labels_from_steps(p2.log, 'lo', mef)
        sm = labels_from_steps(p2.log, 'smoothed', mef, poses=np.array(p2.smoothed))
        ls = self.w.get('label_select') or {}
        return select_labels(c, [P, lo['P'], sm['P']], [inv, lo['invalid_t'], sm['invalid_t']],
                             thr=float(ls.get('thr', 0.2)), voxel=float(ls.get('voxel', 0.2)),
                             margin=float(ls.get('margin', 0.002)))


def imu_drift(seq, io, ws, att_h=(10.0, 30.0, 60.0), vel_h=(1.0, 5.0, 10.0)):
    """IMU-only drift against GT from the hand-over on: raw stream, the stream the PGO used, and the
    pretrained model's (when there is one), so adaptation and the pretrained start can be told apart."""
    streams = {'raw': (seq.gyr, seq.acc), 'used': (io.gyr, io.acc)}
    if ws.pretrained and ws.adapt:
        import torch
        m, ck = load_model(ws.pretrained, 'cuda' if torch.cuda.is_available() else 'cpu')
        dev = next(m.parameters()).device
        streams['pretrained'] = causal_correction(m, seq.gyr, seq.acc, ck['window']['L'], ck['window']['K'], dev)
    out = {}
    for k, (g, a) in streams.items():
        att = drift_rmse([attitude_drift(seq, g, att_h, t_from=ws.t_switch)])
        vel = vel_rmse([velocity_drift(seq, a, vel_h, t_from=ws.t_switch)])
        out[k] = {'att_deg': {str(h): v['rot'] for h, v in att.items()}, 'vel_m_s': {str(h): v for h, v in vel.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--seqs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', nargs='*', default=[], dest='overrides', metavar='KEY=VALUE', help='dotted config overrides')
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=1)
    d = cfg['data']
    summary = {}
    for name in a.seqs:
        print(f'== {name}  (warm-start {cfg["warmstart"]["warmup_s"]} s, latency {cfg["warmstart"]["latency"]})',
              flush=True)
        seq = Sequence(d['root'], name, d.get('type', 'generic'), (d.get('gt_time_offset') or {}).get(name, 0.0),
                       min_range=d.get('min_range', 1.0), max_range=d.get('max_range', 100.0), imu_dir=d.get('imu_dir'),
                       despike=d.get('imu_despike'), imu_ext=d.get('imu_ext', False), gt_file=d.get('gt_file', 'gt_pose.csv'),
                       scan_time_offset=d.get('scan_time_offset', 0.0), lidar_rotvec_deg=d.get('lidar_rotvec_deg'))
        out = os.path.join(a.out, name)
        ws = WarmStart(cfg, a.config, seq, out)
        pgo = run_sequence(cfg, seq, out, hook=ws)
        res = {k: evaluate(np.array(getattr(pgo, k)), seq) for k in ('online', 'smoothed')}
        if ws.t_switch is not None:                    # from the hand-over on (the part the model can change)
            on = np.array(pgo.online)
            res['online_after_switch'] = evaluate(on[on[:, 0] >= ws.t_switch], seq)
            res['imu_drift_after_switch'] = imu_drift(seq, pgo.io, ws)
        res['lo_accept_rate'] = float(np.mean([x['lo'] for x in pgo.log]))
        res['ms_per_scan'] = float(np.mean([x['dt_ms'] for x in pgo.log]))
        res['warmstart'] = ws.info
        summary[name] = res
        with open(os.path.join(out, 'result.json'), 'w') as f:
            json.dump(res, f, indent=1)
        o = res['online']
        print(f'   ATE {o["ate_rmse"]:.3f} m (aligned {o["ate_aligned_rmse"]:.3f})  RPE {o["rpe_pct"]:.2f} %  '
              f'rot {o["rot_rmse_deg"]:.2f} deg  over {o["length_m"]:.0f} m  | LO accepted {res["lo_accept_rate"]:.1%}',
              flush=True)
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
