"""GT-supervised training of the IMU denoiser."""

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from kiss_imu.data import Sequence
from kiss_imu.data.windows import GTWindows, HorizonWindows, samples_per_segment
from kiss_imu.models import build_model

from .horizon import HorizonLoss, correct_span
from .losses import SegmentLoss, per_row, rmse, segment_residuals
from .reweight import MotionBalancer, pair_responsibilities, sample_weights, window_components, window_weights
from kiss_imu.eval.drift import attitude_drift, drift_rmse, vel_rmse, velocity_drift


def check_names(names):
    bad = [n for n in names if not isinstance(n, str)]
    if bad:
        raise TypeError(f'sequence names must be strings, got {bad}: quote them in the YAML (1005_00 parses as 100500)')
    return names


def load_sequences(data_cfg, names, label_dir=None):
    off = data_cfg.get('gt_time_offset', {}) or {}
    check_names(list(names) + list(off))
    return [Sequence(data_cfg['root'], n, data_cfg.get('type', 'generic'), off.get(n, 0.0), imu_dir=data_cfg.get('imu_dir'),
                     label_dir=label_dir, despike=data_cfg.get('imu_despike'), imu_ext=data_cfg.get('imu_ext', False),
                     gt_file=data_cfg.get('gt_file', 'gt_pose.csv')) for n in names]


class Trainer:
    def _l2sp_term(self):
        if getattr(self, 'theta0', None) is None:
            return 0.0
        return self.l2sp * sum(((p - self.theta0[n]) ** 2).sum() for n, p in self.model.named_parameters()
                               if p.requires_grad)

    def __init__(self, cfg, out_dir, train_seqs=None, val_seqs=None):
        """train_seqs / val_seqs: Sequence objects to use instead of loading data.train / data.val (e.g. the
        in-memory pseudo-labelled warm-up of scripts/run_warmstart.py)."""
        self.cfg, self.out = cfg, out_dir
        tc = cfg['train']
        torch.manual_seed(tc.get('seed', 0))
        self.dev = tc.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_window = None
        if tc.get('init_checkpoint'):            # fine-tune: architecture and weights from a trained checkpoint
            ck = torch.load(tc['init_checkpoint'], map_location='cpu', weights_only=False)
            cfg['model'] = ck['model_cfg']
            self.model = build_model(cfg['model'])
            self.model.load_state_dict(ck['state_dict'])
            self.init_window = ck['window']
            # train.l2sp: lambda * sum ||theta - theta_init||^2 (L2-SP, Li et al. 2018), keeps a fine-tune near the
            # starting weights (e.g. a warm-start update on new motion that must not forget the warm-up)
            self.l2sp = float(tc.get('l2sp', 0.0))
            self.theta0 = {n: p.detach().clone() for n, p in self.model.named_parameters()} if self.l2sp > 0 else None
        else:
            self.model = build_model(cfg['model'])
        frozen = [n for n, p in self.model.named_parameters() if any(n.startswith(f) for f in tc.get('freeze') or [])]
        for n, p in self.model.named_parameters():
            p.requires_grad_(n not in frozen)
        if frozen:
            print(f'frozen: {len(frozen)} tensors ({", ".join(tc["freeze"])})', flush=True)
        self.model = self.model.to(self.dev)
        if getattr(self, 'theta0', None) is not None:
            self.theta0 = {n: v.to(self.dev) for n, v in self.theta0.items()}

        d = cfg['data']
        # data.label_dir: train on pseudo-labels (tools/make_pseudo_labels.py); val stays on GT for monitoring
        if train_seqs is None:
            train_seqs = load_sequences(d, d['train'], d.get('label_dir'))
            val_seqs = load_sequences(d, d['val']) if d.get('val') else []
        val_seqs = list(val_seqs or [])
        aug = tc.get('augment') or {}
        if aug.get('time_scales'):           # replayed faster / slower copies of every training sequence
            from .augment import time_scaled
            extra = [time_scaled(q, sc, *(getattr(q, 'bias_hat', None) or (None, None)))
                     for q in train_seqs for sc in aug['time_scales'] if float(sc) != 1.0]
            print(f'time-scaling: +{len(extra)} sequences at x{list(aug["time_scales"])}', flush=True)
            train_seqs = list(train_seqs) + extra
        self.mask = aug.get('mask')           # {ratio, p}: input masking in the horizon loss
        rates = {round(s.imu_rate) for s in train_seqs + val_seqs}
        if len(rates) > 1:
            raise ValueError(f'IMU rates differ across sequences {rates}; train one rate per model')
        self.rate = train_seqs[0].imu_rate
        w = cfg['window']
        self.K = samples_per_segment(self.rate, w['segment_s'], self.model.stride)
        self.M = int(w['segments'])
        if self.init_window and (self.init_window['K'], self.init_window['M']) != (self.K, self.M):
            raise ValueError(f'window K={self.K}, M={self.M} differs from the initial checkpoint {self.init_window}')
        self.train_set = GTWindows(train_seqs, self.K, self.M, hop=self.K * int(w.get('hop_segments', self.M // 2)),
                                   span=tuple(d.get('train_span', (0.0, 1.0))))
        rw = cfg.get('reweight') or {}
        horizon = cfg['loss'].get('mode') == 'horizon'
        self.balancer = None
        if horizon:
            if rw.get('enabled') and rw.get('mode', 'soft') not in ('soft', 'balance'):
                raise ValueError("the horizon loss supports reweight.mode 'soft' or 'balance'")
        elif rw.get('enabled') and rw.get('mode', 'soft') == 'balance':     # KISS-IMU recipe, see reweight.py
            comp, cw, Kc, self.reweight_summary = window_components(self.train_set, train_seqs, rw)
            self.train_set.comp = comp
            self.balancer = MotionBalancer(Kc, cw, rw, seed=tc.get('seed', 0))
            print('motion balancing (GMM):', json.dumps(self.reweight_summary), flush=True)
        elif rw.get('enabled'):
            self.train_set.weights, self.reweight_summary = window_weights(self.train_set, train_seqs, rw)
            print('motion reweighting (GMM):', json.dumps(self.reweight_summary), flush=True)
        self.val_seqs = val_seqs
        self.select = tc.get('select', 'segment')          # segment | drift (val gyro-only attitude drift)
        self.drift_h = float(tc.get('drift_horizon_s', 60.0))
        self.vel_h = float(tc.get('vel_horizon_s', 5.0))
        self._vel_raw, self._rot_raw = None, None
        self.val_set = GTWindows(val_seqs, self.K, self.M, hop=self.K * self.M) if val_seqs else None
        print(f'IMU {self.rate:.0f} Hz, segment K={self.K} samples, window L={self.K * self.M}; '
              f'train {len(self.train_set)} windows, val {len(self.val_set) if self.val_set else 0}', flush=True)

        if not horizon:      # the horizon loss builds its own windows and loader below (the anchored segment windows
            #                  can be empty there, e.g. scan stamps off the IMU sample grid by more than anchor_tol)
            self.loader = DataLoader(self.train_set, tc['batch_size'], shuffle=True, drop_last=True,
                                     num_workers=tc.get('workers', 2))
        self.val_loader = (DataLoader(self.val_set, tc['batch_size'], num_workers=tc.get('workers', 2))
                           if self.val_set and len(self.val_set) else None)
        self.loss = SegmentLoss(cfg['loss']['scale'], cfg['loss'].get('weight', {}))
        self.hloss = None
        if cfg['loss'].get('mode') == 'horizon':      # multi-horizon pseudo-label loss, see horizon.py
            hc = cfg['loss']['horizon']
            self.train_set = HorizonWindows(train_seqs, self.K, self.K * self.M, hc.get('window_s', 20.0),
                                            hc.get('hop_s', 10.0), span=tuple(d.get('train_span', (0.0, 1.0))))
            period = float(np.median(np.diff(train_seqs[0].anchor_t if train_seqs[0].anchor_t is not None
                                                else train_seqs[0].imu_t[::self.K])))
            if rw.get('enabled') and rw.get('mode', 'soft') == 'balance':   # KISS-IMU recipe, unit = label pair
                probs, cw, Kc, self.reweight_summary = pair_responsibilities(train_seqs, self.train_set.used_masks(), rw)
                self.train_set.set_sample_probs(probs)
                self.balancer = MotionBalancer(Kc, cw, rw, seed=tc.get('seed', 0))
                print('motion balancing (GMM, per label pair):', json.dumps(self.reweight_summary), flush=True)
            elif rw.get('enabled'):                   # GMM motion weights per sample -> per label pair
                ws, self.reweight_summary = sample_weights(train_seqs, self.train_set.used_masks(), rw)
                self.train_set.set_sample_weights(ws)
                print('motion reweighting (GMM, per sample):', json.dumps(self.reweight_summary), flush=True)
            self.hloss = HorizonLoss(hc, period, self.train_set.NB, cfg['loss'].get('weight', {}))
            self.loader = DataLoader(self.train_set, hc.get('batch_size', 8), shuffle=True, drop_last=True,
                                     num_workers=tc.get('workers', 2))
            noise = self.hloss.fit_noise(DataLoader(self.train_set, 16, shuffle=True, num_workers=2),
                                         self.train_set.ctx, self.train_set.kmax, self.dev)
            self.reweight_summary = {**(getattr(self, 'reweight_summary', None) or {}), 'horizon': self.hloss.summary()}
            print(f'horizon loss: {len(self.train_set)} windows of {self.train_set.NB} labels, '
                  f'{self.hloss.summary()["pairs_per_window"]} pairs/window, horizons {self.hloss.h_s.tolist()} s', flush=True)
            for k, n in noise.items():
                print(f"  noise {k}: label endpoint sd {np.sqrt(n['s_l2']):.3g}, IMU sd(1 s) {np.sqrt(n['b'] + n['c']):.3g}, "
                      f"sd(10 s) {np.sqrt(10 * n['b'] + 100 * n['c']):.3g}, crossover {n['crossover_s']:.2f} s | "
                      f"D {np.sqrt(n['D']).round(5).tolist()}", flush=True)
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=tc['lr'],
                                     weight_decay=tc.get('weight_decay', 1e-4))
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, tc['lr'], total_steps=tc['epochs'] * len(self.loader))

    def _batch(self, b):
        return {k: v.to(self.dev) for k, v in b.items()}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        raw, net = [], []
        for b in loader:
            b = self._batch(b)
            out = self.model(b['gyr'], b['acc'])
            raw.append(segment_residuals(b, b['gyr'], b['acc'], self.K).cpu())
            net.append(segment_residuals(b, out['gyr'], out['acc'], self.K).cpu())
        raw, net = torch.cat(raw), torch.cat(net)
        out = {'raw': rmse(raw), 'net': rmse(net), 'loss': self.loss(net)[0].item()}
        if self.select == 'drift':
            out['drift'] = self.val_drift()
        self.model.train()
        return out

    @torch.no_grad()
    def val_drift(self):
        from kiss_imu.pgo.modules.io_module import causal_correction
        L = self.K * self.M
        per, pv, pv_raw = [], [], []
        acc_head = 'acc' in self.cfg['model'].get('head_args', {}).get('denoising', {}).get('targets', ['gyr', 'acc'])
        for s in self.val_seqs:
            g, a = causal_correction(self.model, s.gyr, s.acc, L, self.K, self.dev)
            per.append(attitude_drift(s, g, (self.drift_h,)))
            if acc_head:
                pv.append(velocity_drift(s, a, (self.vel_h,)))
                if self._vel_raw is None:
                    pv_raw.append(velocity_drift(s, s.acc, (self.vel_h,)))
        out = drift_rmse(per)[self.drift_h]
        if acc_head:
            if self._vel_raw is None:
                self._vel_raw = vel_rmse(pv_raw)[self.vel_h]
            out['vel'] = vel_rmse(pv)[self.vel_h]
            out['vel_raw'] = self._vel_raw
        return out

    def save(self, name, epoch, val):
        torch.save({'state_dict': self.model.state_dict(), 'model_cfg': self.cfg['model'], 'epoch': epoch, 'val': val,
                    'window': {'K': self.K, 'M': self.M, 'L': self.K * self.M, 'rate': self.rate},
                    'data_type': self.cfg['data'].get('type', 'generic')},
                   os.path.join(self.out, name))

    def fit(self):
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, 'config.json'), 'w') as f:
            json.dump(self.cfg, f, indent=1)
        if getattr(self, 'reweight_summary', None):
            with open(os.path.join(self.out, 'reweight.json'), 'w') as f:
                json.dump(self.reweight_summary, f, indent=1)
        best, hist = float('inf'), []
        for ep in range(self.cfg['train']['epochs']):
            tic, losses = time.time(), []
            for b in self.loader:
                b = self._batch(b)
                if self.hloss is not None:
                    hr = self.hloss.residuals(correct_span(self.model, b['gyr'], b['acc'], self.K * self.M, self.K,
                                                           self.mask),
                                              b, self.train_set.kmax)
                    if self.balancer is not None:
                        L, ok = self.hloss.pair_losses(hr, b)
                        loss = self.balancer(self.hloss.pair_components(b)[ok], L[ok])
                        if loss is None:                   # every pair's component has converged
                            self.sched.step()
                            continue
                    else:
                        loss, parts = self.hloss(hr, b)
                    loss = loss + self._l2sp_term()
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train'].get('grad_clip', 10.0))
                    self.opt.step()
                    self.sched.step()
                    losses.append(loss.item())
                    continue
                out = self.model(b['gyr'], b['acc'])
                r = segment_residuals(b, out['gyr'], out['acc'], self.K)
                if self.balancer is not None:
                    loss = self.balancer(b['comp'], per_row(self.loss, r).reshape(-1, self.M).mean(1))
                    if loss is None:                       # every window's component has converged
                        self.sched.step()
                        continue
                else:
                    loss, parts = self.loss(r, b['w'].repeat_interleave(self.M).double())
                loss = loss + self._l2sp_term()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train'].get('grad_clip', 10.0))
                self.opt.step()
                self.sched.step()
                losses.append(loss.item())
            rec = {'epoch': ep, 'train_loss': float(np.mean(losses)) if losses else float('nan'), 'time_s': time.time() - tic}
            if self.balancer is not None:
                self.balancer.end_epoch()
                bl = self.balancer
                rec['balance'] = {'skipped': bl.skipped, 'converged': int(bl.conv.done.sum()) if bl.conv else 0,
                                  'rate': bl.gate.rate.round(3).tolist() if bl.gate else None}
                print(f"        balance: skipped {bl.skipped} batches, converged {rec['balance']['converged']}/{bl.K}", flush=True)
            if self.val_loader is not None:
                ev = self.evaluate(self.val_loader)
                rec['val'] = ev
                n, r = ev['net'], ev['raw']
                print(f"ep {ep:3d} {rec['time_s']:5.0f}s | train {rec['train_loss']:.3f} | val {ev['loss']:.3f} "
                      f"rot {n['rot_deg']:.4f} (raw {r['rot_deg']:.4f}) deg  vel {n['vel']:.4f} (raw {r['vel']:.4f})  "
                      f"pos {n['pos']:.5f} (raw {r['pos']:.5f})", flush=True)
                score = ev['loss']
                if self.select == 'drift':
                    dr = ev['drift']
                    score = dr['rot']
                    msg = f"        val drift {self.drift_h:.0f} s: rot {dr['rot']:.3f} yaw {dr['yaw']:.3f} deg"
                    if 'vel' in dr:                         # gyro + accelerometer: both drifts relative to raw
                        if self._rot_raw is None:
                            from kiss_imu.eval.drift import attitude_drift as _ad
                            self._rot_raw = drift_rmse([_ad(s, s.gyr, (self.drift_h,)) for s in self.val_seqs])[self.drift_h]['rot']
                        score = dr['rot'] / self._rot_raw + dr['vel'] / dr['vel_raw']
                        msg += f" | vel {self.vel_h:.0f} s {dr['vel']:.4f} (raw {dr['vel_raw']:.4f}) m/s | score {score:.3f}"
                    print(msg, flush=True)
            else:
                print(f"ep {ep:3d} {rec['time_s']:5.0f}s | train {rec['train_loss']:.3f}", flush=True)
                score = rec['train_loss']
            hist.append(rec)
            if score < best:
                best = score
                self.save('best.pt', ep, rec.get('val'))
            self.save('last.pt', ep, rec.get('val'))
            with open(os.path.join(self.out, 'history.json'), 'w') as f:
                json.dump(hist, f, indent=1)
        return hist
