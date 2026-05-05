"""Evaluate one or more KISS-IMU checkpoints on user-specified sequences.

Two modes:
  (1) --ckpt PATH         : evaluate a single .ckpt
  (2) --ckpt-dir PATH     : scan dir for *.ckpt and pick the best one
                            using --select-metric ({ape, rpe_trans, rpe_rot, balanced})

The evaluator slides a fixed window over each sequence and reports
endpoint RPE (translation, rotation) and last-step APE.
"""

import os
import sys
import glob
import ast
import argparse
import numpy as np
import pypose as pp

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.eval_dataset import SeqDataset
from training.integrator import IMUIntegrator
from models.imu_net import IMUNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, required=True,
                   help='dataset root containing <SEQ>/imu.csv, gt_pose.csv, points/')
    p.add_argument('--data-type', type=str, required=True,
                   choices=['mulran', 'yeoncheon', 'kitti', 'diter++', 'diter_os',
                            'tailrobot', 'kimera', 'botanic_velodyne'])
    p.add_argument('--eval-seqs', nargs='+', type=str, required=True)

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--ckpt',     type=str, help='path to a single .ckpt')
    grp.add_argument('--ckpt-dir', type=str, help='dir containing *.ckpt files')

    p.add_argument('--device',     type=str, default='cuda:0')
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--window-size', type=int, default=200,
                   help='IMU window size used for sliding-window evaluation')
    p.add_argument('--result-dir',  type=str, default='eval_results')

    p.add_argument('--select-metric', type=str, default='ape',
                   choices=['ape', 'rpe_trans', 'rpe_rot', 'balanced'])
    p.add_argument('--balance-mode',  type=str, default='euclid',
                   choices=['euclid', 'minimax', 'weighted'])
    p.add_argument('--balance-weights', type=str, default='(1.0,1.0,1.0)',
                   help='(w_trans, w_rot, w_ape)')

    p.add_argument('--save-plot', action='store_true',
                   help='save endpoint XY plot for each (ckpt, seq) pair')
    return p.parse_args()


def as_xyz1d(t: torch.Tensor) -> np.ndarray:
    return t.reshape(-1).detach().cpu().numpy()[:3]


@torch.no_grad()
def eval_state(data_loader, integrator, network, device, save_plot_path=None):
    sum_rpe_t, sum_rpe_r, sum_ape, n = 0.0, 0.0, 0.0, 0
    pred_last, gt_last = [], []

    for sample in data_loader:
        init_pos = sample['gt_pose0'][0][:3].clone().to(device).float()
        init_rot = sample['gt_pose0'][0][3:].clone().to(device).float()
        init_vel = sample['gt_velocity'][0].clone().to(device).float()
        init_rot = init_rot / (torch.linalg.norm(init_rot) + 1e-12)
        init_state = {'rot': pp.SO3(init_rot), 'vel': init_vel, 'pos': init_pos, 'cov': None}

        corr = network(sample)
        out_state, _ = integrator.integrate(
            init_state,
            corr['dts'][0].unsqueeze(0),
            corr['accels_corr'][0].unsqueeze(0),
            corr['gyros_corr'][0].unsqueeze(0),
            corr['acc_cov'][0].unsqueeze(0),
            corr['gyr_cov'][0].unsqueeze(0),
        )
        pred_se3 = pp.SE3(torch.cat([out_state['pos'], out_state['rot'].tensor()], dim=-1)).to(device)

        gt0 = sample['gt_pose0'][0].to(device).float()
        gt_seq = sample['gt_pose1'].to(device).float()
        if gt_seq.ndim == 3:
            gt_seq = gt_seq[0]
        gtT = gt_seq[-1]

        GT0 = pp.SE3(torch.cat([gt0[:3],  gt0[3:]], dim=-1)).to(device)
        GTT = pp.SE3(torch.cat([gtT[:3],  gtT[3:]], dim=-1)).to(device)

        E = (GT0.Inv() * pred_se3).Inv() * (GT0.Inv() * GTT)
        sum_rpe_t += E.translation().norm().item()
        sum_rpe_r += E.rotation().Log().norm().item() * 180.0 / np.pi
        sum_ape   += (pred_se3.translation() - GTT.translation()).norm().item()
        n += 1

        pred_last.append(as_xyz1d(pred_se3.translation()))
        gt_last.append(as_xyz1d(GTT.translation()))

    if n == 0:
        return {'rpe_trans': float('inf'), 'rpe_rot': float('inf'),
                'ape': float('inf'), 'n_windows': 0}

    if save_plot_path is not None and len(pred_last) > 0:
        pred_np = np.stack(pred_last, axis=0)
        gt_np   = np.stack(gt_last,   axis=0)
        plt.figure()
        plt.plot(gt_np[:, 0],   gt_np[:, 1],   '-', label='GT last')
        plt.plot(pred_np[:, 0], pred_np[:, 1], '-', label='Pred last')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x [m]'); plt.ylabel('y [m]')
        plt.legend(); plt.title('Endpoint positions per window (XY)')
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        plt.savefig(save_plot_path, dpi=200)
        plt.close()

    return {
        'rpe_trans': sum_rpe_t / n,
        'rpe_rot':   sum_rpe_r / n,
        'ape':       sum_ape   / n,
        'n_windows': n,
    }


def evaluate_ckpt(ckpt_path, args):
    network = IMUNet(prop_cov=True, device=args.device)
    state = torch.load(ckpt_path, map_location=args.device)
    network.load_state_dict(state['network'])
    network.to(args.device).eval()

    per_seq = {}
    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    for seq in args.eval_seqs:
        ds = SeqDataset(args.data_root, seq, args.data_type, args.window_size)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
        integrator = IMUIntegrator(init_state=ds.init, prop_cov=True,
                                   gravity=ds.gravity, device=args.device)
        plot_path = (os.path.join(args.result_dir, f'{ckpt_tag}_{seq}_xy.png')
                     if args.save_plot else None)
        per_seq[seq] = eval_state(dl, integrator, network, args.device, plot_path)

    avg = {k: float(np.mean([per_seq[s][k] for s in args.eval_seqs]))
           for k in ['rpe_trans', 'rpe_rot', 'ape']}
    return per_seq, avg


def parse_weights(s):
    try:
        t = ast.literal_eval(s)
        if isinstance(t, (list, tuple)):
            if len(t) == 3:  return tuple(map(float, t))
            if len(t) == 2:  return (float(t[0]), float(t[1]), 1.0)
    except Exception:
        pass
    return (1.0, 1.0, 1.0)


def compute_norm_stats(all_results):
    keys = ['rpe_trans', 'rpe_rot', 'ape']
    mins = {k: min(r['avg'][k] for r in all_results.values()) for k in keys}
    maxs = {k: max(r['avg'][k] for r in all_results.values()) for k in keys}
    denoms = {k: (maxs[k] - mins[k] if maxs[k] > mins[k] else 1.0) for k in keys}
    return mins, denoms


def balanced_score(avg, mins, denoms, mode, weights):
    nt = (avg['rpe_trans'] - mins['rpe_trans']) / denoms['rpe_trans']
    nr = (avg['rpe_rot']   - mins['rpe_rot'])   / denoms['rpe_rot']
    na = (avg['ape']       - mins['ape'])       / denoms['ape']
    wt, wr, wa = weights
    if mode == 'weighted':
        return wt * nt + wr * nr + wa * na
    if mode == 'minimax':
        return max(wt * nt, wr * nr, wa * na)
    return (wt * nt) ** 2 + (wr * nr) ** 2 + (wa * na) ** 2


def select_score(avg, key, mins=None, denoms=None, mode='euclid', weights=(1, 1, 1)):
    if key in ('ape', 'rpe_trans', 'rpe_rot'):
        return avg[key]
    return balanced_score(avg, mins, denoms, mode, weights)


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    if args.ckpt is not None:
        ckpt_paths = [args.ckpt]
    else:
        if not os.path.isdir(args.ckpt_dir):
            print(f"[ERROR] no such ckpt dir: {args.ckpt_dir}", file=sys.stderr)
            sys.exit(1)
        ckpt_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, '*.ckpt')))
        if len(ckpt_paths) == 0:
            print(f"[ERROR] no *.ckpt under: {args.ckpt_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] evaluating {len(ckpt_paths)} ckpt(s) on {len(args.eval_seqs)} seq(s)")
    all_results = {}
    for p in ckpt_paths:
        per_seq, avg = evaluate_ckpt(p, args)
        all_results[os.path.basename(p)] = {'per_seq': per_seq, 'avg': avg}
        print(f"  - {os.path.basename(p)}  "
              f"rpe_trans={avg['rpe_trans']:.3f} m  "
              f"rpe_rot={avg['rpe_rot']:.3f} deg  "
              f"ape={avg['ape']:.3f} m")

    if len(ckpt_paths) > 1:
        mins, denoms = compute_norm_stats(all_results)
        weights = parse_weights(args.balance_weights)

        best_name, best_score = None, float('inf')
        for name, res in all_results.items():
            s = select_score(res['avg'], args.select_metric,
                             mins, denoms, args.balance_mode, weights)
            if s < best_score:
                best_score, best_name = s, name

        print("\n================ best checkpoint ================")
        suffix = f' ({args.balance_mode}, weights={weights})' if args.select_metric == 'balanced' else ''
        print(f"select-metric : {args.select_metric}{suffix}")
        print(f"best ckpt     : {best_name}")
        avg = all_results[best_name]['avg']
        print(f"averaged      : rpe_trans={avg['rpe_trans']:.3f} m | "
              f"rpe_rot={avg['rpe_rot']:.3f} deg | ape={avg['ape']:.3f} m")
        print("\n========== per-seq detail (best ckpt) ==========")
        for seq in args.eval_seqs:
            m = all_results[best_name]['per_seq'][seq]
            print(f"  {seq:>20s}  rpe_trans={m['rpe_trans']:.3f} m | "
                  f"rpe_rot={m['rpe_rot']:.3f} deg | ape={m['ape']:.3f} m | "
                  f"windows={m['n_windows']}")
    else:
        name = list(all_results.keys())[0]
        print("\n========== per-seq detail ==========")
        for seq in args.eval_seqs:
            m = all_results[name]['per_seq'][seq]
            print(f"  {seq:>20s}  rpe_trans={m['rpe_trans']:.3f} m | "
                  f"rpe_rot={m['rpe_rot']:.3f} deg | ape={m['ape']:.3f} m | "
                  f"windows={m['n_windows']}")


if __name__ == '__main__':
    main()
