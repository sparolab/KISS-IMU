"""Evaluate a single KISS-IMU checkpoint on user-specified sequences.

Slides a fixed-size window over each sequence and reports endpoint RPE
(translation, rotation) and last-step APE. Training already selects the
best checkpoint by validation, so this script intentionally evaluates one
ckpt only — pass the path to `best_model.ckpt` in --ckpt.
"""

import os
import sys
import argparse
import numpy as np
import pypose as pp

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.eval_dataset import SeqDataset, collate_fn
from training.integrator import IMUIntegrator
from models.imu_net import IMUNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, required=True,
                   help='dataset root containing <SEQ>/imu.csv, gt_pose.csv, points/')
    # Only diter_os / diter++ have dedicated handling; add your own data-type
    # branch in data/seq_dataset.py (and here) for other datasets.
    p.add_argument('--data-type', type=str, required=True,
                   choices=['diter_os', 'diter++'])
    p.add_argument('--eval-seqs', nargs='+', type=str, required=True)

    p.add_argument('--ckpt', type=str, required=True,
                   help='path to the checkpoint to evaluate (best_model.ckpt)')

    p.add_argument('--device',     type=str, default='cuda:0')
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--window-size', type=int, default=200,
                   help='IMU window size used for sliding-window evaluation')
    p.add_argument('--result-dir',  type=str, default='eval_results')

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
        out_state = integrator.integrate(
            init=init_state,
            dts=corr['dts'], accels=corr['accels_corr'], gyros=corr['gyros_corr'],
            cov_accels=corr['acc_cov'], cov_gyros=corr['gyr_cov'],
            motion_mode=False,
        )
        pred_traj = pp.SE3(torch.cat([out_state['pos'], out_state['rot'].tensor()], dim=-1)).to(device)
        pred_se3 = pred_traj[-1]   # endpoint pose of the window

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
        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=150, bbox_inches='tight')
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
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn)
        integrator = IMUIntegrator(init_state=ds.init, prop_cov=True,
                                   gravity=ds.gravity, device=args.device)
        plot_path = (os.path.join(args.result_dir, f'{ckpt_tag}_{seq}_xy.png')
                     if args.save_plot else None)
        per_seq[seq] = eval_state(dl, integrator, network, args.device, plot_path)

    avg = {k: float(np.mean([per_seq[s][k] for s in args.eval_seqs]))
           for k in ['rpe_trans', 'rpe_rot', 'ape']}
    return per_seq, avg


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    if not os.path.isfile(args.ckpt):
        print(f"[ERROR] no such ckpt: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] evaluating {args.ckpt} on {len(args.eval_seqs)} seq(s)")
    per_seq, avg = evaluate_ckpt(args.ckpt, args)

    print(f"\naveraged: rpe_trans={avg['rpe_trans']:.3f} m | "
          f"rpe_rot={avg['rpe_rot']:.3f} deg | ape={avg['ape']:.3f} m")
    print("\n========== per-seq detail ==========")
    for seq in args.eval_seqs:
        m = per_seq[seq]
        print(f"  {seq:>20s}  rpe_trans={m['rpe_trans']:.3f} m | "
              f"rpe_rot={m['rpe_rot']:.3f} deg | ape={m['ape']:.3f} m | "
              f"windows={m['n_windows']}")


if __name__ == '__main__':
    main()
