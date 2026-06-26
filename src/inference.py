"""Run ICP + PGO inference on top of a trained KISS-IMU checkpoint.

Pipeline (one window at a time, same as training):
    1. network(sample)             — corrected IMU
    2. IMU integrator              — IMU-only trajectory + per-window motion
    3. lo_model(sample, ...)       — ICP poses, motions, overlap scores
    4. optimize(...)               — PGO refinement using IMU+ICP factors

For every evaluation sequence we save:
    <out-dir>/<seq>/inference.npz   imu/icp/pgo/gt poses + per-window overlap
    <out-dir>/<seq>/trajectory.png  top-down XY plot (imu / icp / pgo / gt)

This script does NOT compute RPE/APE — that's evaluate.py's job. It exists
so you can keep the actual trajectories around (for plotting, for offline
analysis, for feeding into a downstream stage that needs a corrected pose
stream rather than a metric).
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import pypose as pp
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.seq_dataset import SeqDataset
from data.collate import collate_fn
from training.integrator import IMUIntegrator
from models.imu_net import IMUNet
from models.lo_module import LOModule
from models.pvgo import optimize
from utils.overlap_score import batch_pose_aligned_overlap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, required=True)
    p.add_argument('--data-type', type=str, required=True)
    p.add_argument('--seqs',      nargs='+', type=str, required=True,
                   help='one or more sequence names to run inference on')
    p.add_argument('--ckpt',      type=str, required=True,
                   help='path to best_model.ckpt')

    p.add_argument('--lo-model',  type=str, default='kiss_icp',
                   choices=['kiss_icp', 'fast_gicp', 'small_gicp'])
    p.add_argument('--lm-weight', type=str, default='(1,0.1,1,0.1,0.1)',
                   help='PGO factor weights as a python tuple literal')
    p.add_argument('--use-submap', action='store_true')
    p.add_argument('--use-adaptive-weight', action='store_true',
                   help='weight ICP factors by overlap score and IMU factors '
                        'by their integrated covariance during PGO')

    p.add_argument('--out-dir',     type=str, required=True)
    p.add_argument('--device',      type=str, default='cuda:0')
    p.add_argument('--batch-size',  type=int, default=5)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--no-plot',     action='store_true',
                   help='skip the matplotlib top-down trajectory plot')
    return p.parse_args()


@torch.no_grad()
def run_sequence(seq: str, network: IMUNet, args) -> dict:
    ds = SeqDataset(data_root=args.data_root, data_seq=seq, data_type=args.data_type)
    dl = DataLoader(dataset=ds, batch_size=args.batch_size,
                    num_workers=args.num_workers, shuffle=False,
                    drop_last=True, collate_fn=collate_fn)

    integrator = IMUIntegrator(init_state=ds.init, device=args.device)
    gravity = ds.gravity

    init_pos = ds.init['pos'].numpy() if torch.is_tensor(ds.init['pos']) else ds.init['pos']
    init_rot = ds.init['rot'].numpy() if torch.is_tensor(ds.init['rot']) else ds.init['rot']
    init_vel = ds.init['vel']
    init_pose7 = torch.from_numpy(np.concatenate([init_pos, init_rot])).float()

    lo_model = LOModule(lo_model=args.lo_model,
                        T_I_L=ds.T_I_L, R_I_L=ds.R_I_L,
                        init_state=init_pose7,
                        device_id=args.device,
                        use_submap=args.use_submap)

    import ast
    lm_weight = list(ast.literal_eval(args.lm_weight))

    anchor_pose = init_pose7.clone()
    anchor_vel  = init_vel.clone() if torch.is_tensor(init_vel) else torch.from_numpy(init_vel).float()

    imu_poses_all, icp_poses_all, pgo_poses_all, gt_poses_all = [], [], [], []
    icp_overlap_all, pgo_overlap_all = [], []

    for sample in tqdm(dl, desc=f"inference[{seq}]"):
        init_pos = anchor_pose[:3].clone().to(args.device).float()
        init_rot = anchor_pose[3:].clone().to(args.device).float()
        init_rot = init_rot / (torch.linalg.norm(init_rot) + 1e-12)
        init_v   = anchor_vel.clone().to(args.device).float()
        init_state = {'rot': pp.SO3(init_rot), 'vel': init_v,
                      'pos': init_pos, 'cov': None}

        corr_data = network(sample)
        imu_states = integrator.integrate(
            init=init_state,
            dts=corr_data['dts'], accels=corr_data['accels_corr'],
            gyros=corr_data['gyros_corr'],
            cov_accels=corr_data['acc_cov'], cov_gyros=corr_data['gyr_cov'],
            motion_mode=False,
        )
        zero_init = {'rot': pp.SO3(init_rot),
                     'vel': torch.zeros((1, 3), device=args.device),
                     'pos': torch.zeros((1, 3), device=args.device), 'cov': None}
        imu_dstates = integrator.integrate(
            init=zero_init,
            dts=corr_data['dts'], accels=corr_data['accels_corr'],
            gyros=corr_data['gyros_corr'],
            cov_accels=corr_data['acc_cov'], cov_gyros=corr_data['gyr_cov'],
            motion_mode=True,
        )

        imu_nodes = pp.SE3(torch.cat([imu_states['pos'], imu_states['rot'].tensor()], dim=-1)).to(args.device)
        imu_vels  = imu_states['vel'].to(args.device)
        imu_dcovs = imu_dstates['cov'].detach().to(args.device)
        imu_dts   = torch.stack([d.sum() for d in corr_data['dts']]).unsqueeze(-1).to(args.device)

        icp_poses, icp_motions, icp_overlap = lo_model(sample, pp.SE3(anchor_pose.to(args.device)))
        if args.use_adaptive_weight:
            icp_w = torch.from_numpy(np.asarray(icp_overlap))
            imu_w = imu_dcovs.squeeze(1)
        else:
            icp_w = None
            imu_w = None
        pgo_poses, pgo_vels = optimize(
            nodes=imu_nodes, vels=imu_vels,
            icp_factors=icp_motions,
            imu_drots=imu_dstates['rot'], imu_dvels=imu_dstates['vel'],
            imu_dtrans=imu_dstates['pos'], imu_dts=imu_dts,
            weights=lm_weight, gravity=gravity,
            icp_weights=icp_w, imu_weights=imu_w, device=args.device,
        )
        pgo_motions = pgo_poses[:-1].Inv() @ pgo_poses[1:]
        pgo_overlap = batch_pose_aligned_overlap(
            lo_model.scans0_np, lo_model.scans1_np, pgo_motions,
        )

        imu_poses_all.append(imu_nodes.tensor()[1:].detach().cpu().numpy())
        icp_poses_all.append(icp_poses.tensor()[1:].detach().cpu().numpy())
        pgo_poses_all.append(pgo_poses.tensor()[1:].detach().cpu().numpy())
        # gt_pose1 is one endpoint per window (B rows), with no prepended anchor
        # node — unlike imu/icp/pgo whose [1:] strips the anchor. Keep all B rows
        # so every stream has the same length.
        gt_poses_all.append(sample['gt_pose1'].detach().cpu().numpy())
        icp_overlap_all.append(np.asarray(icp_overlap))
        pgo_overlap_all.append(np.asarray(pgo_overlap))

        anchor_pose = pgo_poses.tensor()[-1].detach().cpu()
        anchor_vel  = pgo_vels[-1].detach().cpu()

    return {
        'imu_poses': np.concatenate(imu_poses_all, axis=0) if imu_poses_all else np.zeros((0, 7)),
        'icp_poses': np.concatenate(icp_poses_all, axis=0) if icp_poses_all else np.zeros((0, 7)),
        'pgo_poses': np.concatenate(pgo_poses_all, axis=0) if pgo_poses_all else np.zeros((0, 7)),
        'gt_poses':  np.concatenate(gt_poses_all,  axis=0) if gt_poses_all  else np.zeros((0, 7)),
        'icp_overlap': np.concatenate(icp_overlap_all, axis=0) if icp_overlap_all else np.zeros((0,)),
        'pgo_overlap': np.concatenate(pgo_overlap_all, axis=0) if pgo_overlap_all else np.zeros((0,)),
    }


def plot_xy(result: dict, out_png: Path, seq: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for name, color in [('gt_poses', 'k'), ('imu_poses', 'tab:blue'),
                        ('icp_poses', 'tab:red'), ('pgo_poses', 'tab:green')]:
        arr = result.get(name)
        if arr is None or len(arr) == 0:
            continue
        label = name.replace('_poses', '').upper()
        ax.plot(arr[:, 0], arr[:, 1], '-', color=color, linewidth=1.5, label=label)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'{seq} — inference trajectories (XY)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        print(f"[inference] ckpt not found: {ckpt_path}", file=sys.stderr)
        return 1

    network = IMUNet(prop_cov=True, device=args.device).to(args.device)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    network.load_state_dict(ckpt['network'] if 'network' in ckpt else ckpt)
    network.eval()
    print(f"[inference] loaded ckpt: {ckpt_path}")

    for seq in args.seqs:
        seq_out = out_root / seq
        seq_out.mkdir(parents=True, exist_ok=True)
        result = run_sequence(seq, network, args)

        npz_path = seq_out / "inference.npz"
        np.savez(npz_path, **result)
        print(f"[inference] {seq}: saved {npz_path}  "
              f"(N={len(result['pgo_poses'])} windows)")

        if not args.no_plot:
            png_path = seq_out / "trajectory.png"
            plot_xy(result, png_path, seq)
            print(f"[inference] {seq}: saved {png_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
