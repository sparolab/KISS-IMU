import os
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as Data

import pypose as pp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.seq_dataset import SeqDataset
from data.collate import collate_fn

from training.losses import get_losses, get_valid_losses
from training.integrator import IMUIntegrator
from training.monitoring import TrainingMonitor, log_training_step, log_training_epoch, log_pose_metrics
from training.covweight import CovWeightController

from utils.arguments import get_args
from utils.overlap_score import calc_symmetric_overlap, batch_pose_aligned_overlap

from models.imu_net import IMUNet
from models.lo_module import LOModule
from models.gmm import GmmModule
from models.pvgo import optimize

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR

args = get_args()

import math
from collections import defaultdict

class FreqGate:
    def __init__(self, K:int, device="cpu",
                 alpha:float=0.7,
                 p_min:float=0.15,
                 p_max:float=1.0,
                 prior:float=10.0,
                 ema_beta:float=0.99
                 ):
        self.K = K
        self.device = torch.device(device)
        self.alpha = alpha
        self.p_min = p_min
        self.p_max = p_max
        self.prior = prior
        self.beta = ema_beta

        self.rate = torch.zeros(K, dtype=torch.float32, device=self.device)
        self._warm = 1e-6

    @torch.no_grad()
    def keep_mask(self, comp_ids: torch.Tensor):
        if comp_ids.numel() == 0:
            z = torch.ones(0, device=self.device)
            return z, z

        freq = (self.rate + self.prior) / ((self.rate.sum() + self.prior * self.K) + self._warm)
        target = 1.0 / float(self.K)

        with torch.no_grad():
            p_keep_per_comp = torch.clamp((target / (freq + self._warm)) ** self.alpha,
                                          min=self.p_min, max=self.p_max)

        comp_ids_dev = comp_ids.detach().to(self.device, non_blocking=True).long()
        p_keep_per_sample = p_keep_per_comp[comp_ids_dev]

        gate = (torch.rand_like(p_keep_per_sample) < p_keep_per_sample).float()

        if gate.sum() < 1:
            rare_sample_idx = torch.argmin(freq[comp_ids_dev])
            gate[rare_sample_idx] = 1.0

        return gate, p_keep_per_sample

    @torch.no_grad()
    def update(self, comp_ids:torch.Tensor, used_mask:torch.Tensor):
        if comp_ids.numel() == 0:
            return
        K = self.K
        comp_ids = comp_ids.to('cpu', non_blocking=True).long()
        used_mask = used_mask.to('cpu', non_blocking=True).float()

        step_hist = torch.zeros(K, dtype=torch.float32)
        if used_mask.sum() > 0:
            binc = torch.bincount(comp_ids, weights=used_mask, minlength=K).float()
            step_hist[:len(binc)] = binc
            step_hist = step_hist / (used_mask.sum() + 1e-6)

        step_hist = step_hist.to(self.device)
        self.rate = self.beta * self.rate + (1.0 - self.beta) * step_hist


def train(lo_model, network, loader, optimizer, integrator, data_seq, epoch_i, gravity, global_step,
          gmm=None, gmm_reduce="soft-mode", conv_mgr=None):
    network.train()
    total_train_loss = 0

    batch_size = args.batch_size
    icp_count = 0
    pgo_count = 0
    if 'gt_poses_list' not in globals():
        global gt_poses_list
        gt_poses_list = []
        
    max_iterations = int(len(loader) * args.train_ratio)
    for i, sample in enumerate(tqdm.tqdm(loader, total=max_iterations)):
        if i >= max_iterations:
            break
        corr_data = network(sample)
        if len(label_poses_list) > 0:
            if isinstance(label_poses_list[-1], torch.Tensor):
                pose_vec = label_poses_list[-1][0] if label_poses_list[-1].ndim == 2 else label_poses_list[-1]
                init_pos = pose_vec[:3].clone().detach().to(args.device).float()
                init_rot = pose_vec[3:].clone().detach().to(args.device).float()
            else:
                pose_vec = label_poses_list[-1][0] if len(label_poses_list[-1].shape) == 2 else label_poses_list[-1]
                init_pos = torch.from_numpy(pose_vec[:3]).to(args.device).float()
                init_rot = torch.from_numpy(pose_vec[3:]).to(args.device).float()
            init_vel = label_vels_list[-1].to(args.device)
        else:
            init_pos = torch.from_numpy(init_state['pos']).to(args.device).float()
            init_rot = torch.from_numpy(init_state['rot']).to(args.device).float()
            init_vel = torch.from_numpy(init_state['vel']).to(args.device).float()

        init_rot = init_rot / torch.linalg.norm(init_rot)

        init_state = { 'rot': pp.SO3(init_rot),
                       'vel': init_vel,
                       'pos': init_pos,
                       'cov': None}

        imu_states = integrator.integrate(init=init_state,
                                          dts=corr_data['dts'], accels=corr_data['accels_corr'], gyros=corr_data['gyros_corr'],
                                          cov_accels=corr_data['acc_cov'], cov_gyros=corr_data['gyr_cov'],
                                          motion_mode=False)

        init_state = { 'rot': pp.SO3(init_rot),
                       'vel': torch.zeros((1, 3), dtype=torch.float32).to(args.device),
                       'pos': torch.zeros((1, 3), dtype=torch.float32).to(args.device),
                       'cov': None}
        imu_dstates = integrator.integrate(init=init_state,
                                           dts=corr_data['dts'], accels=corr_data['accels_corr'], gyros=corr_data['gyros_corr'],
                                           cov_accels=corr_data['acc_cov'], cov_gyros=corr_data['gyr_cov'],
                                           motion_mode=True)

        imu_nodes   = pp.SE3(torch.cat([imu_states['pos'], imu_states['rot'].tensor()], dim=-1)).to(args.device)
        imu_vels    = imu_states['vel'].to(args.device)
        imu_motions = pp.SE3(torch.cat([imu_dstates['pos'], imu_dstates['rot'].tensor()], dim=-1)).to(args.device)
        imu_dvels   = imu_dstates['vel'].to(args.device)
        imu_covs    = imu_states['cov'].to(args.device)
        imu_dcovs   = imu_dstates['cov'].detach().to(args.device)

        imu_dts = torch.stack([d.sum() for d in corr_data['dts']]).unsqueeze(-1).to(args.device)

        if getattr(args, 'use_gt', False):
            prev_anchor = label_poses_list[-1].to(args.device).reshape(1, 7).float()
            gt_seq = sample['gt_pose1'].to(args.device).float()
            label_poses = pp.SE3(torch.cat([prev_anchor, gt_seq], dim=0))
            label_motions = (label_poses[:-1].Inv() @ label_poses[1:]).tensor()
        else:
            icp_poses, icp_motions, icp_overlap_scores = lo_model(sample, pp.SE3(label_poses_list[-1]))

            pgo_poses, pgo_vels = optimize(nodes=imu_nodes, vels=imu_vels,
                                           icp_factors=icp_motions,
                                           imu_drots=imu_dstates['rot'], imu_dvels=imu_dstates['vel'], imu_dtrans=imu_dstates['pos'],
                                           imu_dts=imu_dts,
                                           weights=args.lm_weight,
                                           gravity=gravity,
                                           icp_weights=None,
                                           imu_weights=None,
                                           device=args.device)

            pgo_motions = pgo_poses[:-1].Inv() @ pgo_poses[1:]
            pgo_dvels   = pgo_vels[1:] - pgo_vels[:-1]

            pgo_overlap_scores = batch_pose_aligned_overlap(lo_model.scans0_np, lo_model.scans1_np, pgo_motions)

            better_idx = torch.from_numpy(np.argmax(np.stack([icp_overlap_scores, pgo_overlap_scores]), axis=0)).to(icp_poses.device)

            icp_count += (better_idx == 0).sum().item()
            pgo_count += (better_idx == 1).sum().item()

            poses_stack = torch.stack([icp_poses[1:], pgo_poses[1:]], dim=0)
            motions_stack = torch.stack([icp_motions.tensor(), pgo_motions.tensor()], dim=0)

            batch_indices = torch.arange(len(better_idx), device=better_idx.device)
            label_poses = poses_stack[better_idx, batch_indices]
            label_poses = torch.cat([icp_poses[:1], label_poses], dim=0)
            label_motions = motions_stack[better_idx, batch_indices]

        t_all = label_poses.translation()
        dt_world = t_all[1:].contiguous() - t_all[:-1].contiguous()
        label_dts   = torch.stack([d.sum() for d in corr_data['dts']]).unsqueeze(-1)
        label_vels = dt_world / label_dts
        label_vels = torch.cat([init_vel.unsqueeze(0).detach(), label_vels.detach()], dim=0)
        label_dvels = label_vels[1:] - label_vels[:-1]
        label_motions = pp.SE3(label_motions)

        (rot_loss_v, vel_loss_v, pos_loss_v,
         rot_cov_v, vel_cov_v, pos_cov_v) = get_losses(
            imu_nodes, imu_vels, imu_motions, imu_dvels, imu_covs,
            label_poses, label_vels, label_motions, label_dvels,
            args, reduction="none"
        )

        batch_size = sample['accels'].shape[0]

        if rot_loss_v.dim() == 0:
            rot_loss_v = rot_loss_v.unsqueeze(0)
        if vel_loss_v.dim() == 0:
            vel_loss_v = vel_loss_v.unsqueeze(0)
        if pos_loss_v.dim() == 0:
            pos_loss_v = pos_loss_v.unsqueeze(0)
        if rot_cov_v.dim() == 0:
            rot_cov_v = rot_cov_v.unsqueeze(0)
        if vel_cov_v.dim() == 0:
            vel_cov_v = vel_cov_v.unsqueeze(0)
        if pos_cov_v.dim() == 0:
            pos_cov_v = pos_cov_v.unsqueeze(0)

        if rot_loss_v.shape[0] == 1 and batch_size > 1:
            rot_loss_v = rot_loss_v.expand(batch_size)
            vel_loss_v = vel_loss_v.expand(batch_size)
            pos_loss_v = pos_loss_v.expand(batch_size)
            rot_cov_v = rot_cov_v.expand(batch_size)
            vel_cov_v = vel_cov_v.expand(batch_size)
            pos_cov_v = pos_cov_v.expand(batch_size)

        if global_step % 10 == 0 or global_step == 1:
            for name, tensor in [("rot_loss_v", rot_loss_v), ("vel_loss_v", vel_loss_v),
                               ("pos_loss_v", pos_loss_v), ("rot_cov_v", rot_cov_v),
                               ("vel_cov_v", vel_cov_v), ("pos_cov_v", pos_cov_v)]:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"[WARN] {name} contains NaN/Inf!")
                    print(f"  NaN count: {torch.isnan(tensor).sum()}")
                    print(f"  Inf count: {torch.isinf(tensor).sum()}")
                    tensor.data = torch.where(torch.isnan(tensor) | torch.isinf(tensor),
                                           torch.zeros_like(tensor), tensor)

        if gmm is not None:
            imu_ts_b  = sample['imu_ts']
            accels_b  = sample['accels']
            gyros_b   = sample['gyros']
            valid_len = sample['valid_length']
            B = imu_ts_b.shape[0]
            comp_ids = []
            for b in range(B):
                L = int(valid_len[b])
                comp_id = gmm.predict_window(
                    imu_ts=imu_ts_b[b, :L],
                    accels=accels_b[b, :L],
                    gyros=gyros_b[b, :L],
                    gravity=(gravity.detach().cpu().numpy() if torch.is_tensor(gravity) else gravity),
                    reduce=gmm_reduce,
                    return_proba=False
                )
                comp_ids.append(int(comp_id))
        else:
            comp_ids = [0] * sample['imu_ts'].shape[0]

        K = gmm.gmm.n_components if gmm is not None else 1
        comp_ids_t = torch.as_tensor(comp_ids, device=rot_loss_v.device, dtype=torch.long)
        invalid_mask = (comp_ids_t < 0) | (comp_ids_t >= K)
        if invalid_mask.any():
            if (global_step % 10 == 0) or (global_step == 1):
                monitor.writer.add_scalar('train_gmm/invalid_comp_ids', invalid_mask.sum().item(), global_step)
            comp_ids_t = comp_ids_t.clamp_(0, K-1)
            print(f"\033[91m[WARN] Invalid comp_ids detected; clamped to range 0..{K-1}\033[0m")
        comp_ids_list = [int(x) for x in comp_ids_t.detach().cpu().tolist()]

        if gmm is not None:
            base_w = gmm.torch_weights_from_labels(
                np.array(comp_ids_list), device=rot_loss_v.device, dtype=rot_loss_v.dtype
            )
        else:
            base_w = torch.ones_like(rot_loss_v)

        if conv_mgr is not None:
            active_mask_np = np.array([not conv_mgr.is_converged(k) for k in comp_ids_list], dtype=np.float32)
            active_mask = torch.from_numpy(active_mask_np).to(rot_loss_v.device, rot_loss_v.dtype)
            if conv_mgr.all_converged(comp_ids_list):
                print(f"\033[92m[INFO] batch skipped (all components converged: {sorted(set(comp_ids_list))})\033[0m")
                continue
        else:
            active_mask = torch.ones_like(rot_loss_v)

        if hasattr(train, "_freq_gate") and gmm is not None:
            keep_mask, p_keep = train._freq_gate.keep_mask(comp_ids_t)
            keep_mask = keep_mask.to(dtype=rot_loss_v.dtype, device=rot_loss_v.device)

            active_bool = (active_mask > 0).float()
            eff_candidate = active_bool * keep_mask
            if eff_candidate.sum() < 1:
                active_idx = torch.nonzero(active_bool > 0, as_tuple=False).squeeze(1)
                if active_idx.numel() > 0:
                    _, rel_idx = torch.max(p_keep[active_idx], dim=0)
                    pick = active_idx[rel_idx]
                    keep_mask[pick] = 1.0

            if global_step % 10 == 0 or global_step == 1:
                monitor.writer.add_scalar('train_gmm/keep_prob_mean', p_keep.mean().item(), global_step)
                monitor.writer.add_scalar('train_gmm/keep_rate_batch', keep_mask.mean().item(), global_step)
        else:
            keep_mask = torch.ones_like(rot_loss_v)

        effective_w = base_w * active_mask * keep_mask

        per_sample_total = (
            args.rot_w * rot_loss_v + args.cov_r_w * rot_cov_v +
            args.vel_w * vel_loss_v + args.cov_v_w * vel_cov_v +
            args.pos_w * pos_loss_v + args.cov_t_w * pos_cov_v
        )

        if global_step % 10 == 0 or global_step == 1:
            if torch.isnan(per_sample_total).any() or torch.isinf(per_sample_total).any():
                print(f"[WARN] NaN/Inf detected in per_sample_total!")
                print(f"  NaN count: {torch.isnan(per_sample_total).sum()}")
                print(f"  Inf count: {torch.isinf(per_sample_total).sum()}")
                per_sample_total = torch.where(torch.isnan(per_sample_total) | torch.isinf(per_sample_total),
                                            torch.zeros_like(per_sample_total), per_sample_total)

        main_loss = (effective_w * per_sample_total).mean()

        if hasattr(train, "_freq_gate") and gmm is not None:
            with torch.no_grad():
                freq = (train._freq_gate.rate + train._freq_gate.prior) / \
                       (train._freq_gate.rate.sum() + train._freq_gate.prior * train._freq_gate.K + 1e-6)
                target = 1.0 / float(train._freq_gate.K)
                rarity = torch.clamp((target / (freq + 1e-6)) ** train._freq_gate.alpha,
                                     min=train._freq_gate.p_min, max=train._freq_gate.p_max)
                r_per = rarity[comp_ids_t]
                topk_ratio = getattr(args, "rare_topk_ratio", 0.25)
                k = max(1, int(topk_ratio * per_sample_total.numel()))

                k = min(k, r_per.numel())
                if k > 0:
                    idx = torch.topk(r_per, k=k, largest=True).indices
                    rare_mask = torch.zeros_like(per_sample_total)
                    valid_idx = idx[idx < rare_mask.numel()]
                    if valid_idx.numel() > 0:
                        rare_mask[valid_idx] = 1.0
                    else:
                        rare_mask = torch.zeros_like(per_sample_total)
                else:
                    rare_mask = torch.zeros_like(per_sample_total)
        else:
            rare_mask = torch.zeros_like(per_sample_total)

        rare_loss  = (rare_mask * per_sample_total).sum() / (rare_mask.sum() + 1e-6)
        rare_boost = getattr(args, "rare_boost", 0.3)
        loss = main_loss + rare_boost * rare_loss

        optimizer.zero_grad()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ERROR] Loss is NaN/Inf: {loss.item()}")
            print(f"[ERROR] Skipping this batch due to invalid loss")
            continue

        if not loss.requires_grad:
            print(f"[WARN] Loss doesn't require grad: {loss.requires_grad}")
            continue

        if loss.item() > 1e6:
            print(f"[WARN] Loss too large: {loss.item()}, clipping to 1e6")
            loss = torch.clamp(loss, max=1e6)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()

        if hasattr(train, "_freq_gate") and gmm is not None:
            usage_w = torch.logical_or(effective_w > 0, rare_mask > 0).float()
            train._freq_gate.update(comp_ids_t, usage_w)

        rot_loss_log = args.rot_w * rot_loss_v.mean().item()
        vel_loss_log = args.vel_w * vel_loss_v.mean().item()
        pos_loss_log = args.pos_w * pos_loss_v.mean().item()
        rot_cov_log  = args.cov_r_w * rot_cov_v.mean().item()
        vel_cov_log  = args.cov_v_w * vel_cov_v.mean().item()
        pos_cov_log  = args.cov_t_w * pos_cov_v.mean().item()

        imu_poses_list.extend(imu_nodes.tensor()[1:].detach())
        imu_motions_list.extend(imu_motions.detach())
        imu_vels_list.extend(imu_vels[1:].detach())
        imu_covs_list.extend(imu_covs[1:].detach())

        if not getattr(args, 'use_gt', False):
            icp_poses_list.extend(icp_poses.tensor()[1:].detach())
            icp_motions_list.extend(icp_motions.detach())
            icp_overlap_score_list.extend(icp_overlap_scores)

            pgo_poses_list.extend(pgo_poses.tensor()[1:].detach())
            pgo_motions_list.extend(pgo_motions.detach())
            pgo_vels_list.extend(pgo_vels[1:].detach())
            pgo_overlap_score_list.extend(pgo_overlap_scores)

        label_poses_list.extend(label_poses.tensor()[1:].clone().detach())
        label_motions_list.extend(label_motions.clone().detach())
        label_vels_list.extend(label_vels[1:].clone().detach())
        gt_poses_list.extend(sample['gt_pose1'][1:].clone().detach())

        total_train_loss += loss.item()

        global_step += 1
        if global_step % 10 == 0 or global_step == 1:
            monitor.log_losses({
                'loss': loss.item(),
                'main_loss': main_loss.item(),
                'rare_loss': rare_loss.item(),
                'rot_loss': rot_loss_log,
                'vel_loss': vel_loss_log,
                'pos_loss': pos_loss_log,
                'rot_cov_loss': rot_cov_log,
                'vel_cov_loss': vel_cov_log,
                'pos_cov_loss': pos_cov_log
            }, step=global_step, prefix="train/")
            log_pose_metrics(monitor, global_step,
                pose_losses={'total': main_loss.item(), 'rot': rot_loss_log, 'vel': vel_loss_log, 'pos': pos_loss_log},
                cov_losses={'total': rot_cov_log + vel_cov_log + pos_cov_log,
                            'rot': rot_cov_log, 'vel': vel_cov_log, 'pos': pos_cov_log},
                rel_errors={'rot': 0.0, 'trans': 0.0},
                prefix="train_")

            if len(pgo_poses_list) > 1 and len(gt_poses_list) > 1:
                pgo_poses_np = np.stack([t.detach().cpu().numpy() for t in pgo_poses_list])
                gt_poses_np  = np.stack([t.detach().cpu().numpy() for t in gt_poses_list])
                poses_dict = {'gt': gt_poses_np, 'pgo': pgo_poses_np}
                if len(icp_poses_list) > 1:
                    icp_poses_np = np.stack([t.detach().cpu().numpy() for t in icp_poses_list])
                    poses_dict['icp'] = icp_poses_np
                if len(label_poses_list) > 1:
                    label_poses_np = np.stack([t.detach().cpu().numpy() for t in label_poses_list])
                    poses_dict['label'] = label_poses_np

                monitor.log_multiple_pose_comparison(
                    poses_dict, global_step, f"train_all_trajectories_{data_seq}", epoch=epoch_i, train_valid="train"
                )

            monitor.log_learning_rate(optimizer, global_step)
            monitor.log_gradients(network, global_step, prefix="train_")

            if hasattr(train, "_cov_ctrl"):
                try:
                    comp_stats = train._cov_ctrl.get_component_stats()
                    if comp_stats['component_usage']:
                        component_ids = np.array(sorted(comp_stats['component_usage'].keys()))
                        usage_counts  = np.array([comp_stats['component_usage'][cid] for cid in component_ids], dtype=int)
                        fig, ax = plt.subplots(figsize=(8,3))
                        ax.bar(component_ids, usage_counts)
                        ax.set_xlabel('Component ID'); ax.set_ylabel('# used in optimize')
                        ax.set_xticks(component_ids); ax.grid(True, axis='y', alpha=0.3)
                        monitor.writer.add_figure('train_gmm/usage_bar', fig, global_step)
                        plt.close(fig)
                except Exception as e:
                    print(f"Warning: Error logging GMM component usage: {e}")

            monitor.writer.flush()
            if hasattr(monitor.writer, '_file_writer'):
                monitor.writer._file_writer.flush()
    if icp_count + pgo_count > 0:
        labels = ['ICP', 'PGO']
        counts = [icp_count, pgo_count]

        monitor.writer.add_histogram(
            f'pose_selection/{data_seq}_epoch_{epoch_i}',
            torch.tensor([0] * icp_count + [1] * pgo_count),
            bins=2,
            global_step=epoch_i
        )

        total_count = icp_count + pgo_count
        monitor.writer.add_scalar(f'pose_selection/{data_seq}_icp_ratio', icp_count / total_count, epoch_i)
        monitor.writer.add_scalar(f'pose_selection/{data_seq}_pgo_ratio', pgo_count / total_count, epoch_i)
        monitor.writer.add_scalar(f'pose_selection/{data_seq}_icp_count', icp_count, epoch_i)
        monitor.writer.add_scalar(f'pose_selection/{data_seq}_pgo_count', pgo_count, epoch_i)
        
        print(f"  * {data_seq} Epoch {epoch_i}: ICP={icp_count}, PGO={pgo_count} (ICP ratio: {icp_count/total_count:.3f})")

    return total_train_loss / max_iterations, global_step


def as_xyz1d(t: torch.Tensor) -> np.ndarray:
    a = t.reshape(-1).detach().cpu().numpy()
    return a[:3]

def validate(data_loader, network, integrator, data_seq, epoch_i):
    device = args.device

    sum_ep_rpe_trans = 0.0
    sum_ep_rpe_rot_deg = 0.0
    sum_last_ape = 0.0
    n_windows = 0

    pred_last_list = []
    gt_last_list = []

    for sample in tqdm.tqdm(data_loader):
        init_pos = sample['gt_pose0'][0][:3].clone().to(device).float()
        init_rot = sample['gt_pose0'][0][3:].clone().to(device).float()
        init_vel = sample['gt_velocity'][0].clone().to(device).float()
        init_rot = init_rot / (torch.linalg.norm(init_rot) + 1e-12)
        init_state = {'rot': pp.SO3(init_rot), 'vel': init_vel, 'pos': init_pos, 'cov': None}

        corr_data = network(sample)
        out_state = integrator.integrate(init=init_state,
                                         dts=corr_data['dts'], accels=corr_data['accels_corr'], gyros=corr_data['gyros_corr'],
                                         cov_accels=corr_data['acc_cov'], cov_gyros=corr_data['gyr_cov'],
                                         motion_mode=False)
 
        pred_last_se3 = pp.SE3(torch.cat([out_state['pos'], out_state['rot'].tensor()], dim=-1)).to(device)

        gt0 = sample['gt_pose0'][0].to(device).float()
        gt_seq = sample['gt_pose1'].to(device).float()
        if gt_seq.ndim == 3:
            gt_seq = gt_seq[0]
        gt_last = gt_seq[-1]

        q0 = gt0[3:]
        qT = gt_last[3:]
        GT0 = pp.SE3(torch.cat([gt0[:3], q0], dim=-1)).to(device)
        GTT = pp.SE3(torch.cat([gt_last[:3], qT], dim=-1)).to(device)

        dP = GT0.Inv() * pred_last_se3
        dQ = GT0.Inv() * GTT
        E  = dP.Inv() * dQ

        trans_err   = E.translation().norm().item()
        rot_err_deg = (E.rotation().Log().norm().item() * 180.0 / np.pi)
        ape_last    = (pred_last_se3.translation() - GTT.translation()).norm().item()

        sum_ep_rpe_trans   += trans_err
        sum_ep_rpe_rot_deg += rot_err_deg
        sum_last_ape       += ape_last
        n_windows          += 1

        pred_last_list.append(as_xyz1d(pred_last_se3.translation()))
        gt_last_list.append(as_xyz1d(GTT.translation()))

    if n_windows > 0:
        mean_ep_rpe_trans = sum_ep_rpe_trans / n_windows
        mean_ep_rpe_rot   = sum_ep_rpe_rot_deg / n_windows
        mean_last_ape     = sum_last_ape / n_windows
    else:
        mean_ep_rpe_trans = mean_ep_rpe_rot = mean_last_ape = float('inf')

    print("====================================================")
    print(f"[Summary] RTE in {data_seq} at epoch {epoch_i}: {mean_ep_rpe_trans:.3f} m, ")
    print(f"[Summary] RRE in {data_seq} at epoch {epoch_i}: {mean_ep_rpe_rot:.3f} deg")
    print("====================================================")
    return mean_ep_rpe_trans+mean_ep_rpe_rot


def init_list(dataset):
    global init_state, dataiter
    init_state = dataset.init
    init_pose = np.concatenate((init_state['pos'], init_state['rot']))

    global imu_poses_list, imu_motions_list, imu_vels_list, imu_covs_list
    imu_poses_list = [torch.from_numpy(init_pose).reshape(7)]
    imu_motions_list = []
    imu_vels_list = [init_state['vel']]
    imu_covs_list = []

    global icp_poses_list, icp_motions_list, icp_vels_list, icp_overlap_score_list
    icp_poses_list = [torch.from_numpy(init_pose).reshape(7)]
    icp_motions_list = []
    icp_vels_list = [init_state['vel']]
    icp_overlap_score_list = []

    global pgo_poses_list, pgo_motions_list, pgo_vels_list, pgo_overlap_score_list
    pgo_poses_list = [torch.from_numpy(init_pose).reshape(7)]
    pgo_motions_list = []
    pgo_vels_list = [init_state['vel']]
    pgo_overlap_score_list = []

    global label_poses_list, label_motions_list, label_vels_list
    label_poses_list = [torch.from_numpy(init_pose).reshape(7)]
    label_motions_list = []
    label_vels_list = [init_state['vel']]

    global valid_poses_list, valid_vels_list
    valid_poses_list = [torch.from_numpy(init_pose).reshape(7)]
    valid_vels_list = [init_state['vel']]

    global gt_poses_list
    gt_poses_list = []


def plot_pose_trajs(icp_poses, pgo_poses, label_poses, gt_poses, filename, data_seq, epoch_i, train_set=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        fig, ax = plt.subplots()
        if gt_poses is not None and len(gt_poses) > 0:
            gt_np = np.asarray(gt_poses)
            ax.plot(gt_np[:,0], gt_np[:,1], 'k--', label='Ground Truth', linewidth=3, zorder=1)
        if icp_poses is not None and len(icp_poses) > 0:
            icp_np = np.asarray(icp_poses)
            ax.plot(icp_np[:,0], icp_np[:,1], 'r-',  label=f'ICP({args.lo_model}) Pose', linewidth=2.5, zorder=2)
        if pgo_poses is not None and len(pgo_poses) > 0:
            pgo_np = np.asarray(pgo_poses)
            ax.plot(pgo_np[:,0], pgo_np[:,1], 'b-',  label='PGO Pose', linewidth=2.5, zorder=3)
        if label_poses is not None and len(label_poses) > 0:
            label_np = np.asarray(label_poses)
            ax.plot(label_np[:,0], label_np[:,1], 'g--', label='Label Pose', linewidth=2.5, zorder=4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if train_set:
            ax.set_title(f'Train Pose Trajectories in {data_seq} epoch {epoch_i:04d}')
        else:
            ax.set_title(f'Valid Pose Trajectories in {data_seq} epoch {epoch_i:04d}')
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"Warning: Error saving trajectory plot to {filename}: {e}")
        try:
            plt.close('all')
        except:
            pass


def save_component_usage_histogram(cov_ctrl, result_dir):
    import matplotlib.pyplot as plt

    try:
        stats = cov_ctrl.get_component_stats()
        component_usage = stats['component_usage']
        total_updates = stats['total_updates']
        unique_components = stats['unique_components']

        if not component_usage:
            print("No component usage data to save")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        component_ids = list(component_usage.keys())
        usage_counts = list(component_usage.values())

        bright_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                         '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                         '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2']
        colors = bright_colors[:len(component_ids)]
        ax1.bar(component_ids, usage_counts, alpha=0.7, color=colors, edgecolor='black')
        ax1.set_xlabel('Component ID')
        ax1.set_ylabel('Usage Count')
        ax1.set_title('Component Usage Histogram')
        ax1.grid(True, alpha=0.3)

        for i, v in enumerate(usage_counts):
            ax1.text(component_ids[i], v + max(usage_counts) * 0.01, str(v),
                     ha='center', va='bottom', fontweight='bold')

        if total_updates > 0:
            percentages = [count/total_updates * 100 for count in usage_counts]
            ax2.pie(usage_counts, labels=[f'Comp {cid}\n({p:.1f}%)' for cid, p in zip(component_ids, percentages)],
                    autopct='%1.1f%%', startangle=90, colors=colors)
            ax2.set_title('Component Usage Distribution')

        fig.suptitle(f'Component Usage Statistics\nTotal Updates: {total_updates}, Unique Components: {unique_components}',
                     fontsize=14)
        plt.tight_layout()

        histogram_path = os.path.join(result_dir, "component_usage_histogram.png")
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        plt.close()

        stats_path = os.path.join(result_dir, "component_usage_stats.txt")
        with open(stats_path, 'w') as f:
            f.write("Component Usage Statistics\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Updates: {total_updates}\n")
            f.write(f"Unique Components: {unique_components}\n")
            f.write("\nComponent-wise Usage:\n")
            f.write("-" * 30 + "\n")
            for comp_id, count in sorted(component_usage.items()):
                percentage = (count / total_updates * 100) if total_updates > 0 else 0
                f.write(f"Component {comp_id}: {count} times ({percentage:.2f}%)\n")

        print(f"Component usage histogram saved to: {histogram_path}")
        print(f"Component usage statistics saved to: {stats_path}")

    except Exception as e:
        print(f"Warning: Error saving component usage histogram: {e}")
        try:
            plt.close('all')
        except:
            pass


def save_gmm_initial_histogram(gmm, result_dir):
    import matplotlib.pyplot as plt

    try:
        component_weights = gmm.gmm.weights_
        component_means   = gmm.gmm.means_
        n_components = len(component_weights)

        if n_components == 0:
            print("No GMM components to visualize")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        component_ids = list(range(n_components))
        bright_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                         '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                         '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2']
        colors = bright_colors[:n_components]

        bars = ax1.bar(component_ids, component_weights, alpha=0.7, color=colors, edgecolor='black')
        ax1.set_xlabel('Component ID')
        ax1.set_ylabel('Weight')
        ax1.set_title('GMM Component Weights')
        ax1.grid(True, alpha=0.3)
        for i, w in enumerate(component_weights):
            ax1.text(i, w + max(component_weights) * 0.01, f'{w:.3f}',
                     ha='center', va='bottom', fontweight='bold', fontsize=8)

        ax2.pie(component_weights, labels=[f'Comp {i}\n({w:.3f})' for i, w in enumerate(component_weights)],
                autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('GMM Component Weight Distribution')

        if component_means.shape[1] >= 3:
            ax3.scatter(component_means[:, 0], component_means[:, 1],
                        s=component_weights * 1000, alpha=0.7, c=component_ids, cmap='viridis')
            ax3.set_xlabel('Dimension 1')
            ax3.set_ylabel('Dimension 2')
            ax3.set_title('GMM Component Means (Dim 1 vs Dim 2)')
            ax3.grid(True, alpha=0.3)
            for i, (x, y) in enumerate(zip(component_means[:, 0], component_means[:, 1])):
                ax3.annotate(f'C{i}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        else:
            available_dims = min(component_means.shape[1], 2)
            if available_dims >= 2:
                ax3.scatter(component_means[:, 0], component_means[:, 1],
                            s=component_weights * 1000, alpha=0.7, c=component_ids, cmap='viridis')
                ax3.set_xlabel('Dimension 1')
                ax3.set_ylabel('Dimension 2')
                ax3.set_title(f'GMM Component Means (Dim 1 vs Dim 2)')
            else:
                ax3.scatter(component_means[:, 0], [0] * n_components,
                            s=component_weights * 1000, alpha=0.7, c=component_ids, cmap='viridis')
                ax3.set_xlabel('Dimension 1')
                ax3.set_ylabel('Fixed Y')
                ax3.set_title(f'GMM Component Means (1D)')
            ax3.grid(True, alpha=0.3)
            for i, x in enumerate(component_means[:, 0]):
                y_pos = 0 if available_dims < 2 else component_means[i, 1]
                ax3.annotate(f'C{i}', (x, y_pos), xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax4.axis('off')
        stats_text = f"GMM Component Statistics\n\n"
        stats_text += f"Total Components: {n_components}\n"
        stats_text += f"Total Weight: {component_weights.sum():.3f}\n"
        stats_text += f"Min Weight: {component_weights.min():.3f}\n"
        stats_text += f"Max Weight: {component_weights.max():.3f}\n"
        stats_text += f"Weight Std: {component_weights.std():.3f}\n\n"
        stats_text += "Component Details:\n"
        for i in range(n_components):
            stats_text += f"C{i}: w={component_weights[i]:.3f}, "
            if component_means.shape[1] >= 3:
                stats_text += f"μ=({component_means[i, 0]:.2f}, {component_means[i, 1]:.2f}, {component_means[i, 2]:.2f})\n"
            else:
                stats_text += f"μ=({component_means[i, 0]:.2f}"
                if component_means.shape[1] >= 2:
                    stats_text += f", {component_means[i, 1]:.2f}"
                stats_text += ")\n"

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace')

        fig.suptitle(f'Initial GMM Component Distribution\n{n_components} Components', fontsize=16)
        plt.tight_layout()

        histogram_path = os.path.join(result_dir, "gmm_initial_distribution.png")
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        plt.close()

        stats_path = os.path.join(result_dir, "gmm_initial_stats.txt")
        with open(stats_path, 'w') as f:
            f.write("Initial GMM Component Statistics\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Components: {n_components}\n")
            f.write(f"Total Weight: {component_weights.sum():.3f}\n")
            f.write(f"Min Weight: {component_weights.min():.3f}\n")
            f.write(f"Max Weight: {component_weights.max():.3f}\n")
            f.write(f"Weight Std: {component_weights.std():.3f}\n")
            f.write(f"Weight Mean: {component_weights.mean():.3f}\n\n")
            f.write("Component-wise Details:\n")
            f.write("-" * 40 + "\n")
            for i in range(n_components):
                f.write(f"Component {i}:\n")
                f.write(f"  Weight: {component_weights[i]:.6f}\n")
                f.write(f"  Mean: {component_means[i]}\n")
                if hasattr(gmm.gmm, 'covariances_'):
                    f.write(f"  Covariance Shape: {gmm.gmm.covariances_[i].shape}\n")
                f.write("\n")

        print(f"Initial GMM distribution histogram saved to: {histogram_path}")
        print(f"Initial GMM statistics saved to: {stats_path}")

    except Exception as e:
        print(f"Warning: Error saving initial GMM histogram: {e}")
        try:
            plt.close('all')
        except:
            pass


def save_ckpt(network, optimizer, epoch_i, data_seq=None, save_best = False, train_set=False, valid_set=False):
    if train_set:
        base_dir = os.path.join(args.result_dir, "train", data_seq)
    elif valid_set:
        base_dir = os.path.join(args.result_dir, "valid", data_seq)
    else:
        base_dir = os.path.join(args.result_dir)

    ckpt_dir = os.path.join(base_dir, "ckpt")
    if save_best:
        save_path = os.path.join(base_dir, 'best_model.ckpt')
    else:
        save_path = os.path.join(ckpt_dir, f"{epoch_i:04d}.ckpt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if train_set:
        def _stack_tensors(lst):
            return np.stack([t.detach().cpu() for t in lst]) if len(lst) > 0 else None
        def _stack_arrays(lst):
            return np.stack(lst) if len(lst) > 0 else None

        torch.save({
            'epoch': epoch_i,
            'network': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'imu_poses_list': _stack_tensors(imu_poses_list),
            'imu_motions_list': _stack_tensors(imu_motions_list),
            'imu_vels_list': _stack_tensors(imu_vels_list),
            'imu_covs_list': _stack_tensors(imu_covs_list),
            'icp_poses_list': _stack_tensors(icp_poses_list),
            'icp_motions_list': _stack_tensors(icp_motions_list),
            'icp_vels_list': _stack_tensors(icp_vels_list),
            'icp_overlap_score_list': _stack_arrays(icp_overlap_score_list),
            'pgo_poses_list': _stack_tensors(pgo_poses_list),
            'pgo_motions_list': _stack_tensors(pgo_motions_list),
            'pgo_vels_list': _stack_tensors(pgo_vels_list),
            'pgo_overlap_score_list': _stack_arrays(pgo_overlap_score_list),
            'label_poses_list': _stack_tensors(label_poses_list),
            'label_motions_list': _stack_tensors(label_motions_list),
            'label_vels_list': _stack_tensors(label_vels_list),
        }, save_path)
    else:
        torch.save({
            'epoch': epoch_i,
            'network': network.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, save_path)
    return ckpt_dir


def create_gmm_data_packs(train_packs, train_ratio):
    gmm_packs = []

    for seq, dataset, loader in train_packs:
        total_samples = len(dataset)
        limited_samples = int(total_samples * train_ratio)

        limited_dataset = SeqDataset(
            data_root=dataset.data_root,
            data_seq=dataset.data_seq,
            data_type=dataset.data_type
        )

        limited_dataset.imu_ts = dataset.imu_ts[:limited_samples]
        limited_dataset.accels = dataset.accels[:limited_samples]
        limited_dataset.gyros = dataset.gyros[:limited_samples]
        limited_dataset.gravity = dataset.gravity

        dummy_loader = Data.DataLoader(
            dataset=limited_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            drop_last=False
        )

        gmm_packs.append((seq, limited_dataset, dummy_loader))

        print(f"  * GMM data for {seq}: {total_samples} -> {limited_samples} samples (ratio: {train_ratio:.2f})")

    return gmm_packs


if __name__ == "__main__":
    os.makedirs(args.result_dir, exist_ok=True)

    monitor = TrainingMonitor(
        log_dir=os.path.join(args.result_dir, "monitoring"),
        experiment_name=f"train_01_monitoring_{args.lo_model}",
        hyperparams={
            'scheduler': args.scheduler,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epoch,
            'lo_model': args.lo_model,
            'rot_w': args.rot_w,
            'vel_w': args.vel_w,
            'pos_w': args.pos_w,
            'cov_r_w': args.cov_r_w,
            'cov_v_w': args.cov_v_w,
            'cov_t_w': args.cov_t_w,
            'lm_weight': args.lm_weight
        }
    )

    network = IMUNet(prop_cov=args.prop_cov, device=args.device).to(args.device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    train_packs = []
    for seq in args.train_seqs:
        ds = SeqDataset(data_root=args.data_root, data_seq=seq, data_type=args.data_type)
        dl = Data.DataLoader(dataset=ds,
                             batch_size=args.batch_size,
                             num_workers=args.worker_num,
                             shuffle=False,
                             drop_last=True,
                             collate_fn=collate_fn)
        train_packs.append((seq, ds, dl))

    valid_packs = []
    for seq in args.valid_seqs:
        ds = SeqDataset(data_root=args.data_root, data_seq=seq, data_type=args.data_type)
        dl = Data.DataLoader(dataset=ds,
                             batch_size=args.batch_size,
                             num_workers=args.worker_num,
                             shuffle=False,
                             drop_last=True,
                             collate_fn=collate_fn)
        valid_packs.append((seq, ds, dl))

    use_gmm_weights = True
    gmm_reduce_mode = "soft-mode"
    gmm = None
    conv_mgr = None
    if use_gmm_weights:
        print(f"Creating limited GMM dataset with train_ratio: {args.train_ratio}")
        gmm_train_packs = create_gmm_data_packs(train_packs, args.train_ratio)

        gmm = GmmModule(gmm_train_packs, K=None, win_sec=0.2).fit()
        gmm_initial = gmm

        _ = gmm.compute_component_weights(
            source="train", method="effective", beta=0.999,
            normalize="mean1", clamp=(0.1, 10.0)
        )

        print("Saving initial GMM distribution...")
        save_gmm_initial_histogram(gmm, args.result_dir)

        gmm_save_path = os.path.join(args.result_dir, "gmm.joblib")
        gmm.save(gmm_save_path)
        print(f"GMM saved to: {gmm_save_path}")

        from training.convergence import CompConvergenceManager
        conv_mgr = CompConvergenceManager(
            K=gmm.gmm.n_components, ema_beta=0.9,
            improve_tol=1e-3, abs_tol=None, patience=5, min_seen=100
        )

        if not hasattr(train, "_freq_gate"):
            train._freq_gate = FreqGate(
                K=gmm.gmm.n_components,
                device=args.device,
                alpha=getattr(args, "freq_alpha", 0.7),
                p_min=getattr(args, "freq_pmin", 0.15),
                p_max=getattr(args, "freq_pmax", 1.0),
                prior=getattr(args, "freq_prior", 10.0),
                ema_beta=getattr(args, "freq_ema", 0.99),
            )

    train_global_step = 0
    val_global_step = 0

    if args.pretrained_model != 'None':
        print('=')
        network.load_state_dict(torch.load(args.pretrained_model)['network'])
        print(f"  * Loaded Pretrained Model: {args.pretrained_model} *   ")

    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            verbose=True,
            min_lr=args.scheduler_min_lr
        )
    elif args.scheduler == 'StepLR':
        scheduler = StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_factor
        )
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epoch,
            eta_min=args.scheduler_min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    best_valid_loss = float('inf')
    epoch_train_loss = 0
    epoch_valid_loss = 0
    for epoch_i in range(1, args.epoch+1):
        print(f"====== Epoch {epoch_i}/{args.epoch+1} ======")

        for train_seq, train_dataset, train_loader in train_packs:
            print(f"  * Training on sequence: {train_seq} *   ")

            init_list(train_dataset)

            if args.use_submap:
                lo_model = LOModule(lo_model=args.lo_model, T_I_L=train_dataset.T_I_G, R_I_L=train_dataset.R_I_L,
                                    init_state=icp_poses_list[-1], device_id=args.device, use_submap=True)
            else:
                lo_model = LOModule(lo_model=args.lo_model, T_I_L=train_dataset.T_I_G, R_I_L=train_dataset.R_I_L,
                                    init_state=icp_poses_list[-1], device_id=args.device, use_submap=False)
            integrator = IMUIntegrator(init_state=train_dataset.init, device=args.device)

            gravity = train_dataset.gravity
            train_loss, train_global_step = train(
                lo_model=lo_model, network=network, loader=train_loader, optimizer=optimizer,
                integrator=integrator, data_seq=train_seq, epoch_i=epoch_i, gravity=gravity,
                global_step=train_global_step,
                gmm=gmm, gmm_reduce=gmm_reduce_mode, conv_mgr=conv_mgr
            )
            epoch_train_loss += train_loss
            print(f"  * Train Loss: {train_loss} *   ")
            ckpt_dir = save_ckpt(network=network, optimizer=optimizer, epoch_i=epoch_i,
                                 data_seq=train_seq, save_best=False, train_set=True)
            icp_pose   = np.stack([t.detach().cpu() for t in icp_poses_list])
            pgo_pose   = np.stack([t.detach().cpu() for t in pgo_poses_list])
            label_pose = np.stack([t.detach().cpu() for t in label_poses_list])
            gt_pose    = np.stack([t.detach().cpu() for t in gt_poses_list]) if gt_poses_list else np.array([])
            plot_pose_trajs(icp_pose, pgo_pose, label_pose, gt_pose,
                            os.path.join(ckpt_dir, f"{epoch_i:04d}.png"),
                            data_seq=train_seq, epoch_i=epoch_i, train_set=True)

            if len(gt_poses_list) > 1:
                gt_poses_np    = np.stack([t.detach().cpu().numpy() for t in gt_poses_list])
                pgo_poses_np   = np.stack([t.detach().cpu().numpy() for t in pgo_poses_list])
                icp_poses_np   = np.stack([t.detach().cpu().numpy() for t in icp_poses_list])
                label_poses_np = np.stack([t.detach().cpu().numpy() for t in label_poses_list])

        print(f"  * Total Train Loss in {epoch_i} epoch: {epoch_train_loss / len(args.train_seqs)} *   ")

        monitor.log_epoch_summary(
            epoch=epoch_i,
            train_losses={'total_loss': epoch_train_loss / len(args.train_seqs)},
            val_losses={'total_loss': epoch_valid_loss / len(args.valid_seqs)} if args.valid_seqs else None
        )

        epoch_train_loss = 0

        for valid_seq, valid_dataset, valid_loader in valid_packs:
            print(f"  * Validating on sequence: {valid_seq} *   ")

            init_list(valid_dataset)

            integrator = IMUIntegrator(init_state=valid_dataset.init, device=args.device)

            gravity = valid_dataset.gravity
            valid_loss = validate(data_loader=valid_loader, network=network, integrator=integrator, data_seq=valid_seq, epoch_i=epoch_i)

            epoch_valid_loss += valid_loss
            print(f"  * Valid Loss: {valid_loss} *   ")
            ckpt_dir = save_ckpt(network=network, optimizer=optimizer, epoch_i=epoch_i,
                                 data_seq=valid_seq, save_best=False, valid_set=True)
            icp_pose = np.stack([t.detach().cpu() for t in icp_poses_list])
            pgo_pose = np.stack([t.detach().cpu() for t in pgo_poses_list])
            gt_pose  = np.stack([t.detach().cpu() for t in gt_poses_list]) if gt_poses_list else np.array([])
            plot_pose_trajs(icp_pose, pgo_pose, None, gt_pose,
                            os.path.join(ckpt_dir, f"{epoch_i:04d}.png"),
                            data_seq=valid_seq, epoch_i=epoch_i, train_set=False)

            if len(gt_poses_list) > 1:
                gt_poses_np  = np.stack([t.detach().cpu().numpy() for t in gt_poses_list])
                pgo_poses_np = np.stack([t.detach().cpu().numpy() for t in pgo_poses_list])
                icp_poses_np = np.stack([t.detach().cpu().numpy() for t in icp_poses_list])

        print(f"  * Total Valid Loss in {epoch_i} epoch: {epoch_valid_loss / len(args.valid_seqs)} *   ")

        epoch_valid_loss = epoch_valid_loss / len(args.valid_seqs)
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            ckpt_dir = save_ckpt(network=network, optimizer=optimizer, epoch_i=epoch_i,
                                 save_best=True, train_set=False, valid_set=False)
            print(f"  * Update Best Valid Loss: {best_valid_loss} *   ")

        if args.scheduler in ('CosineAnnealingLR', 'StepLR'):
            scheduler.step()
        else:
            scheduler.step(epoch_valid_loss)

        epoch_valid_loss = 0

    if hasattr(train, "_cov_ctrl"):
        save_component_usage_histogram(train._cov_ctrl, args.result_dir)

    monitor.close()
    print("Training completed. TensorBoard logs saved.")
