import os
import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless-friendly backend
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from torch.utils.tensorboard import SummaryWriter

import pypose as pp

import torch
import torch.optim as optim
import torch.utils.data as Data

from data.seq_dataset import SeqDataset
from data.collate import collate_fn

from training.losses import get_losses, get_valid_losses
from training.integrator import IMUIntegrator
from utils.arguments import get_args
from utils.overlap_score import calc_symmetric_overlap, batch_pose_aligned_overlap

from models.imu_net import IMUNet
from models.lo_module import LOModule
from models.pvgo import optimize


import torch
import matplotlib
import matplotlib.pyplot as plt

args = get_args()


def enforce_equal_limits(ax, pad=0.0, anchor='min'):
    ax.relim()
    ax.autoscale_view()

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    x0, x1 = x0 - pad, x1 + pad
    y0, y1 = y0 - pad, y1 + pad

    w = x1 - x0
    h = y1 - y0
    s = max(w, h)

    if anchor == 'min':
        ax.set_xlim(x0, x0 + s)
        ax.set_ylim(y0, y0 + s)

    elif anchor == 'max':
        ax.set_xlim(x1 - s, x1)
        ax.set_ylim(y1 - s, y1)

    elif anchor == 'center':
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        ax.set_xlim(cx - s/2, cx + s/2)
        ax.set_ylim(cy - s/2, cy + s/2)

    elif anchor == 'origin':
        x0 = min(x0, 0.0)
        y0 = min(y0, 0.0)
        ax.set_xlim(x0, x0 + s)
        ax.set_ylim(y0, y0 + s)

    else:
        raise ValueError("anchor must be one of {'min','max','center','origin'}")

    ax.set_aspect('equal', adjustable='box')


def update_plot(ax, icp_line, pgo_line, label_line=None, gt_line=None, icp_poses=None, pgo_poses=None, label_poses=None, gt_poses=None,):
    try:
        if gt_poses is not None and len(gt_poses) > 0:
            gt_np = np.asarray(gt_poses)
            gt_line.set_xdata(gt_np[:, 0])
            gt_line.set_ydata(gt_np[:, 1])
        if label_poses is not None and len(label_poses) > 0:
            label_np = np.asarray(label_poses)
            label_line.set_xdata(label_np[:, 0])
            label_line.set_ydata(label_np[:, 1])
        if icp_poses is not None and len(icp_poses) > 0:
            icp_np = np.asarray(icp_poses)
            icp_line.set_xdata(icp_np[:, 0])
            icp_line.set_ydata(icp_np[:, 1])
        if pgo_poses is not None and len(pgo_poses) > 0:
            pgo_np = np.asarray(pgo_poses)
            pgo_line.set_xdata(pgo_np[:, 0])
            pgo_line.set_ydata(pgo_np[:, 1])

        if gt_line is not None:
            gt_line.set_zorder(1)
        if icp_line is not None:
            icp_line.set_zorder(3)
        if pgo_line is not None:
            pgo_line.set_zorder(4)
        if label_line is not None:
            label_line.set_zorder(5)

        ax.relim()
        ax.autoscale_view()
        
        plt.draw()
        plt.pause(0.01)
        
    except Exception as e:
        print(f"Plot update error (continuing training): {e}")
        pass

def update_plot_dual(ax_xy, ax_xz, lines,
                     icp_poses=None, pgo_poses=None, label_poses=None, gt_poses=None):
    def set_2d(line, arr, ix, iy):
        if line is not None and arr is not None and arr.size:
            line.set_data(arr[:, ix], arr[:, iy])

    # 데이터 반영
    set_2d(lines.get('gt_xy'),    gt_poses,  0, 1)
    set_2d(lines.get('icp_xy'),   icp_poses, 0, 1)
    set_2d(lines.get('pgo_xy'),   pgo_poses, 0, 1)
    set_2d(lines.get('label_xy'), label_poses, 0, 1)

    set_2d(lines.get('gt_xz'),    gt_poses,  0, 2)
    set_2d(lines.get('icp_xz'),   icp_poses, 0, 2)
    set_2d(lines.get('pgo_xz'),   pgo_poses, 0, 2)
    set_2d(lines.get('label_xz'), label_poses, 0, 2)

    for ax in (ax_xy, ax_xz):
        ax.autoscale(enable=True, axis='both', tight=False)
        ax.relim(visible_only=True)
        ax.autoscale_view()
        ax.set_aspect('equal', adjustable='datalim')

    ax_xy.figure.canvas.draw_idle()
    ax_xy.figure.canvas.flush_events()

def inference(lo_model, loader, integrator, data_seq, gravity):
    total_inference_loss = 0

    plt.ion()
    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(20, 8), dpi=120, constrained_layout=True)

    gt_xy_line,  = ax_xy.plot([], [], 'y--', label='GT',  linewidth=6)
    icp_xy_line, = ax_xy.plot([], [], 'r-',  label=f'ICP({args.lo_model})', linewidth=3)
    pgo_xy_line, = ax_xy.plot([], [], 'b-',  label='PGO', linewidth=3)
    label_xy_line = None

    ax_xy.set_title(f'XY view · {data_seq}')
    ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y')
    ax_xy.grid(True); ax_xy.legend()
    ax_xy.set_aspect('equal', adjustable='datalim')

    gt_xz_line,  = ax_xz.plot([], [], 'y--', label='GT',  linewidth=6)
    icp_xz_line, = ax_xz.plot([], [], 'r-',  label=f'ICP({args.lo_model})', linewidth=3)
    pgo_xz_line, = ax_xz.plot([], [], 'b-',  label='PGO', linewidth=3)
    label_xz_line = None

    ax_xz.set_title(f'XZ view · {data_seq}')
    ax_xz.set_xlabel('X'); ax_xz.set_ylabel('Z')
    ax_xz.grid(True); ax_xz.legend()
    ax_xz.set_aspect('equal', adjustable='datalim')

    lines = {
        'gt_xy': gt_xy_line, 'icp_xy': icp_xy_line, 'pgo_xy': pgo_xy_line, 'label_xy': label_xy_line,
        'gt_xz': gt_xz_line, 'icp_xz': icp_xz_line, 'pgo_xz': pgo_xz_line, 'label_xz': label_xz_line,
    }

    with torch.no_grad():
        for i, sample in enumerate(tqdm.tqdm(loader)):
            if isinstance(inference_poses_list[-1], torch.Tensor):
                init_pos = inference_poses_list[-1][:3].clone().detach().to(args.device).float()
                init_rot = inference_poses_list[-1][3:].clone().detach().to(args.device).float()
            else:
                init_pos = torch.from_numpy(inference_poses_list[-1][:3]).to(args.device).float()
                init_rot = torch.from_numpy(inference_poses_list[-1][3:]).to(args.device).float()
                
            init_vel = inference_vels_list[-1].to(args.device)
            init_rot = init_rot / torch.linalg.norm(init_rot)
            
            init_state = { 'rot': pp.SO3(init_rot),
                            'vel': init_vel,
                            'pos': init_pos,
                            'cov': None}
            
            imu_dts = sample['imu_dts'] 
            accels  = sample['accels']
            gyros   = sample['gyros']
            valid   = sample.get('valid_length', sample.get('valid_lenth'))
            
            valid = torch.as_tensor(valid, device=imu_dts.device).flatten().to(torch.long)
            T = imu_dts.size(-1)

            L = int(valid.clamp_min(0).clamp_max(T).amin().item())

            dts    = imu_dts[..., :L]
            accels = accels[..., :L, :] if accels.dim() >= 3 else accels[..., :L]
            gyros  = gyros[..., :L, :]  if gyros.dim()  >= 3 else gyros[..., :L]
            
            corr_data = {
                'dts': dts.to(args.device),
                'accels': accels.to(args.device),
                'gyros': gyros.to(args.device),
            }

            imu_states = integrator.integrate(init=init_state, 
                                                dts=corr_data['dts'], accels=corr_data['accels'], gyros=corr_data['gyros'],
                                                motion_mode=False)
            
            init_state = { 'rot': pp.SO3(init_rot),
                            'vel': torch.zeros((1, 3), dtype=torch.float32).to(args.device),
                            'pos': torch.zeros((1, 3), dtype=torch.float32).to(args.device),
                            'cov': None}
            imu_dstates = integrator.integrate(init=init_state, 
                                                dts=corr_data['dts'], accels=corr_data['accels'], gyros=corr_data['gyros'],
                                                motion_mode=True)
            imu_nodes = pp.SE3(torch.cat([imu_states['pos'], imu_states['rot'].tensor()], dim=-1)).to(args.device)
            imu_vels = imu_states['vel'].to(args.device)
            imu_motions = pp.SE3(torch.cat([imu_dstates['pos'], imu_dstates['rot'].tensor()], dim=-1)).to(args.device)
            imu_dvels = imu_dstates['vel'].to(args.device)
            imu_covs = imu_states['cov'].to(args.device)
            imu_dcovs = imu_dstates['cov'].detach().to(args.device)
            imu_dts = torch.stack([d.sum() for d in corr_data['dts']]).unsqueeze(-1).to(args.device)
            icp_poses, icp_motions, icp_overlap_scores = lo_model(sample, pp.SE3(inference_poses_list[-1]))

            if args.use_adaptive_weight:
                pgo_poses, _ = optimize(nodes=imu_nodes, vels=imu_vels,
                                        icp_factors=icp_motions,
                                        imu_drots=imu_dstates['rot'], imu_dvels=imu_dstates['vel'], imu_dtrans=imu_dstates['pos'],
                                        imu_dts=imu_dts,
                                        weights=args.lm_weight,
                                        gravity=gravity,
                                        icp_weights=torch.from_numpy(icp_overlap_scores),
                                        imu_weights=imu_dcovs.squeeze(1),
                                        device=args.device)
            else:
                pgo_poses, _ = optimize(nodes=imu_nodes, vels=imu_vels,
                                        icp_factors=icp_motions,
                                        imu_drots=imu_dstates['rot'], imu_dvels=imu_dstates['vel'], imu_dtrans=imu_dstates['pos'],
                                        imu_dts=imu_dts,
                                        weights=args.lm_weight,
                                        gravity=gravity,
                                        icp_weights=None,  # No adaptive ICP weights
                                        imu_weights=None,  # No adaptive IMU weights
                                        device=args.device)
            
            pgo_motions = pgo_poses[:-1].Inv() @ pgo_poses[1:]
            pgo_dts = torch.stack([d.sum() for d in corr_data['dts']]).unsqueeze(-1)
            pgo_vels = pgo_motions.tensor()[:, :3]/pgo_dts
            gt_poses = pp.SE3(sample['gt_pose1']).to(args.device)

            diff = torch.diff(gt_poses.tensor()[:, :3], dim=0)
            gt_vels = torch.zeros_like(gt_poses.tensor()[:, :3])
            gt_vels[1:] = diff / imu_dts[1:, :]
            gt_vels[0] = init_vel
            
            # ======= Save List =======
            imu_poses_list.extend(imu_nodes.tensor()[1:].detach())
            imu_motions_list.extend(imu_motions.detach())
            imu_vels_list.extend(imu_vels[1:].detach())
            imu_covs_list.extend(imu_covs[1:].detach())

            icp_poses_list.extend(icp_poses.tensor()[1:].detach())
            icp_motions_list.extend(icp_motions.detach())
            icp_overlap_score_list.extend(icp_overlap_scores)
            
            pgo_poses_list.extend(pgo_poses.tensor()[1:].detach())
            pgo_motions_list.extend(pgo_motions.detach())
            pgo_vels_list.extend(pgo_vels[1:].detach())
            gt_poses_list.extend(sample['gt_pose1'][1:].clone().detach())
            # ========================

            
            imu_poses = pp.SE3(torch.cat([imu_states['pos'][1:, :], imu_states['rot'].tensor()[1:, :]], dim=-1)).to(args.device)
            inference_loss = get_valid_losses(imu_poses, imu_vels[1:], gt_poses, gt_vels, args)
            inference_vels_list.extend(pgo_vels[1:].detach())
            inference_poses_list.extend(pgo_poses.tensor()[1:].clone().detach())
            
            total_inference_loss += inference_loss.item()
            pgo_pose = np.stack([t.detach().cpu() for t in pgo_poses_list]) if pgo_poses_list else np.array([])
            icp_pose = np.stack([t.detach().cpu() for t in icp_poses_list]) if icp_poses_list else np.array([])
            gt_pose = np.stack([t.detach().cpu() for t in gt_poses_list]) if gt_poses_list else np.array([])
            update_plot_dual(ax_xy, ax_xz, lines,
                            pgo_poses=pgo_pose, icp_poses=icp_pose,
                            gt_poses=gt_pose, label_poses=None)
    if args.use_adaptive_weight:
        base_dir = os.path.join(args.result_dir, "raw_with_adaptive_weight", data_seq)
    else:
        base_dir = os.path.join(args.result_dir, "raw_with_fixed_weight", data_seq)
    os.makedirs(base_dir, exist_ok=True)
    save_path = os.path.join(base_dir, f"raw.png")
    os.makedirs(base_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)        
    return total_inference_loss / len(loader)
       
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
    
    global pgo_poses_list, pgo_motions_list, pgo_vels_list,pgo_overlap_score_list
    pgo_poses_list = [torch.from_numpy(init_pose).reshape(7)]
    pgo_motions_list = []
    pgo_vels_list = [init_state['vel']]
    pgo_overlap_score_list = []
    
    global label_poses_list, label_motions_list, label_vels_list
    label_poses_list = [torch.from_numpy(init_pose).reshape(7)]
    label_motions_list = []
    label_vels_list = [init_state['vel']]
    
    global inference_poses_list, inference_vels_list
    inference_poses_list = [torch.from_numpy(init_pose).reshape(7)]
    inference_vels_list = [init_state['vel']]
    
    global gt_poses_list
    gt_poses_list = []


def plot_pose_trajs(icp_poses, pgo_poses, label_poses, gt_poses, filename):
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
    ax.set_title('Pose Trajectories')
    ax.grid(True)
    ax.axis('equal')
    ax.legend()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    
def save_ckpt(data_seq=None):
    if args.use_adaptive_weight:
        ckpt_dir = os.path.join(args.result_dir, "raw_with_adaptive_weight", data_seq)
    else:
        ckpt_dir = os.path.join(args.result_dir, "raw_with_fixed_weight", data_seq)
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"raw.ckpt")
    torch.save({
        'imu_poses_list': np.stack([t.detach().cpu() for t in imu_poses_list]),
        'imu_motions_list': np.stack([t.detach().cpu() for t in imu_motions_list]),
        'imu_vels_list': np.stack([t.detach().cpu() for t in imu_vels_list]),
        'imu_covs_list': np.stack([t.detach().cpu() for t in imu_covs_list]),
        'icp_poses_list': np.stack([t.detach().cpu() for t in icp_poses_list]),
        'icp_motions_list': np.stack([t.detach().cpu() for t in icp_motions_list]),
        'icp_vels_list': np.stack([t.detach().cpu() for t in icp_vels_list]),
        'pgo_poses_list': np.stack([t.detach().cpu() for t in pgo_poses_list]),
        'pgo_motions_list': np.stack([t.detach().cpu() for t in pgo_motions_list]),
        'pgo_vels_list': np.stack([t.detach().cpu() for t in pgo_vels_list]),
    }, save_path)
    return ckpt_dir
    
if __name__ == "__main__":
    plt.rcParams['figure.max_open_warning'] = 0
        
    for inference_seq in args.inference_seqs:
        print(f"  * Inferencing on sequence: {inference_seq} *   ")
        
        inference_dataset = SeqDataset(data_root=args.data_root, data_seq=inference_seq, data_type=args.data_type)
        inference_loader = Data.DataLoader(dataset=inference_dataset, batch_size=args.batch_size, num_workers=args.worker_num, shuffle=False, drop_last=True, collate_fn=collate_fn)
        init_list(inference_dataset)
        
        lo_model = LOModule(lo_model=args.lo_model, T_I_L=inference_dataset.T_I_G, R_I_L=inference_dataset.R_I_L, init_state=icp_poses_list[-1], device_id=args.device)
        integrator = IMUIntegrator(init_state=inference_dataset.init, device=args.device)
        
        inference_loss = inference(lo_model=lo_model, loader=inference_loader, integrator=integrator, data_seq=inference_seq, gravity=inference_dataset.gravity)
        print(f"  * Inference Loss: {inference_loss} *   ")
        ckpt_dir = save_ckpt(data_seq=inference_seq)
        icp_pose = np.stack([t.detach().cpu() for t in icp_poses_list])
        pgo_pose = np.stack([t.detach().cpu() for t in pgo_poses_list])
        gt_pose = np.stack([t.detach().cpu() for t in gt_poses_list]) if gt_poses_list else np.array([])
    print(f"  * Total Inference Loss : {inference_loss} *   ")
        