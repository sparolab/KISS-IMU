from mpmath.ctx_iv import ivmpf_constant
import torch
from torch import nn

import numpy as np

import pypose as pp
import pypose.optim.solver as ppos
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau

class GraphOptimizer(nn.Module):
    def __init__(self, nodes, vels):
        super().__init__()
        
        self.nodes = pp.Parameter(nodes.clone())
        self.vels = torch.nn.Parameter(vels.clone())
        
    def forward(self, icp_factors, imu_drots, imu_dtrans, imu_dvels, imu_dts, gravity):
        nodes = self.nodes           # SE3 (N)
        vels  = self.vels            # (N,3)
        E = len(nodes) - 1

        g = gravity

        # ---------- Pose-between (LO) ----------
        pose1 = nodes[:-1]
        pose2 = nodes[1:]
        lo_err = (icp_factors.Inv() @ pose1.Inv() @ pose2).Log().tensor()     # (E,6)

        # ---------- IMU Rotation ----------
        rot1 = nodes.rotation()[:-1]   # SO3 (E)
        rot2 = nodes.rotation()[1:]
        imu_rot_err = (imu_drots.Inv() @ rot1.Inv() @ rot2).Log().tensor()    # (E,3)

        dt     = imu_dts.reshape(-1, 1)                                      # (E,1)
        R_i    = rot1.matrix()                                               # (E,3,3)
        R_i_T  = R_i.transpose(-1, -2)                                       # (E,3,3)

        # ---------- IMU Velocity ----------
        dv_world = (vels[1:] - vels[:-1]) - g * dt                            # (E,3)
        dv_i     = torch.einsum('nij,nj->ni', R_i_T, dv_world)                # (E,3)

        if getattr(self, 'imu_delta_in_body_frame', True):
            imu_vel_err = dv_i - imu_dvels                                    # (E,3)
        else:
            imu_vel_err = (vels[1:] - vels[:-1]) - g * dt - imu_dvels         # (E,3)

        # ---------- IMU Translation ----------
        p_i    = nodes.translation()[:-1]                                     # (E,3)
        p_ip1  = nodes.translation()[1:]
        dp_w   = (p_ip1 - p_i) - vels[:-1]*dt - 0.5 * g * (dt**2)             # (E,3)
        dp_i   = torch.einsum('nij,nj->ni', R_i_T, dp_w)                      # (E,3)

        if getattr(self, 'imu_delta_in_body_frame', True):
            imu_trans_err = dp_i - imu_dtrans                                 # (E,3)
        else:
            imu_trans_err = (p_ip1 - p_i) - vels[:-1]*dt - 0.5*g*(dt**2) - imu_dtrans

        return lo_err, imu_rot_err, imu_vel_err, imu_trans_err
    
    def align_to(self, target, idx=0):
        source = self.nodes[idx].detach()
        
        inv_source = source.Inv()

        correction = target @ inv_source 
        nodes = correction @ self.nodes
        vels = correction.rotation() @ self.vels
        
        rotation = nodes.rotation()
        rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True)
        nodes = pp.SE3(torch.cat([nodes.translation(), rotation], dim=-1))  
        return nodes, vels
        
    def set_constant_weights(self, loss_weight, device):
        icp_rot_info   = np.ones(len(self.nodes)-1) * loss_weight[0]**2
        icp_trans_info = np.ones(len(self.nodes)-1) * loss_weight[1]**2
        imu_rot_info   = np.ones(len(self.nodes)-1) * loss_weight[2]**2
        imu_vel_info   = np.ones(len(self.nodes)-1) * loss_weight[3]**2
        transvel_info  = np.ones(len(self.nodes)-1) * loss_weight[4]**2

        lo_info_mat       = [torch.diag(torch.tensor([icp_trans_info[i]]*3 + [icp_rot_info[i]]*3))
                             for i in range(len(icp_trans_info))]
        imu_rot_info_mat  = [torch.diag(torch.tensor([imu_rot_info[i]]*3)) for i in range(len(imu_rot_info))]
        imu_vel_info_mat  = [torch.diag(torch.tensor([imu_vel_info[i]]*3)) for i in range(len(imu_vel_info))]
        transvel_info_mat = [torch.diag(torch.tensor([transvel_info[i]]*3)) for i in range(len(transvel_info))]
        
        lo_info_mat = torch.stack(lo_info_mat).to(device).to(torch.float32)
        imu_rot_info_mat = torch.stack(imu_rot_info_mat).to(device).to(torch.float32)
        imu_vel_info_mat = torch.stack(imu_vel_info_mat).to(device).to(torch.float32)
        transvel_info_mat = torch.stack(transvel_info_mat).to(device).to(torch.float32)

        weights = [lo_info_mat, imu_rot_info_mat, imu_vel_info_mat, transvel_info_mat]        
        return weights

    def set_adaptive_weights(self, icp_overlaps, imu_covariances, device):        
        # ========= ICP Weights ================= 
        icp_overlaps = icp_overlaps.detach().to(device) 
        icp_rot_info = (icp_overlaps * 2)**2
        icp_trans_info = (icp_overlaps * 0.5)*2
        icp_info_mat = [torch.diag(torch.tensor([icp_trans_info[i]]*3 + [icp_rot_info[i]]*3)) 
                        for i in range(len(icp_trans_info))] 
        icp_info_mat = torch.stack(icp_info_mat).to(device).to(torch.float32)
        
        # ========= IMU Weights ================= 
        imu_covariances = torch.abs(imu_covariances).detach().to(device) 
        # Extract diagonal elements from 3D covariance matrices 
        imu_rot_dcov = torch.diagonal(imu_covariances[:, :3, :3], dim1=1, dim2=2) 
        # (E, 3) 
        imu_vel_dcov = torch.diagonal(imu_covariances[:, 3:6, 3:6], dim1=1, dim2=2) # (E, 3) 
        imu_trans_dcov = torch.diagonal(imu_covariances[:, 6:9, 6:9], dim1=1, dim2=2) # (E, 3) 

        imu_rot_dcov = torch.diag_embed(imu_rot_dcov) # (E, 3, 3) 
        imu_vel_dcov = torch.diag_embed(imu_vel_dcov) # (E, 3, 3) 
        imu_trans_dcov = torch.diag_embed(imu_trans_dcov) # (E, 3, 3) 

        eigvals_rr = torch.linalg.eigvalsh(imu_rot_dcov) # (E, 3) 
        eigvals_vv = torch.linalg.eigvalsh(imu_vel_dcov) # (E, 3) 
        eigvals_tt = torch.linalg.eigvalsh(imu_trans_dcov) # (E, 3) 
        
        # ================ Forest_new ================
        # icp_overlaps = icp_overlaps.detach().to(device) 
        # icp_rot_info = (icp_overlaps * 2)**2
        # icp_trans_info = (icp_overlaps * 0.5)*2
        # icp_info_mat = [torch.diag(torch.tensor([icp_trans_info[i]]*3 + [icp_rot_info[i]]*3)) 
        #                 for i in range(len(icp_trans_info))] 
        # icp_info_mat = torch.stack(icp_info_mat).to(device).to(torch.float32)
        # imu_rot_dcov = torch.diag_embed(torch.log10(1/eigvals_rr)*1.5)**2 # (E, 3, 3)  0.5 -> 1.0 -> 1.5 -> 2 -> 2.5
        # imu_vel_dcov = torch.diag_embed(torch.log10(1/eigvals_vv) * 1e-2)**2 # (E, 3, 3)  2e-2 -> 5e-2
        # imu_trans_dcov = torch.diag_embed(torch.log10(1/eigvals_tt) * 1e-2)**2 # (E, 3, 3) 4e-2 -> 5e-2
        # ================================================
        
        # ================ small_gicp with submap (1,0.1,1,0.1,0.1) ================
        icp_overlaps = icp_overlaps.detach().to(device) 
        icp_rot_info = (icp_overlaps * 2.5)**2
        icp_trans_info = (icp_overlaps * 0.5)*2
        icp_info_mat = [torch.diag(torch.tensor([icp_trans_info[i]]*3 + [icp_rot_info[i]]*3)) 
                        for i in range(len(icp_trans_info))] 
        icp_info_mat = torch.stack(icp_info_mat).to(device).to(torch.float32)

        imu_rot_dcov = torch.diag_embed(torch.log10(1/eigvals_rr)*5e-02)**2 # (E, 3, 3)  0.5 -> 1.0 -> 1.5 -> 2 -> 2.5
        imu_vel_dcov = torch.diag_embed(torch.log10(1/eigvals_vv) * 5e-03)**2 # (E, 3, 3)  2e-2 -> 5e-2
        imu_trans_dcov = torch.diag_embed(torch.log10(1/eigvals_tt) * 5e-03)**2 # (E, 3, 3) 4e-2 -> 5e-2
        # ===============================================================

        imu_rot_info_mat = imu_rot_dcov.to(device).to(torch.float32) # (E, 3, 3) 
        imu_vel_info_mat = imu_vel_dcov.to(device).to(torch.float32) # (E, 3, 3) 
        imu_trans_info_mat = imu_trans_dcov.to(device).to(torch.float32) # (E, 3, 3) 
        # ======================================= 
        
        weights = [icp_info_mat, imu_rot_info_mat, imu_vel_info_mat, imu_trans_info_mat] 
        return weights
        
        
    
def optimize(nodes, vels, icp_factors, imu_drots, imu_dtrans, imu_dvels, imu_dts, gravity, is_valid=False, icp_weights=None, imu_weights=None, weights=[1,0.1,1,0.1,0.1], device='cuda:0'):
    go = GraphOptimizer(nodes.detach(), vels.detach())
    
    if icp_weights is not None and imu_weights is not None:
        weights = go.set_adaptive_weights(icp_weights, imu_weights, device=device)
    else:
        weights = go.set_constant_weights(weights, device=device)
        
    icp_factors = icp_factors.detach().to(device)
    imu_drots   = imu_drots.detach().to(device)
    imu_dtrans  = imu_dtrans.detach().to(device)
    imu_dvels   = imu_dvels.detach().to(device)
    imu_dts     = imu_dts.detach().to(device)
    gravity     = gravity.detach().to(device)
    
    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=1e3) # original = 1e3
    
    optimizer = pp.optim.LM(go, solver=solver, strategy=strategy, min=1e-4, vectorize=True)
    # optimizer = pp.optim.GaussNewton(go, solver=solver, vectorize=True)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=False)

    while scheduler.continual():
        loss = optimizer.step(input=(icp_factors, imu_drots, imu_dtrans, imu_dvels, imu_dts, gravity), weight=weights)
        scheduler.step(loss)

    nodes, vels = go.align_to(nodes[0])
    return nodes, vels