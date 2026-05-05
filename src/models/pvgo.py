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
        """
        GTSAM/Forster 스타일 residuals
        - Δp_imu, Δv_imu 는 i(바디) 프레임 기준이라고 가정 (필요시 self.imu_delta_in_body_frame로 전환)
        """
        nodes = self.nodes           # SE3 (N)
        vels  = self.vels            # (N,3)
        E = len(nodes) - 1

        # 안전한 중력값 (self.g 없으면 기본값 사용)
        g = gravity

        # ---------- Pose-between (LO) ----------
        pose1 = nodes[:-1]
        pose2 = nodes[1:]
        lo_err = (icp_factors.Inv() @ pose1.Inv() @ pose2).Log().tensor()     # (E,6)

        # ---------- IMU Rotation ----------
        rot1 = nodes.rotation()[:-1]   # SO3 (E)
        rot2 = nodes.rotation()[1:]
        imu_rot_err = (imu_drots.Inv() @ rot1.Inv() @ rot2).Log().tensor()    # (E,3)

        # 공통
        dt     = imu_dts.reshape(-1, 1)                                      # (E,1)
        R_i    = rot1.matrix()                                               # (E,3,3)
        R_i_T  = R_i.transpose(-1, -2)                                       # (E,3,3)

        # ---------- IMU Velocity ----------
        # r_v = R_i^T (v_{i+1}-v_i - g*dt) - Δv_imu     [Δv_imu in body-i]
        dv_world = (vels[1:] - vels[:-1]) - g * dt                            # (E,3)
        dv_i     = torch.einsum('nij,nj->ni', R_i_T, dv_world)                # (E,3)

        if getattr(self, 'imu_delta_in_body_frame', True):
            imu_vel_err = dv_i - imu_dvels                                    # (E,3)
        else:
            imu_vel_err = (vels[1:] - vels[:-1]) - g * dt - imu_dvels         # (E,3)

        # ---------- IMU Translation ----------
        # r_p = R_i^T (p_{i+1}-p_i - v_i*dt - 0.5*g*dt^2) - Δp_imu  [Δp_imu in body-i]
        p_i    = nodes.translation()[:-1]                                     # (E,3)
        p_ip1  = nodes.translation()[1:]
        dp_w   = (p_ip1 - p_i) - vels[:-1]*dt - 0.5 * g * (dt**2)             # (E,3)
        dp_i   = torch.einsum('nij,nj->ni', R_i_T, dp_w)                      # (E,3)

        if getattr(self, 'imu_delta_in_body_frame', True):
            imu_trans_err = dp_i - imu_dtrans                                 # (E,3)
        else:
            imu_trans_err = (p_ip1 - p_i) - vels[:-1]*dt - 0.5*g*(dt**2) - imu_dtrans

        return lo_err, imu_rot_err, imu_vel_err, imu_trans_err

    
    # def forward(self, icp_factors, imu_drots, imu_dtrans, imu_dvels, imu_dts):
    #     nodes = self.nodes
    #     vels = self.vels
        
    #     trans_std = 0.5
    #     rot_std   = 0.03
    #     vel_std   = 0.5
    
    #     ## ===== LO Constraints =============================
    #     num_nodes = len(nodes)
    #     pose1 = nodes[:num_nodes-1, :]
    #     pose2 = nodes[1:num_nodes, :]
    #     error = icp_factors.Inv() @ pose1.Inv() @ pose2
    #     pgerr_raw = error.Log().tensor()
    #     # <- 여기서 inplace 연산 대신!
    #     pgerr = torch.cat([pgerr_raw[:, 0:3] / trans_std, pgerr_raw[:, 3:6] / rot_std], dim=-1)
    #     ## ==================================================

    #     ## ===== Delta Velocity Constraints =================
    #     adjvelerr = imu_dvels - torch.diff(vels, dim=0)
    #     adjvelerr = adjvelerr / vel_std   # (out-of-place)
    #     ## ==================================================

    #     ## ===== IMU Rotation Constraints ===================
    #     rot1 = nodes.rotation()[:-1]
    #     rot2 = nodes.rotation()[1:]
    #     error = imu_drots.Inv() @ rot1.Inv() @ rot2
    #     imuroterr = error.Log().tensor() / rot_std  # (out-of-place)
    #     ## ==================================================

    #     ## ===== IMU Translation Constraints ================
    #     trans = nodes.translation()
    #     transvelerr = (torch.diff(trans, dim=0) - (vels[:-1]*imu_dts + imu_dtrans)) / trans_std
    #     ## ==================================================

    #     return pgerr, imuroterr, adjvelerr, transvelerr

    
    def align_to(self, target, idx=0):
        # align nodes[idx] to target
        source = self.nodes[idx].detach()
        
        inv_source = source.Inv()

        correction = target @ inv_source 
        nodes = correction @ self.nodes
        vels = correction.rotation() @ self.vels
        
        rotation = nodes.rotation()
        rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True)
        nodes = pp.SE3(torch.cat([nodes.translation(), rotation], dim=-1))  
        return nodes, vels
        
    def set_weights(self, loss_weight, device):
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

        # weights = [lo_info_mat, imu_rot_info_mat, transvel_info_mat]        
        weights = [lo_info_mat, imu_rot_info_mat, imu_vel_info_mat, transvel_info_mat]        
        return weights

    def set_adaptive_weights(self, icp_overlaps, imu_covariances, device):
        # icp_overlaps: (E)
        # imu_covariances: (E, 9, 9)
        
        # ========= ICP Weights =================
        # icp_overlaps = icp_overlaps.detach().to(device)
        # # icp_rot_info = 1.0
        # # icp_trans_info = 0.01
        # # icp_info_mat   = [torch.diag(torch.tensor([icp_trans_info[i]]*3 + [icp_rot_info[i]]*3))
        # #                   for i in range(len(icp_trans_info))]
        # # icp_info_mat = torch.stack(icp_info_mat).to(device).to(torch.float32) # (E, 6, 6)
        # E = len(icp_overlaps)

        # icp_rot_info_val   = 1.0
        # icp_trans_info_val = 0.01
        # icp_info_mat = torch.diag_embed(
        #     torch.cat([
        #         torch.full((E,3), icp_trans_info_val, device=device),
        #         torch.full((E,3), icp_rot_info_val,   device=device)
        #     ], dim=1).to(torch.float32)
        # ).to(device)  # (E,6,6)
        # eps = 1e-12
        # ov = icp_overlaps.detach().to(device).reshape(-1).clamp_min(eps)  # (E,)

        # def rescale_1d(x, target, cmin, cmax):
        #     med = torch.median(x)
        #     scale = target / (med + eps)
        #     return (x * scale).clamp(cmin, cmax)  # (E,)

        # # 목표 중앙값: rot=1.0, trans=0.01  /  클램프 범위는 네가 정한 값으로
        # rot_info   = rescale_1d(ov, target=1.0,  cmin=0.8,   cmax=1.2)   # (E,)
        # trans_info = rescale_1d(ov, target=0.01, cmin=0.008, cmax=0.02)  # (E,)

        # # (E,6) → (E,6,6)
        # d = torch.stack([trans_info, trans_info, trans_info,
        #                 rot_info,   rot_info,   rot_info], dim=1).to(torch.float32)
        # icp_info_mat = torch.diag_embed(d)
        # # =======================================
        # # print(icp_rot_info)
        # # print(icp_trans_info)
        # # ========= IMU Weights =================
        # # ---------- IMU (공분산 → 정보) ----------
        # C = imu_covariances.detach().to(device)
        # C = 0.5 * (C + C.transpose(-1, -2))  # 대칭화
        # eps = 1e-9
        # E = C.shape[0]
        # I9 = torch.eye(9, device=device).unsqueeze(0).expand(E, -1, -1)
        # C = C + eps * I9                      # 지터

        # # 블록 분할
        # C_rr = C[:, 0:3, 0:3]
        # C_vv = C[:, 3:6, 3:6]
        # C_tt = C[:, 6:9, 6:9]

        # # 대각만 신뢰 (full inverse 쓰고 싶으면 아래 3줄 대신 inv 쓰면 됨)
        # var_rr = torch.diagonal(C_rr, dim1=1, dim2=2).clamp_min(eps)  # (E,3)
        # var_vv = torch.diagonal(C_vv, dim1=1, dim2=2).clamp_min(eps)  # (E,3)
        # var_tt = torch.diagonal(C_tt, dim1=1, dim2=2).clamp_min(eps)  # (E,3)

        # info_rr = 1.0 / var_rr     # (E,3)
        # info_vv = 1.0 / var_vv     # (E,3)
        # info_tt = 1.0 / var_tt     # (E,3)

        # # 중앙값 정렬 스케일
        # def rescale_to_target(info_diag, target_median, clip_min, clip_max):
        #     # info_diag: (E,3) 형태의 대각 정보들
        #     med = torch.median(info_diag)
        #     scaled = info_diag * (target_median / (med + 1e-12))
        #     scaled = scaled.clamp(min=clip_min, max=clip_max)
        #     return torch.diag_embed(scaled).to(torch.float32)

        # imu_rot_info_mat  = rescale_to_target(info_rr, target_median=1.0,   clip_min=0.8,   clip_max=1.2)
        # imu_vel_info_mat  = rescale_to_target(info_vv, target_median=0.01,  clip_min=0.008, clip_max=0.02)
        # imu_trans_info_mat= rescale_to_target(info_tt, target_median=0.01,  clip_min=0.008, clip_max=0.02)
        
        # print(imu_rot_info_mat)
        # print(imu_vel_info_mat)
        # print(imu_trans_info_mat)
        # print('=============')
        
        
        # # =======================================

        # weights = [icp_info_mat, imu_rot_info_mat, imu_vel_info_mat, imu_trans_info_mat]
        # return weights
        
        # ========= ICP Weights ================= 
        icp_overlaps = icp_overlaps.detach().to(device) 
        icp_rot_info = (icp_overlaps * 2)**2
        icp_trans_info = (icp_overlaps * 0.5)*2
        icp_info_mat = [torch.diag(torch.tensor([icp_trans_info[i]]*3 + [icp_rot_info[i]]*3)) 
                        for i in range(len(icp_trans_info))] 
        icp_info_mat = torch.stack(icp_info_mat).to(device).to(torch.float32)
        
        # E = icp_overlaps.shape[0]
        # d_icp = torch.cat([
        #     torch.full((E, 3), 0.01, device=device),  # tx, ty, tz
        #     torch.full((E, 3), 1.00, device=device),  # rx, ry, rz
        # ], dim=1)                                     # (E, 6)

        # icp_info_mat = torch.diag_embed(d_icp).to(device).to(torch.float32)   # (E, 6, 6)
        # ======================================= 
        
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
        
        # ================ kiss_icp (1,0.1,1,0.1,0.1) ================
        # icp_overlaps = icp_overlaps.detach().to(device) 
        # icp_rot_info = (icp_overlaps * 2)**2
        # icp_trans_info = (icp_overlaps * 0.5)*2
        # icp_info_mat = [torch.diag(torch.tensor([icp_trans_info[i]]*3 + [icp_rot_info[i]]*3)) 
        #                 for i in range(len(icp_trans_info))] 
        # icp_info_mat = torch.stack(icp_info_mat).to(device).to(torch.float32)

        # imu_rot_dcov = torch.diag_embed(torch.log10(1/eigvals_rr)*0.1)**2 # (E, 3, 3)  0.5 -> 1.0 -> 1.5 -> 2 -> 2.5
        # imu_vel_dcov = torch.diag_embed(torch.log10(1/eigvals_vv) * 0.015)**2 # (E, 3, 3)  2e-2 -> 5e-2
        # imu_trans_dcov = torch.diag_embed(torch.log10(1/eigvals_tt) * 0.015)**2 # (E, 3, 3) 4e-2 -> 5e-2
        # ===============================================================
        
        # ================ small_gicp (1,0.1,5,0.1,0.1) ================
        # icp_overlaps = icp_overlaps.detach().to(device) 
        # icp_rot_info = (icp_overlaps * 1.5)**2
        # icp_trans_info = (icp_overlaps * 0.5)*2
        # icp_info_mat = [torch.diag(torch.tensor([icp_trans_info[i]]*3 + [icp_rot_info[i]]*3)) 
        #                 for i in range(len(icp_trans_info))] 
        # icp_info_mat = torch.stack(icp_info_mat).to(device).to(torch.float32)
        # imu_rot_dcov = torch.diag_embed(torch.log10(1/eigvals_rr)*2.5)**2 # (E, 3, 3)  0.5 -> 1.0 -> 1.5 -> 2 -> 2.5
        # imu_vel_dcov = torch.diag_embed(torch.log10(1/eigvals_vv) * 2e-2)**2 # (E, 3, 3)  2e-2 -> 5e-2
        # imu_trans_dcov = torch.diag_embed(torch.log10(1/eigvals_tt) * 1e-2)**2 # (E, 3, 3) 4e-2 -> 5e-2
        # ===============================================================
        
        # ================ small_gicp (1,0.1,1,0.1,0.1) ================
        # icp_overlaps = icp_overlaps.detach().to(device) 
        # icp_rot_info = (icp_overlaps * 1.5)**2
        # icp_trans_info = (icp_overlaps * 0.5)*2
        # icp_info_mat = [torch.diag(torch.tensor([icp_trans_info[i]]*3 + [icp_rot_info[i]]*3)) 
        #                 for i in range(len(icp_trans_info))] 
        # icp_info_mat = torch.stack(icp_info_mat).to(device).to(torch.float32)
        # imu_rot_dcov = torch.diag_embed(torch.log10(1/eigvals_rr)*3)**2 # (E, 3, 3)  0.5 -> 1.0 -> 1.5 -> 2 -> 2.5
        # imu_vel_dcov = torch.diag_embed(torch.log10(1/eigvals_vv) * 2e-2)**2 # (E, 3, 3)  2e-2 -> 5e-2
        # imu_trans_dcov = torch.diag_embed(torch.log10(1/eigvals_tt) * 1e-2)**2 # (E, 3, 3) 4e-2 -> 5e-2
        # ===============================================================
        
        # ================ small_gicp with submap (1,0.1,5,0.1,0.1) ================
        # icp_overlaps = icp_overlaps.detach().to(device) 
        # icp_rot_info = (icp_overlaps * 1.5)**2
        # icp_trans_info = (icp_overlaps * 0.3)*2
        # icp_info_mat = [torch.diag(torch.tensor([icp_trans_info[i]]*3 + [icp_rot_info[i]]*3)) 
        #                 for i in range(len(icp_trans_info))] 
        # icp_info_mat = torch.stack(icp_info_mat).to(device).to(torch.float32)
        # imu_rot_dcov = torch.diag_embed(torch.log10(eigvals_rr)*0.1)**2 # (E, 3, 3)  0.5 -> 1.0 -> 1.5 -> 2 -> 2.5
        # imu_vel_dcov = torch.diag_embed(torch.log10(1/eigvals_vv) * 0.01)**2 # (E, 3, 3)  2e-2 -> 5e-2
        # imu_trans_dcov = torch.diag_embed(torch.log10(1/eigvals_tt) * 0.01)**2 # (E, 3, 3) 4e-2 -> 5e-2
        # ===============================================================
        
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
        # print(imu_rot_dcov)
        # print(imu_vel_dcov)
        # print(imu_trans_dcov)
        # print(icp_rot_info)
        # print(icp_trans_info)
        # print('=====')
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
        weights = go.set_weights(weights, device=device)
        
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