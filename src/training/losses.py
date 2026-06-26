# utils/losses.py
import math
import torch
import torch.nn.functional as F
import pypose as pp

def _to_BT(x, feat_nd=1):
    """
    Force the input tensor into (B, T, D...) layout.
    - feat_nd: number of feature dims (e.g. (T,3) -> feat_nd=1, (T,9,9) -> feat_nd=2)
    Rules:
      * x.ndim == 1+feat_nd  -> (T, ...) => (1, T, ...)
      * x.ndim == 2+feat_nd  -> (B, T, ... ) as-is
    """
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.ndim == 1 + feat_nd:
        return x.unsqueeze(0)          # (1, T, ...)
    elif x.ndim == 2 + feat_nd:
        return x                       # (B, T, ...)
    else:
        raise ValueError(f"Unexpected shape {tuple(x.shape)} for feat_nd={feat_nd}")

def _reduce_BT_energy(x, reduction: str):
    """
    x: squared-error / energy in (B, T, D) layout.
    - reduction='none' -> returns (B,) (averaged over T, D)
    - reduction='mean' -> scalar
    """
    # (B, T)
    per_t = x.sum(dim=-1)
    # (B,)
    per_b = per_t.mean(dim=1)
    if reduction == "none":
        return per_b
    elif reduction == "mean":
        return per_b.mean()
    else:
        raise ValueError(f"Unknown reduction={reduction}")

def _reduce_BT_nll(x, reduction: str):
    """
    x: NLL terms in (B, T, D) layout (already including 0.5*(...)+const).
    Reduced the same way as above.
    """
    return _reduce_BT_energy(x, reduction)

def get_losses(imu_poses, imu_vels, imu_motions, imu_dvels, imu_covs,
               label_poses, label_vels, label_moitons, label_dvels, args,
               add_nll_const: bool = True,
               reduction: str = "mean"):
    """
    Returns:
      - reduction='mean' : 6 scalars
      - reduction='none' : 6 (B,) tensors (per sample)
    Input tensor shapes may be (T, ...) or (B, T, ...).
    Node losses for the first node (start state) are dropped (only T-1 used).
    """
    # ---------- Edge(state) losses ----------
    # (T,3) / (B,T,3)
    adjvelerr  = imu_dvels - label_dvels
    # (T,6) / (B,T,6)  (here T is the number of edges = num_nodes - 1)
    error_edge = (imu_motions.Inv() @ label_moitons).Log().tensor()

    # align to (B,T,3)/(B,T,6)
    adjvelerr_bt  = _to_BT(adjvelerr,  feat_nd=1)
    pose_edge_bt  = _to_BT(error_edge, feat_nd=1)

    rot_edge_bt = pose_edge_bt[..., 3:]    # (B,T,3)
    pos_edge_bt = pose_edge_bt[..., :3]    # (B,T,3)

    rot_loss = _reduce_BT_energy(rot_edge_bt.pow(2), reduction)   # scalar or (B,)
    vel_loss = _reduce_BT_energy(adjvelerr_bt.pow(2), reduction)
    pos_loss = _reduce_BT_energy(pos_edge_bt.pow(2), reduction)

    # ---------- Node(covariance) losses: exclude the first node ----------
    # (T+1,6)/(B,T+1,6)
    error_node = (label_poses.Inv() @ imu_poses).Log().tensor()
    error_node_bt = _to_BT(error_node, feat_nd=1)

    # drop node 0 -> (B, T, 6)
    error_node_bt = error_node_bt[:, 1:, :]

    roterr_bt = error_node_bt[..., 3:].pow(2)   # (B,T,3)
    poserr_bt = error_node_bt[..., :3].pow(2)   # (B,T,3)
    velerr_bt = (label_vels - imu_vels)         # (T+1,3)/(B,T+1,3)
    velerr_bt = _to_BT(velerr_bt, feat_nd=1)[:, 1:, :].pow(2)  # (B,T,3)

    # --- prepare covariance diagonal ---
    # imu_covs: (T+1,9,9)/(B,T+1,9,9) -> (B,T+1,9,9)
    imu_covs_bt = _to_BT(imu_covs.squeeze(1), feat_nd=2)
    raw_diag    = torch.diagonal(imu_covs_bt, dim1=-2, dim2=-1)      # (B,T+1,9)
    # sigma2      = (F.softplus(raw_diag) + 1e-8).clamp(1e-8, 1e2)     # (B,T+1,9)
    raw_diag      = raw_diag[:, 1:, :]                                   # (B,T,9)

    # variance tempering
    t_r = getattr(args, "cov_r_temp", 1.0)
    t_v = getattr(args, "cov_v_temp", 1.0)
    t_t = getattr(args, "cov_t_temp", 1.0)

    rot_cov_diag = (raw_diag[..., 0:3] * t_r).clamp(1e-8, 1e2)         # (B,T,3)
    vel_cov_diag = (raw_diag[..., 3:6] * t_v).clamp(1e-8, 1e2)
    pos_cov_diag = (raw_diag[..., 6:9] * t_t).clamp(1e-8, 1e2)
    
    rot_cov_term = 0.5 * (roterr_bt / rot_cov_diag + args.rot_cov_scaler * torch.log(rot_cov_diag))
    vel_cov_term = 0.5 * (velerr_bt / vel_cov_diag + args.vel_cov_scaler * torch.log(vel_cov_diag))
    pos_cov_term = 0.5 * (poserr_bt / pos_cov_diag + args.pos_cov_scaler * torch.log(pos_cov_diag))

    rot_cov_loss = _reduce_BT_nll(rot_cov_term, reduction)
    vel_cov_loss = _reduce_BT_nll(vel_cov_term, reduction)
    pos_cov_loss = _reduce_BT_nll(pos_cov_term, reduction)

    return rot_loss, vel_loss, pos_loss, rot_cov_loss, vel_cov_loss, pos_cov_loss


def get_valid_losses(imu_poses, imu_vels, label_poses, label_vels, args):
    """
    Validity loss (scalar). Input may be (T,...) or (B,T,...).
    Averaged after dropping the first node.
    """
    error = (label_poses.Inv() @ imu_poses).Log().tensor()  # (T+1,6)/(B,T+1,6)
    error_bt = _to_BT(error, feat_nd=1)                     # (B,T+1,6)
    err_no0 = error_bt[:, 1:, :]                            # (B,T,6)
    # (B,)
    per_b = torch.sqrt(err_no0.pow(2).sum(dim=-1).mean(dim=1))
    return per_b.mean()
