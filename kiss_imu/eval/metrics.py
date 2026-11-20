"""Trajectory metrics against GT. Estimates are TUM rows (t, x, y, z, qx, qy, qz, qw).

ate_rmse            no alignment beyond the start: the first pose's yaw and position are put on the GT (identity for
                    runs started at the GT pose; GT-free starts, run.init=static, have an arbitrary heading/origin)
ate_aligned_rmse    after an SE(3) (Umeyama, no scale) alignment of the positions
rot_rmse_deg        orientation error, same start alignment (yaw only, so an initial tilt error counts)
rpe_pct             relative translation error over rpe_dist metres, % (KITTI-style, see evaluate)
"""

import numpy as np
from scipy.spatial.transform import Rotation


def umeyama_se3(src, dst):
    """R, t minimising |dst - (R src + t)| (no scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def _chordal_mean(Rs):
    U, _, Vt = np.linalg.svd(Rs.sum(0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def evaluate(est, seq, rpe_dist=10.0, x_window_s=30.0):
    t = est[:, 0]
    ok = seq.in_gt_span(t)
    t, p = t[ok], est[ok, 1:4]
    Rq = Rotation.from_quat(est[ok, 4:8]).as_matrix()
    gR, gp, _ = seq.gt_states(t)
    A = gR[0] @ Rq[0].T                                  # start alignment: heading only, then the origin
    yaw = np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1])
    Az = Rotation.from_euler('z', yaw).as_matrix()
    if not np.allclose(Az, np.eye(3), atol=1e-9) or np.linalg.norm(p[0] - gp[0]) > 1e-6:
        p = (p - p[0]) @ Az.T + gp[0]
        Rq = np.einsum('ij,njk->nik', Az, Rq)
    err = np.linalg.norm(p - gp, axis=1)
    R, tt = umeyama_se3(p, gp)
    err_al = np.linalg.norm(p @ R.T + tt - gp, axis=1)
    rot_err = np.degrees(np.linalg.norm(Rotation.from_matrix(np.einsum('nji,njk->nik', gR, Rq)).as_rotvec(), axis=1))
    # RPE (KITTI-style): for segments of rpe_dist metres of GT path, the displacement in the start frame,
    # estimate vs GT, as % of the distance. A constant rotation X between the GT body and the IMU
    # (BotanicGarden: ~2 deg of yaw) would read as drift, so it is estimated (chordal mean of R_gt^T R_est over
    # the first x_window_s seconds, before the estimate drifts) and removed: the estimate's displacement is
    # taken in the frame R_est X^T
    first = t <= t[0] + x_window_s
    X = _chordal_mean(np.einsum('nji,njk->nik', gR[first], Rq[first]))
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(gp, axis=0), axis=1))]
    rpe = []
    for a in range(len(t)):
        b = np.searchsorted(s, s[a] + rpe_dist)
        if b >= len(t):
            break
        d_est = X @ Rq[a].T @ (p[b] - p[a])
        d_gt = gR[a].T @ (gp[b] - gp[a])
        rpe.append(np.linalg.norm(d_est - d_gt) / rpe_dist * 100)
    return dict(n=int(ok.sum()), length_m=float(s[-1]),
                ate_rmse=float(np.sqrt(np.mean(err ** 2))), ate_aligned_rmse=float(np.sqrt(np.mean(err_al ** 2))),
                final_err=float(err[-1]), rot_rmse_deg=float(np.sqrt(np.mean(rot_err ** 2))),
                rpe_pct=float(np.mean(rpe)) if rpe else float('nan'))
