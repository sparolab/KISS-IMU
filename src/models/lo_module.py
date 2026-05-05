import torch
import torch.nn as nn

import cv2
import numpy as np
import pypose as pp
import open3d as o3d

from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor

## ======= G-ICP ==========================
from utils.point_module import *
## ========================================

## ======= Fast GICP =====================
import pygicp
## =======================================

## ======= KISS GICP =====================
from kiss_icp.config import KISSConfig
from kiss_icp.deskew import get_motion_compensator
from kiss_icp.mapping import get_voxel_hash_map
from kiss_icp.preprocess import get_preprocessor
from kiss_icp.registration import get_registration
from kiss_icp.threshold import get_threshold_estimator
from kiss_icp.voxelization import voxel_down_sample
## =======================================

## ======= Small GICP =====================
import small_gicp
## =======================================


from utils.overlap_score import calc_symmetric_overlap


# ------------------------
# Helpers
# ------------------------
def pose7_to_mat44(pose7):
    t = pose7[:3]
    q = pose7[3:]
    # float64로 유지 (E)
    q = np.array(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    Rm = Rotation.from_quat(q).as_matrix()
    U, _, Vt = np.linalg.svd(Rm)
    Rm_ortho = U @ Vt
    if np.linalg.det(Rm_ortho) < 0:
        U[:, -1] *= -1
        Rm_ortho = U @ Vt
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rm_ortho
    mat[:3, 3] = t
    return mat

def remove_nan_inf(pts):
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask]

def _apply_T_left(pts, T):
    """pts는 (N,3). 반환은 pts.dtype 유지."""
    if pts.size == 0:
        return pts
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    out = (T @ homo.T).T[:, :3]
    return out.astype(pts.dtype, copy=False)

def _voxel_downsample_np(pts, voxel):
    """결정적 다운샘플(A): unique+정렬, 랜덤 없음."""
    if pts.size == 0 or voxel <= 0:
        return pts
    grid = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(grid, axis=0, return_index=True)
    return pts[np.sort(idx)]

# ------------------------
# ICP wrappers
# ------------------------
class G_ICP:
    def __init__(self):
        self.icp_initial = np.eye(4, dtype=np.float64)

    def get_motion(self, source_raw, target_raw):
        if isinstance(self.icp_initial, pp.LieTensor):
            self.icp_initial = self.icp_initial.matrix().cpu().numpy().squeeze().astype(np.float64)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_raw.astype(np.float64))),
            target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_raw.astype(np.float64))),
            max_correspondence_distance=0.5,
            init=self.icp_initial,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=0.001, max_iteration=20
            )
        )
        T_ICP_Relative = reg_p2p.transformation.astype(np.float64)
        self.icp_initial = T_ICP_Relative
        return T_ICP_Relative


class Fast_GICP:
    def __init__(self):
        self.icp_initial = np.eye(4, dtype=np.float64)

    def get_motion(self, source_raw, target_raw):
        target = pygicp.downsample(target_raw, 3.0)
        source = pygicp.downsample(source_raw, 3.0)
        gicp = pygicp.FastGICP()
        gicp.set_num_threads(4)
        gicp.set_input_target(target.astype(np.float64))
        gicp.set_input_source(source.astype(np.float64))
        T_ICP_Relative = gicp.align().astype(np.float64)
        return T_ICP_Relative


class Small_GICP:
    def __init__(self, method='example1'):
        self.method = method
        self.icp_initial = np.eye(4, dtype=np.float64)

    def get_motion(self, source_raw, target_raw):
        if self.method == 'example1':
            motion = small_gicp.align(target_raw, source_raw, num_threads=2, downsampling_resolution=2.0)
        elif self.method == 'example2':
            target, target_tree = small_gicp.preprocess_points(target_raw, downsampling_resolution=2.0)
            source, source_tree = small_gicp.preprocess_points(source_raw, downsampling_resolution=2.0)
            motion = small_gicp.align(target, source, target_tree)
        elif self.method == 'small1':
            target_pc = small_gicp.PointCloud(target_raw)
            source_pc = small_gicp.PointCloud(source_raw)
            target, target_tree = small_gicp.preprocess_points(target_pc, downsampling_resolution=2.0)
            source, source_tree = small_gicp.preprocess_points(source_pc, downsampling_resolution=2.0)
            motion = small_gicp.align(target, source, target_tree)
        elif self.method == 'small2':
            target_pc = small_gicp.PointCloud(target_raw)
            source_pc = small_gicp.PointCloud(source_raw)
            target = small_gicp.voxelgrid_sampling(target_pc, 2.0)
            source = small_gicp.voxelgrid_sampling(source_pc, 2.0)
            target_tree = small_gicp.KdTree(target)
            source_tree = small_gicp.KdTree(source)
            small_gicp.estimate_covariances(target, target_tree)
            small_gicp.estimate_covariances(source, source_tree)
            motion = small_gicp.align(target, source, target_tree)
        self.motion = motion.T_target_source.astype(np.float64)
        return self.motion


class KISS_ICP:
    def __init__(self, last_pose):
        self.config = KISSConfig()
        self.compensator = get_motion_compensator(self.config)
        self.adaptive_threshold = get_threshold_estimator(self.config)
        self.registration = get_registration(self.config)
        self.local_map = get_voxel_hash_map(self.config)
        self.preprocess = get_preprocessor(self.config)
        self.last_pose = pose7_to_mat44(last_pose).astype(np.float64)
        self.last_delta = np.eye(4, dtype=np.float64)

    def voxelize(self, iframe, config):
        frame_downsample = voxel_down_sample(iframe, config.mapping.voxel_size * 0.5)
        source = voxel_down_sample(frame_downsample, config.mapping.voxel_size * 1.5)
        return source, frame_downsample

    def get_motion(self, curr_scan, curr_ts):
        scan = self.compensator.deskew_scan(curr_scan, curr_ts, self.last_delta)
        scan = self.preprocess(scan)
        source, scan_down = self.voxelize(scan, self.config)
        sigma = self.adaptive_threshold.get_threshold()
        initial_guess = self.last_pose @ self.last_delta
        new_pose = self.registration.align_points_to_map(
            points=source, voxel_map=self.local_map, initial_guess=initial_guess,
            max_correspondance_distance=3*sigma, kernel=sigma/3
        )
        model_dev = np.linalg.inv(initial_guess) @ new_pose
        self.adaptive_threshold.update_model_deviation(model_dev)
        self.local_map.remove_far_away_points(new_pose[:3, 3])
        self.local_map.update(scan_down, new_pose)
        self.last_delta = np.linalg.inv(self.last_pose) @ new_pose
        self.last_pose = new_pose
        T_ICP_Global = new_pose.astype(np.float64)
        T_ICP_Relative = self.last_delta.astype(np.float64)
        return T_ICP_Global, T_ICP_Relative


# ------------------------
# Worker for non-submap path
# ------------------------
def icp_and_overlap_worker(args):
    model, curr, prev, transformScan = args
    curr = remove_nan_inf(curr)
    prev = remove_nan_inf(prev)
    curr_trans = transformScan(curr)
    prev_trans = transformScan(prev)
    curr_down = curr_trans[::2]
    prev_down = prev_trans[::2]
    T_ICP_Relative = model.get_motion(
        source_raw=curr_trans.astype(np.float64),
        target_raw=prev_trans.astype(np.float64)
    )
    curr_homo = np.concatenate([curr_trans, np.ones((curr_trans.shape[0], 1), dtype=curr_trans.dtype)], axis=1)
    curr_in_prev = (T_ICP_Relative @ curr_homo.T).T[:, :3]
    overlap = calc_symmetric_overlap(prev_trans, curr_in_prev)  # B는 submap 경로에서만 바꿈
    return T_ICP_Relative, curr_trans, prev_trans, overlap, curr_down, prev_down


# ------------------------
# LOModule with submap fixes (A~E)
# ------------------------
class LOModule(nn.Module):
    def __init__(
        self,
        lo_model,
        T_I_L,
        R_I_L,
        init_state=None,
        device_id='cuda:0',
        use_submap=False,
        submap_voxel=0.5,
        submap_max_points=200_000
    ):
        super(LOModule, self).__init__()
        self.T_I_L = T_I_L
        self.R_I_L = R_I_L
        self.device_id = device_id
        self.scans0_np = None
        self.scans1_np = None

        # submap 옵션
        self.use_submap = bool(use_submap)
        self.submap_voxel = float(submap_voxel)
        self.submap_max_points = int(submap_max_points)
        self.submap_pts = None  # 항상 "현재(curr) 프레임" 좌표계로 유지 (A,E)

        # 모델
        if lo_model == 'g_icp':
            self.model = G_ICP()
        elif lo_model == 'fast_gicp':
            self.model = Fast_GICP()
        elif lo_model == 'small_gicp':
            self.model = Small_GICP()
        elif lo_model == 'kiss_icp':
            self.model = KISS_ICP(init_state)

    def reset_submap(self):
        self.submap_pts = None

    def transformScan(self, scan):
        return (self.R_I_L @ scan.T).T + self.T_I_L

    def forward(self, sample, last_state=None):
        scans0 = sample['scan0']
        scans1 = sample['scan1']
        B = len(scans0)

        # D: 시퀀스 경계에서 리셋
        if last_state is None or sample.get('is_new_seq', False) or sample.get('reset_submap', False):
            self.reset_submap()

        # 초기 글로벌
        if last_state is not None:
            if isinstance(last_state, pp.LieTensor):
                curr_glb_mat = last_state.matrix().cpu().numpy().astype(np.float64)
            elif isinstance(last_state, torch.Tensor) and last_state.shape == (7,):
                curr_glb_mat = pp.SE3(last_state.unsqueeze(0)).matrix().cpu().numpy().astype(np.float64)
            else:
                curr_glb_mat = np.eye(4, dtype=np.float64)
        else:
            curr_glb_mat = np.eye(4, dtype=np.float64)

        # ① small_gicp + use_submap: 순차 처리 (A,B,C,E)
        if isinstance(self.model, Small_GICP) and self.use_submap:
            motions_mat = [None] * B
            overlap_scores = np.zeros(B, dtype=np.float32)
            rel_mats, global_mats = [], [curr_glb_mat.copy()]
            self.scans0_np, self.scans1_np = [], []

            building_submap_this_batch = (self.submap_pts is None)
            submap_local = None  # 배치 내 누적(최초 생성용)

            for idx in range(B):
                curr = scans1[idx].cpu().numpy()
                prev = scans0[idx].cpu().numpy()
                curr = remove_nan_inf(curr)
                prev = remove_nan_inf(prev)
                curr_trans = self.transformScan(curr)
                prev_trans = self.transformScan(prev)

                # target: 서브맵 있으면 서브맵, 없으면 prev
                if (not building_submap_this_batch) and (self.submap_pts is not None) and self.submap_pts.size > 0:
                    target_pts = self.submap_pts
                else:
                    target_pts = prev_trans

                # 정합 (float64 유지, E)
                T_ICP_Relative = self.model.get_motion(
                    source_raw=curr_trans.astype(np.float64),
                    target_raw=target_pts.astype(np.float64)
                ).astype(np.float64)

                # 글로벌 누적 (float64, E)
                T_ICP_Global = curr_glb_mat @ T_ICP_Relative
                motions_mat[idx] = T_ICP_Relative
                rel_mats.append(T_ICP_Relative.copy())
                global_mats.append(T_ICP_Global.copy())
                curr_glb_mat = T_ICP_Global

                # B: overlap 기준을 타깃에 맞춤
                curr_in_target = _apply_T_left(curr_trans, T_ICP_Relative)
                overlap = calc_symmetric_overlap(target_pts, curr_in_target)
                overlap_scores[idx] = overlap

                # 시각화용 저장(다운샘플만)
                self.scans1_np.append(curr_trans[::2])
                self.scans0_np.append(target_pts[::2] if target_pts is self.submap_pts else prev_trans[::2])

                # --- 서브맵 갱신 (A,C,E): "신규 프레임만 voxel", FIFO, 랜덤 없음 ---
                T_rel_inv = np.linalg.inv(T_ICP_Relative)

                if building_submap_this_batch:
                    # 배치 종료 후 세팅을 위해 로컬 누적
                    prev_in_curr = _apply_T_left(prev_trans, T_rel_inv)
                    curr_ds = _voxel_downsample_np(curr_trans, self.submap_voxel)  # curr만 voxel
                    if submap_local is None:
                        submap_local = np.vstack([prev_in_curr, curr_ds])
                    else:
                        submap_local = np.vstack([_apply_T_left(submap_local, T_rel_inv), curr_ds])
                    # 용량 제한(FIFO)
                    if submap_local.shape[0] > self.submap_max_points:
                        submap_local = submap_local[-self.submap_max_points:]
                else:
                    # 이미 서브맵 존재: 이전 서브맵을 curr로 이동 후 curr_ds만 붙임
                    moved = _apply_T_left(self.submap_pts, T_rel_inv) if self.submap_pts is not None else np.empty((0, 3), dtype=curr_trans.dtype)
                    curr_ds = _voxel_downsample_np(curr_trans, self.submap_voxel)
                    merged = np.vstack([moved, curr_ds])
                    # FIFO 제한
                    if merged.shape[0] > self.submap_max_points:
                        merged = merged[-self.submap_max_points:]
                    self.submap_pts = merged  # 재-voxel 금지(그리드 흔들림 방지)

            # 배치 끝: 최초 서브맵 생성/설정 (C)
            if building_submap_this_batch and (submap_local is not None):
                if submap_local.shape[0] > self.submap_max_points:
                    submap_local = submap_local[-self.submap_max_points:]
                self.submap_pts = submap_local

        # ② 그 외: 기존 병렬 / KISS 경로
        else:
            if not isinstance(self.model, KISS_ICP):
                args_list = [
                    (self.model, scans1[i].cpu().numpy(), scans0[i].cpu().numpy(), self.transformScan)
                    for i in range(B)
                ]
                results = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for out in executor.map(icp_and_overlap_worker, args_list):
                        results.append(out)

                motions_mat = [None] * B
                overlap_scores = np.zeros(B, dtype=np.float32)
                rel_mats, global_mats = [], [curr_glb_mat.copy()]
                self.scans0_np, self.scans1_np = [], []

                for idx, (T_ICP_Relative, curr_trans, prev_trans, overlap, curr_down, prev_down) in enumerate(results):
                    T_ICP_Relative = T_ICP_Relative.astype(np.float64)  # E
                    T_ICP_Global = curr_glb_mat @ T_ICP_Relative
                    motions_mat[idx] = T_ICP_Relative
                    rel_mats.append(T_ICP_Relative.copy())
                    global_mats.append(T_ICP_Global.copy())
                    overlap_scores[idx] = overlap
                    curr_glb_mat = T_ICP_Global
                    self.scans1_np.append(curr_down)
                    self.scans0_np.append(prev_down)
            else:
                # KISS_ICP 경로(원형 유지, E만 캐스팅)
                self.scans0_np = []
                self.scans1_np = []
                motions_mat = [None] * B
                overlap_scores = np.zeros(B, dtype=np.float32)
                rel_mats = []
                global_mats = [curr_glb_mat.copy()]

                # last_state 반영
                if last_state is not None:
                    if isinstance(last_state, pp.LieTensor):
                        curr_glb_mat = last_state.matrix().cpu().numpy().astype(np.float64)
                    elif isinstance(last_state, torch.Tensor) and last_state.shape == (7,):
                        curr_glb_mat = pp.SE3(last_state.unsqueeze(0)).matrix().cpu().numpy().astype(np.float64)
                else:
                    curr_glb_mat = np.eye(4, dtype=np.float64)

                for idx in range(B):
                    curr = scans1[idx].cpu().numpy()
                    prev = scans0[idx].cpu().numpy()
                    curr = remove_nan_inf(curr)
                    prev = remove_nan_inf(prev)
                    curr_trans = self.transformScan(curr)
                    prev_trans = self.transformScan(prev)
                    curr_down = curr_trans[::2]
                    prev_down = prev_trans[::2]
                    T_ICP_Global, T_ICP_Relative = self.model.get_motion(
                        curr_trans, sample['scan1_ts'][idx].cpu().numpy()
                    )
                    T_ICP_Global = T_ICP_Global.astype(np.float64)
                    T_ICP_Relative = T_ICP_Relative.astype(np.float64)
                    motions_mat[idx] = T_ICP_Relative
                    rel_mats.append(T_ICP_Relative.copy())
                    global_mats.append(T_ICP_Global.copy())
                    curr_glb_mat = T_ICP_Global
                    curr_in_prev = _apply_T_left(curr_trans, T_ICP_Relative)
                    overlap = calc_symmetric_overlap(prev_trans, curr_in_prev)
                    overlap_scores[idx] = overlap
                    self.scans0_np.append(prev_down)
                    self.scans1_np.append(curr_down)

        # 공통 반환 (torch로 변환 시에만 float32)
        def mat_to_vec7(mat44):
            t = mat44[:3, 3]
            Rm = mat44[:3, :3]
            q = Rotation.from_matrix(Rm.copy()).as_quat()
            return np.concatenate((t, q), axis=0).astype(np.float32)

        se3_rel_vecs = [mat_to_vec7(m) for m in rel_mats]
        se3_glb_vecs = [mat_to_vec7(m) for m in global_mats]

        motions_se3_rel = pp.SE3(torch.from_numpy(np.stack(se3_rel_vecs)).to(self.device_id))
        motions_se3_glb = pp.SE3(torch.from_numpy(np.stack(se3_glb_vecs)).to(self.device_id))

        return motions_se3_glb, motions_se3_rel, overlap_scores
