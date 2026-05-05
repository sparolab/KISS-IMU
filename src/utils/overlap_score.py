import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import pygicp


def calc_symmetric_overlap(cloud1, cloud2, dis_threshold=0.3):
    tree2 = cKDTree(cloud2)
    dists1, _ = tree2.query(cloud1, k=1)
    match_num = np.sum(dists1 < dis_threshold)

    tree1 = cKDTree(cloud1)
    dists2, _ = tree1.query(cloud2, k=1)
    match_num += np.sum(dists2 < dis_threshold)

    overlap = match_num / (cloud1.shape[0] + cloud2.shape[0])
    return min(overlap, 1.0)

def apply_pose(cloud, pose7):
    if hasattr(pose7, 'cpu'):
        pose7 = pose7.detach().cpu().numpy()
    t = pose7[:3]
    q = pose7[3:]
    Rm = Rotation.from_quat(q).as_matrix()  # (3, 3)
    return (cloud @ Rm.T) + t  # (N, 3)

def overlap_one(args):
    source, target, motion, dis_threshold = args
    target_in_source = apply_pose(target, motion)
    return calc_symmetric_overlap(source, target_in_source, dis_threshold)

def batch_pose_aligned_overlap(source_clouds, target_clouds, motions, dis_threshold=0.3):
    def _to_numpy(x):
        if hasattr(x, 'cpu'):
            return x.detach().cpu().numpy()
        return x

    source_clouds = [_to_numpy(x) for x in source_clouds]
    target_clouds = [_to_numpy(x) for x in target_clouds]
    motions = [_to_numpy(x) for x in motions]

    args_list = [(source_clouds[i], target_clouds[i], motions[i], dis_threshold) for i in range(len(source_clouds))]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(overlap_one, args_list))
    return np.array(results, dtype=np.float32)

def overlap_one_with_ds(args):
    source, target, motion, dis_threshold, ds_resolution = args
    
    source_ds = pygicp.downsample(source.astype(np.float64), ds_resolution)
    target_ds = pygicp.downsample(target.astype(np.float64), ds_resolution)
    
    source_ds = source_ds[np.isfinite(source_ds).all(axis=1)]
    target_ds = target_ds[np.isfinite(target_ds).all(axis=1)]

    target_in_source = apply_pose(target_ds, motion)
    return calc_symmetric_overlap(source_ds, target_in_source, dis_threshold)


def batch_pose_aligned_overlap_with_ds(scan0, scan1, pgo_motion, dis_threshold=0.3, ds_resolution=2.0):
    def _to_numpy(x):
        if hasattr(x, 'cpu'):
            return x.detach().cpu().numpy()
        return x

    scan0 = [_to_numpy(x) for x in scan0]
    scan1 = [_to_numpy(x) for x in scan1]
    pgo_motion = [_to_numpy(x) for x in pgo_motion]

    args_list = [
        (scan0[i], scan1[i], pgo_motion[i], dis_threshold, ds_resolution)
        for i in range(len(scan0))
    ]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(overlap_one_with_ds, args_list))
    
    return np.array(results, dtype=np.float32)