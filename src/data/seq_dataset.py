import os
import numpy as np
import pandas as pd
import pypose as pp
import open3d as o3d

import torch
import torch.utils.data as Data

from scipy.spatial.transform import Rotation as R, Slerp

from data.collate import collate_fn


class SeqDataset(Data.Dataset):
    def __init__(self, data_root, data_seq=None, data_type='mulran', batch_shift=1, method='ours', train_ratio=100):
        self.data_root = data_root
        self.data_seq = data_seq
        self.data_dir = os.path.join(self.data_root, self.data_seq)        
        self.data_type = data_type
        self.batch_shift = batch_shift
        self.method = method
        self.train_ratio = train_ratio
        self.gravity = torch.tensor([0., 0., 9.81], dtype=torch.float32)
        if self.data_type == 'mulran':
            self.lidar_dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
            self.acc_idx = 11; self.gyr_idx = 8
            self.T_I_L = np.array([1.77, -0.00, -0.05])
            self.R_I_L = np.array([[-1,  0,  0],[ 0, -1,  0],[ 0,  0,  1]])
            self.T_I_G = np.array([6.57695566e-02, 1.06635747e-02, -1.75469073e+00])
            self.R_I_G = np.array([[ 9.99982948e-01, -5.83983848e-03,  5.23598776e-06],
                                   [ 5.83983849e-03,  9.99982948e-01, -1.74532925e-06],
                                   [-5.22570603e-06,  1.77587681e-06,  1.00000000e+00]])
            self.imu_window_size = 15
            
        elif self.data_type == 'yeoncheon':
            self.lidar_dtype = [('x', np.float64), ('y', np.float64), ('z', np.float64), ('intensity', np.float64)]
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_L = np.zeros(3)
            self.R_I_L = np.eye(3)
            self.T_I_G = np.zeros(3)
            self.R_I_G = np.eye(3)
            self.imu_window_size = 15
            
        elif self.data_type == 'kitti':
            self.lidar_dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
            self.acc_idx = 15; self.gyr_idx = 21
            self.T_I_L = np.array([-8.086759e-01,3.195559e-01,-7.997231e-01])
            self.R_I_L = np.array([[9.999976e-01,7.553071e-04,-2.035826e-03],
                                   [-7.854027e-04,9.998898e-01,-1.482298e-02],
                                   [2.024406e-03,1.482454e-02,9.998881e-01]])
            self.T_I_G = np.zeros(3)
            self.R_I_G = np.array([[  0,  0,  1 ],
                                   [ -1,  0,  0 ],
                                   [  0, -1,  0]])
            self.imu_window_size = 15
            
        elif self.data_type == 'diter++':
            self.lidar_dtype = [('x', np.float64), ('y', np.float64), ('z', np.float64), ('intensity', np.float64)]
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_L = np.zeros(3)
            self.R_I_L = np.array([[-1, 0, 0],
                                   [ 0,-1, 0],
                                   [ 0, 0, 1]])
            self.T_I_G = np.zeros(3)
            self.R_I_G = np.array([[-1, 0, 0],
                                   [ 0,-1, 0],
                                   [ 0, 0, 1]])
            self.imu_window_size = 55
            
        elif self.data_type == 'diter_os':
            self.lidar_dtype = [('x', np.float64),('y',np.float64),('z',np.float64),('intensity',np.float64)]
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_L = np.zeros(3)
            self.R_I_L = np.eye(3)
            self.T_I_G = np.zeros(3)
            self.R_I_G = np.eye(3)
            self.imu_window_size = 15
            
        elif self.data_type == 'tailrobot':
            self.lidar_dtype = [('x', np.float64),('y',np.float64),('z',np.float64),('intensity',np.float64)]
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_L = np.zeros(3)
            self.R_I_L = np.eye(3)
            self.T_I_G = np.zeros(3)
            self.R_I_G = np.eye(3)
            self.imu_window_size = 15
            
        elif self.data_type == 'kimera':
            self.lidar_dtype = [('x', np.float64),('y',np.float64),('z',np.float64),('intensity',np.float64)]
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_L = np.zeros(3)
            self.R_I_L = np.array([[ 0, 1, 0],
                                   [ 0, 0,-1],
                                   [-1, 0, 0]])
            self.T_I_G = np.zeros(3)
            self.R_I_G = np.eye(3)
            self.gravity = torch.tensor([0., -9.81, 0.], dtype=torch.float32)
            self.imu_window_size = 35
        else:
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_L = np.zeros(3);
            self.R_I_L = np.eye(3)
            self.T_I_G = np.zeros(3);
            self.R_I_G = np.eye(3)
            self.imu_window_size = 15
        
        self.imu_ts, self.imu_dts, self.accels, self.gyros = self.load_imu(self.data_dir)
        self.scan_files, self.scan_ts = self.load_scan(self.data_dir)
        self.gt_ts, self.gt_poses = self.load_gt(self.data_dir)
        
        self.align_start_time()
        
        # Set the first GT pose translation to [0,0,0] by subtracting the first translation
        if len(self.gt_poses) > 0:
            self.gt_poses[:, 0, :3] -= self.gt_poses[0, 0, :3].copy()
        
        gt_vels = np.diff(self.gt_poses[:, 0, 0:3], axis=0)
        q_xyzw_t = torch.tensor(self.gt_poses[0, 0, 3:7], dtype=torch.float32).unsqueeze(0)
        R_so3    = pp.SO3(q_xyzw_t)
        R_mat_t  = R_so3.matrix().squeeze(0)
        init_rot = self.R_I_G.T@ R_mat_t.detach().numpy()
        init_quat = R.from_matrix(init_rot).as_quat()
        
        self.init = {
            'rot': torch.tensor(init_quat, dtype=torch.float32),
            'pos': torch.zeros(3, dtype=torch.float32),
            'vel': torch.tensor(gt_vels[0, :], dtype=torch.float32)
            }
        
        self.links = []
        for i in range(len(self.scan_files) - 1):
            self.links.append([i, i + 1])
    
    def load_imu(self, data_root):
        if self.method == 'ours':
            data_path = os.path.join(data_root, 'imu.csv')
        else:
            data_path = os.path.join(data_root, self.method, f'imu_{self.train_ratio}.csv')
        imu_data = pd.read_csv(data_path, header=None)
        imu_ts = np.array(imu_data.iloc[:, 0].values).astype(np.float64)
        imu_dts = np.diff(imu_ts).astype(np.float64)
        accels = np.array(imu_data.iloc[:, self.acc_idx:self.acc_idx + 3].values)
        gyros = np.array(imu_data.iloc[:, self.gyr_idx:self.gyr_idx + 3].values)        
        return imu_ts, imu_dts, accels, gyros
    
    def load_scan(self, data_root):
        data_path = os.path.join(data_root, 'points', 'data')
        time_path = os.path.join(data_root, 'points', 'timestamps.txt')

        if not (os.path.isdir(data_path) and os.path.isfile(time_path)):
            raise FileNotFoundError(
                f"expected 'points/data/' and 'points/timestamps.txt' under '{data_root}'"
            )

        scan_files = sorted(
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith('.bin')
        )
        scan_ts = np.loadtxt(time_path, ndmin=1).astype(np.float64)

        if len(scan_files) != len(scan_ts):
            n = min(len(scan_files), len(scan_ts))
            scan_files = scan_files[:n]
            scan_ts = scan_ts[:n]

        return scan_files, scan_ts
    
    def load_gt(self, data_root):
        data_path = os.path.join(data_root, 'gt_pose.csv')
        gt_data = pd.read_csv(data_path, header=None)

        gt_ts = gt_data.iloc[:, 0].to_numpy(dtype=np.float64)
        pose_vals = gt_data.iloc[:, 1:].to_numpy(dtype=np.float64)
        num_poses, D = pose_vals.shape

        gt_poses = np.zeros((num_poses, 1, 7), dtype=np.float64)

        if D == 7:
            # [x, y, z, qx, qy, qz, qw]
            pos_all  = pose_vals[:, 0:3]                # (N,3)
            quat_all = pose_vals[:, 3:7]                # (N,4)
            first_trans = pos_all[0].copy()

            for i in range(num_poses):
                trans = pos_all[i] - first_trans
                orig_rot_mat = R.from_quat(quat_all[i]).as_matrix()

                # Apply coordinate frame rotation (preserving original code logic)
                rotated_trans   = self.R_I_G @ trans
                rotated_rot_mat = self.R_I_G @ orig_rot_mat
                rotated_quat    = R.from_matrix(rotated_rot_mat).as_quat()  # [qx,qy,qz,qw]

                gt_poses[i, 0, :] = np.hstack((rotated_trans, rotated_quat))

        elif D == 12:
            # [r11, r12, r13, x, r21, r22, r23, y, r31, r32, r33, z]
            R_cols = [0, 1, 2, 4, 5, 6, 8, 9, 10]
            T_cols = [3, 7, 11]
            R_all = pose_vals[:, R_cols].reshape(num_poses, 3, 3)  # (N,3,3)
            pos_all = pose_vals[:, T_cols]                         # (N,3)

            first_trans = pos_all[0].copy()

            for i in range(num_poses):
                trans = pos_all[i] - first_trans
                orig_rot_mat = R_all[i]

                rotated_trans   = self.R_I_G @ trans
                rotated_rot_mat = self.R_I_G @ orig_rot_mat
                rotated_quat    = R.from_matrix(rotated_rot_mat).as_quat()  # [qx,qy,qz,qw]

                gt_poses[i, 0, :] = np.hstack((rotated_trans, rotated_quat))
        else:
            raise ValueError(f"Unsupported number of pose columns: {D}. (Expected 7 or 12)")

        return gt_ts, gt_poses

    def align_start_time(self):
        imu_start = self.imu_ts[0]
        gt_start = self.gt_ts[0]
        scan_start = self.scan_ts[0]

        latest_start = max(imu_start, gt_start, scan_start)

        imu_start_idx = np.searchsorted(self.imu_ts, latest_start, side='left')
        gt_start_idx  = np.searchsorted(self.gt_ts, latest_start, side='left')
        scan_start_idx = np.searchsorted(self.scan_ts, latest_start, side='left')

        self.scan_files = self.scan_files[scan_start_idx:]
        self.scan_ts = self.scan_ts[scan_start_idx:]

        self.imu_ts = self.imu_ts[imu_start_idx:]
        
        self.imu_dts = np.diff(self.imu_ts).astype(np.float64)
        
        self.accels = self.accels[imu_start_idx:]
        self.gyros = self.gyros[imu_start_idx:]
        self.gt_ts = self.gt_ts[gt_start_idx:]
        self.gt_poses = self.gt_poses[gt_start_idx:]
        
        filtered_scan_files = []
        filtered_scan_ts = []
        for sf, st in zip(self.scan_files, self.scan_ts):
            num_imu_before = np.sum(self.imu_ts < st)
            if num_imu_before >= self.imu_window_size:
                filtered_scan_files.append(sf)
                filtered_scan_ts.append(st)
        filtered_scan_files = np.array(filtered_scan_files)
        filtered_scan_ts = np.array(filtered_scan_ts)
        
        self.scan_files = filtered_scan_files
        self.scan_ts = filtered_scan_ts
        
    def scan_vstack(self, scan):
        return np.vstack((scan['x'], scan['y'], scan['z'])).T
    
    def __getitem__(self, index):
        result = self.get_pair(self.links[index][0], self.links[index][1])
        if result is None:
            raise ValueError(f"get_pair returned None for index {index}. Check if imu_window_size={self.imu_window_size} is too large.")
        return result
    
    def __len__(self):
        return len(self.links)
    
    def get_pair(self, i, j):
        res = {}
        t0 = self.scan_ts[i]
        t1 = self.scan_ts[j]

        scan0 = np.fromfile(self.scan_files[i], dtype=self.lidar_dtype)
        scan0 = self.scan_vstack(scan0)
        scan1 = np.fromfile(self.scan_files[j], dtype=self.lidar_dtype)
        scan1 = self.scan_vstack(scan1)

        idx_in_window = np.where((self.imu_ts >= t0) & (self.imu_ts < t1))[0]
        num_imu_in_window = len(idx_in_window)

        imu_window_size = self.imu_window_size

        accels_real = self.accels[idx_in_window]
        gyros_real = self.gyros[idx_in_window]
        imu_ts_real = self.imu_ts[idx_in_window]
        imu_dts_real = self.imu_dts[idx_in_window]

        def pad_or_truncate(arr, size):
            if arr.shape[0] == size:
                return arr
            elif arr.shape[0] > size:
                return arr[:size]
            else:
                pad_shape = (size - arr.shape[0],) + arr.shape[1:]
                pad = np.zeros(pad_shape, dtype=arr.dtype)
                return np.concatenate([arr, pad], axis=0)
        
        accels_fixed = pad_or_truncate(accels_real, imu_window_size)
        gyros_fixed = pad_or_truncate(gyros_real, imu_window_size)
        imu_ts_fixed = pad_or_truncate(imu_ts_real.reshape(-1, 1), imu_window_size).reshape(-1)
        imu_dts_fixed = pad_or_truncate(imu_dts_real, imu_window_size)

        valid_length = min(num_imu_in_window, imu_window_size)

        gt_pose0 = self.interpolate_gt_pose(t0)
        gt_pose1 = self.interpolate_gt_pose(t1)

        dt = t1 - t0
        gt_velocity = (gt_pose1[:3] - gt_pose0[:3]) / dt

        res['scan0_ts'] = t0
        res['scan1_ts'] = t1
        res['scan0'] = scan0
        res['scan1'] = scan1
        res['imu_ts'] = imu_ts_fixed
        res['imu_dts'] = imu_dts_fixed
        res['accels'] = accels_fixed
        res['gyros'] = gyros_fixed
        res['valid_length'] = valid_length
        res['gt_pose0'] = gt_pose0
        res['gt_pose1'] = gt_pose1
        res['gt_velocity'] = gt_velocity
        return res
    
    def interpolate_gt_pose(self, target_time):
        time_diff = np.abs(self.gt_ts - target_time)
        nearest_idx = np.argmin(time_diff)
        
        if target_time <= self.gt_ts[0]:
            return self.gt_poses[0, 0, :]
        elif target_time >= self.gt_ts[-1]:
            return self.gt_poses[-1, 0, :]
        
        if target_time >= self.gt_ts[nearest_idx]:
            if nearest_idx + 1 < len(self.gt_ts):
                idx1, idx2 = nearest_idx, nearest_idx + 1
            else:
                return self.gt_poses[nearest_idx, 0, :]
        else:
            if nearest_idx > 0:
                idx1, idx2 = nearest_idx - 1, nearest_idx
            else:
                return self.gt_poses[nearest_idx, 0, :]
        
        pose1 = self.gt_poses[idx1, 0, :]
        pose2 = self.gt_poses[idx2, 0, :]
        
        t1, t2 = self.gt_ts[idx1], self.gt_ts[idx2]
        alpha = (target_time - t1) / (t2 - t1)
        
        trans1, trans2 = pose1[:3], pose2[:3]
        trans_interp = trans1 + alpha * (trans2 - trans1)
        
        quat1, quat2 = pose1[3:], pose2[3:]
        key_times = [0, 1]
        key_rots = R.from_quat([quat1, quat2])
        slerp = Slerp(key_times, key_rots)
        rot_interp = slerp(alpha)
        quat_interp = rot_interp.as_quat()
        
        gt_pose_interp = np.concatenate([trans_interp, quat_interp])
        return gt_pose_interp


def create_shifted_loader(dataset, batch_size, batch_shift, shuffle=False, num_workers=0):
    """
    Create a DataLoader with batch shift applied
    """
    class ShiftedSampler(Data.Sampler):
        def __init__(self, dataset, batch_size, shift_size=1):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shift_size = shift_size
            
            # Compute total number of batches (accounting for shift)
            # If shift_size is 1, generates nearly every possible batch
            # If shift_size equals batch_size, behaves like the original
            if shift_size >= batch_size:
                self.total_batches = len(dataset) // batch_size
            else:
                # Smaller shift_size yields more batches
                self.total_batches = (len(dataset) - batch_size) // shift_size + 1
        
        def __iter__(self):
            for batch_idx in range(self.total_batches):
                start_idx = batch_idx * self.shift_size
                end_idx = start_idx + self.batch_size
                
                if end_idx <= len(self.dataset):
                    yield list(range(start_idx, end_idx))
        
        def __len__(self):
            return self.total_batches
    
    sampler = ShiftedSampler(dataset, batch_size, batch_shift)
    
    return Data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
