import os
import numpy as np
import pandas as pd
import pypose as pp
import open3d as o3d

import torch
import torch.utils.data as Data

from scipy.spatial.transform import Rotation as R, Slerp


def collate_fn(batch):
    if not batch:
        return None
    
    collated = {}
    
    collated['imu_ts'] = torch.stack([torch.from_numpy(item['imu_ts']).float() for item in batch])
    collated['imu_dts'] = torch.stack([torch.from_numpy(item['imu_dts']).float() for item in batch])
    collated['accels'] = torch.stack([torch.from_numpy(item['accels']).float() for item in batch])
    collated['gyros'] = torch.stack([torch.from_numpy(item['gyros']).float() for item in batch])
    collated['valid_length'] = torch.tensor([item['valid_length'] for item in batch], dtype=torch.long)
    
    collated['gt_pose0'] = torch.stack([torch.from_numpy(item['gt_pose0']).float() for item in batch])
    collated['gt_pose1'] = torch.stack([torch.from_numpy(item['gt_pose1']).float() for item in batch])
    collated['gt_velocity'] = torch.stack([torch.from_numpy(item['gt_velocity']).float() for item in batch])
    
    return collated


class SeqDataset(Data.Dataset):
    def __init__(self, data_root, data_seq=None, data_type='mulran', window_size=500, eval_method='raw', train_ratio=100):
        self.data_root = data_root
        self.data_seq = data_seq
        self.data_dir = os.path.join(self.data_root, self.data_seq)        
        self.data_type = data_type
        self.window_size = window_size 
        self.eval_method = eval_method
        self.train_ratio = train_ratio
        self.gravity = torch.tensor([0., 0., 9.81], dtype=torch.float32)
        if self.data_type == 'mulran':
            self.lidar_dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
            self.acc_idx = 11; self.gyr_idx = 8
            self.T_I_G = np.array([6.57695566e-02, 1.06635747e-02, -1.75469073e+00])
            self.R_I_G = np.array([[ 9.99982948e-01, -5.83983848e-03,  5.23598776e-06],
                                   [ 5.83983849e-03,  9.99982948e-01, -1.74532925e-06],
                                   [-5.22570603e-06,  1.77587681e-06,  1.00000000e+00]])
            
        elif self.data_type == 'yeoncheon':
            self.lidar_dtype = [('x', np.float64), ('y', np.float64), ('z', np.float64), ('intensity', np.float64)]
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_G = np.zeros(3)
            self.R_I_G = np.eye(3)
            
        elif self.data_type == 'kitti':
            self.lidar_dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
            self.acc_idx = 15; self.gyr_idx = 21
            self.T_I_G = np.zeros(3)
            self.R_I_G = np.array([[  0,  0,  1 ],
                                   [ -1,  0,  0 ],
                                   [  0, -1,  0]])
            
        elif self.data_type == 'diter++':
            self.lidar_dtype = [('x', np.float64), ('y', np.float64), ('z', np.float64), ('intensity', np.float64)]
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_G = np.zeros(3)
            # self.R_I_G = np.array([[ 0,-1, 0],
            #                        [-1, 0, 0],
            #                        [ 0, 0, 1]])
            self.R_I_G = np.array([[-1, 0, 0],
                                   [ 0,-1, 0],
                                   [ 0, 0, 1]])
            
        elif self.data_type == 'diter_os':
            self.lidar_dtype = [('x', np.float64),('y',np.float64),('z',np.float64),('intensity',np.float64)]
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_G = np.zeros(3)
            self.R_I_G = np.eye(3)
            
        elif self.data_type == 'tailrobot':
            self.lidar_dtype = [('x', np.float64),('y',np.float64),('z',np.float64),('intensity',np.float64)]
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_G = np.zeros(3)
            self.R_I_G = np.eye(3)
            
        elif self.data_type == 'kimera':
            self.lidar_dtype = [('x', np.float64),('y',np.float64),('z',np.float64),('intensity',np.float64)]
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_G = np.zeros(3)
            self.R_I_G = np.eye(3)
            self.gravity = torch.tensor([0., -9.81, 0.], dtype=torch.float32)
            
        elif self.data_type == 'botanic_velodyne':
            self.lidar_dtype = [('x', np.float64),('y',np.float64),('z',np.float64),('intensity',np.float64)]
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_L = np.zeros(3)
            self.R_I_L = np.eye(3)
            self.T_I_G = np.zeros(3)
            self.R_I_G = np.eye(3)
            self.imu_window_size = 55
            
        else:
            self.lidar_dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
            self.acc_idx = 4; self.gyr_idx = 1
            self.T_I_G = np.zeros(3);
            self.R_I_G = np.eye(3)
        
        self.imu_ts, self.imu_dts, self.accels, self.gyros = self.load_imu(self.data_dir)
        self.gt_ts, self.gt_poses = self.load_gt(self.data_dir)
        
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
        
        self.links = self._create_time_aligned_links()
    
    def _create_time_aligned_links(self):

        links = []
        
        imu_start_time = self.imu_ts[0]
        imu_end_time = self.imu_ts[-1]
        gt_start_time = self.gt_ts[0]
        gt_end_time = self.gt_ts[-1]
        
        common_start_time = max(imu_start_time, gt_start_time)
        common_end_time = min(imu_end_time, gt_end_time)
        
        if common_start_time >= common_end_time:
            print("⚠️ Warning: GT and IMU data time ranges do not overlap!")
            return links

        imu_start_idx = np.searchsorted(self.imu_ts, common_start_time, side='left')
        imu_end_idx = np.searchsorted(self.imu_ts, common_end_time, side='right')
        
        for i in range(imu_start_idx, imu_end_idx - self.window_size, self.window_size):
            links.append([i, i + self.window_size])
        
        return links
    
    def load_imu(self, data_root):
        if self.eval_method == 'raw' or self.eval_method == 'ours':
            data_path = os.path.join(data_root, 'imu.csv')
        else:
            data_path = os.path.join(data_root, self.eval_method, f'imu_{self.train_ratio}.csv')

        imu_data = pd.read_csv(data_path, header=None)
        imu_ts = np.array(imu_data.iloc[:, 0].values).astype(np.float64)
        imu_dts = np.diff(imu_ts).astype(np.float64)
        accels = np.array(imu_data.iloc[:, self.acc_idx:self.acc_idx + 3].values)
        gyros = np.array(imu_data.iloc[:, self.gyr_idx:self.gyr_idx + 3].values)        
        return imu_ts, imu_dts, accels, gyros

    
    def load_gt(self, data_root):
        data_path = os.path.join(data_root, 'gt_pose.csv')
        gt_data = pd.read_csv(data_path, header=None)

        gt_ts = gt_data.iloc[:, 0].to_numpy(dtype=np.float64)
        pose_vals = gt_data.iloc[:, 1:].to_numpy(dtype=np.float64)
        
        unique_indices = []
        seen_times = set()
        for i, t in enumerate(gt_ts):
            if t not in seen_times:
                unique_indices.append(i)
                seen_times.add(t)
        
        gt_ts = gt_ts[unique_indices]
        pose_vals = pose_vals[unique_indices]
        
        sort_indices = np.argsort(gt_ts)
        gt_ts = gt_ts[sort_indices]
        pose_vals = pose_vals[sort_indices]
        
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

    
    def __getitem__(self, index):
        result = self.get_imu_gt_pair(self.links[index][0], self.links[index][1])
        if result is None:
            raise ValueError(f"get_imu_gt_pair returned None for index {index}. Check if window_size={self.window_size} is too large.")
        return result
    
    def __len__(self):
        return len(self.links)
    
    def get_imu_gt_pair(self, start_idx, end_idx):
        res = {}
        
        imu_window_indices = np.arange(start_idx, end_idx)
        
        valid_indices = imu_window_indices[imu_window_indices < len(self.imu_ts)]
        
        if len(valid_indices) == 0:
            return None
            
        accels_window = self.accels[valid_indices]
        gyros_window = self.gyros[valid_indices]
        imu_ts_window = self.imu_ts[valid_indices]
        imu_dts_window = self.imu_dts[valid_indices]
        
        def pad_or_truncate(arr, size):
            if arr.shape[0] == size:
                return arr
            elif arr.shape[0] > size:
                return arr[:size]
            else:
                pad_shape = (size - arr.shape[0],) + arr.shape[1:]
                pad = np.zeros(pad_shape, dtype=arr.dtype)
                return np.concatenate([arr, pad], axis=0)
        
        accels_fixed = pad_or_truncate(accels_window, self.window_size)
        gyros_fixed = pad_or_truncate(gyros_window, self.window_size)
        imu_ts_fixed = pad_or_truncate(imu_ts_window.reshape(-1, 1), self.window_size).reshape(-1)
        imu_dts_fixed = pad_or_truncate(imu_dts_window, self.window_size)
        
        valid_length = len(valid_indices)
        
        t_start = imu_ts_window[0]
        t_end = imu_ts_window[-1]
        
        gt_pose_start = self.interpolate_gt_pose(t_start)
        gt_pose_end = self.interpolate_gt_pose(t_end)
        
        dt = t_end - t_start
        gt_velocity = (gt_pose_end[:3] - gt_pose_start[:3]) / dt if dt > 0 else np.zeros(3)

        res['imu_ts'] = imu_ts_fixed
        res['imu_dts'] = imu_dts_fixed
        res['accels'] = accels_fixed
        res['gyros'] = gyros_fixed
        res['valid_length'] = valid_length
        res['gt_pose0'] = gt_pose_start
        res['gt_pose1'] = gt_pose_end
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
        
        if abs(t2 - t1) < 1e-9:  
            return pose1
        
        alpha = (target_time - t1) / (t2 - t1)
        
        alpha = np.clip(alpha, 0.0, 1.0)
        
        trans1, trans2 = pose1[:3], pose2[:3]
        trans_interp = trans1 + alpha * (trans2 - trans1)
        
        quat1, quat2 = pose1[3:], pose2[3:]
        
        quat1 = quat1 / np.linalg.norm(quat1)
        quat2 = quat2 / np.linalg.norm(quat2)
        
        dot_product = np.dot(quat1, quat2)
        if abs(dot_product) > 0.9999:
            quat_interp = quat1
        else:
            key_times = [0, 1]
            key_rots = R.from_quat([quat1, quat2])
            slerp = Slerp(key_times, key_rots)
            rot_interp = slerp(alpha)
            quat_interp = rot_interp.as_quat()
        
        gt_pose_interp = np.concatenate([trans_interp, quat_interp])
        return gt_pose_interp

