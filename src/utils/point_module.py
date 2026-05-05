import numpy as np
import open3d as o3d


class PointModule:
    def __init__(self, min_distnace, max_distnace, voxel_size, lidar_dtype):
        self.min_distnace = min_distnace
        self.max_distnace = max_distnace
        self.voxel_size   = voxel_size
        self.lidar_dtype  = lidar_dtype

    def readScan(self, bin_path):
        scan        = np.fromfile(bin_path, dtype=self.lidar_dtype)
        points      = np.stack((scan['x'], scan['y'], scan['z']), axis = -1)
        return points

    def remove_closest_points(self, points):
        dists = np.sum(np.square(points[:, :3]), axis=1)
        cloud_out = points[dists > self.min_distnace*self.min_distnace]
        return cloud_out

    def remove_far_points(self, points):
        dists = np.sum(np.square(points[:, :3]), axis=1)
        cloud_out = points[dists < self.max_distnace*self.max_distnace]
        return cloud_out
    
    def down_sampling(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
        down_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        orig_points_np = np.asarray(pcd.points)
        down_points_np = np.asarray(down_pcd.points)
        return down_points_np