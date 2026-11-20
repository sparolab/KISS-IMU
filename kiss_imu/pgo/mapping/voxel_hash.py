import numpy as np

from .base import LocalMap, register


@register('voxel_hash')
class VoxelHashMap(LocalMap):
    """KISS-ICP style: up to `max_points_per_voxel` points per voxel, voxels farther than
    `max_distance` from the latest inserted pose are dropped."""

    def __init__(self, voxel=1.0, max_points_per_voxel=20, max_distance=100.0):
        self.voxel, self.nmax, self.max_distance = voxel, max_points_per_voxel, max_distance
        self.vox = {}
        self.version = 0
        self._cache = None

    def insert(self, points_world, pose):
        keys = np.floor(points_world / self.voxel).astype(np.int64)
        for k, p in zip(map(tuple, keys), points_world):
            v = self.vox.get(k)
            if v is None:
                self.vox[k] = [p]
            elif len(v) < self.nmax:
                v.append(p)
        c = pose[:3, 3]
        far = [k for k, v in self.vox.items() if np.linalg.norm(v[0] - c) > self.max_distance]
        for k in far:
            del self.vox[k]
        self.version += 1
        self._cache = None

    def points(self, center=None):
        if self._cache is None:
            self._cache = np.array([p for v in self.vox.values() for p in v]) if self.vox else np.zeros((0, 3))
        return self._cache

    @property
    def empty(self):
        return not self.vox
