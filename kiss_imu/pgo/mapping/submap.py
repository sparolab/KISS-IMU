from collections import deque

import numpy as np

from .base import LocalMap, register, voxel_downsample


@register('keyframe_submap')
class KeyframeSubmap(LocalMap):
    """The last `max_keyframes` keyframe scans at their poses, merged and voxelised."""

    def __init__(self, voxel=0.5, max_keyframes=20, max_points=300_000):
        self.voxel, self.max_points = voxel, max_points
        self.frames = deque(maxlen=max_keyframes)
        self.version = 0
        self._cache = None

    def insert(self, points_world, pose):
        self.frames.append(voxel_downsample(points_world, self.voxel))
        self.version += 1
        self._cache = None

    def points(self, center=None):
        if self._cache is None:
            pts = voxel_downsample(np.concatenate(list(self.frames)), self.voxel) if self.frames else np.zeros((0, 3))
            self._cache = pts[-self.max_points:]
        return self._cache

    @property
    def empty(self):
        return not self.frames
