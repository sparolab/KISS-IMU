import numpy as np

REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def build_map(cfg):
    cfg = dict(cfg)
    name = cfg.pop('name')
    if name not in REGISTRY:
        raise KeyError(f'unknown map {name!r}; known: {sorted(REGISTRY)}')
    return REGISTRY[name](**cfg)


def voxel_downsample(points, voxel):
    """First point per voxel, deterministic."""
    if voxel <= 0 or len(points) == 0:
        return points
    _, idx = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(idx)]


class LocalMap:
    """World-frame point map. `version` changes whenever points() would change, so the
    registration target is only rebuilt when needed."""
    version = 0

    def insert(self, points_world, pose):
        raise NotImplementedError

    def points(self, center=None):
        raise NotImplementedError

    @property
    def empty(self):
        raise NotImplementedError
