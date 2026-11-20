"""Local maps the scans are registered against. Add one by subclassing LocalMap and
registering it with @register('name'); select it with lo.map.name in the PGO config."""

from .base import REGISTRY, LocalMap, build_map, register, voxel_downsample  # noqa: F401
from . import submap, voxel_hash  # noqa: F401  (register 'keyframe_submap', 'voxel_hash')
