import numpy as np
import small_gicp

from .base import Registration, RegistrationResult, register


@register('small_gicp')
class SmallGICP(Registration):
    """small_gicp (GICP by default; registration_type 'ICP' / 'PLANE_ICP' / 'GICP')."""

    def __init__(self, source_voxel=0.5, target_voxel=0.5, num_neighbors=10, max_correspondence_distance=1.0,
                 registration_type='GICP', max_iterations=20, num_threads=4):
        self.source_voxel, self.target_voxel = source_voxel, target_voxel
        self.num_neighbors, self.max_corr = num_neighbors, max_correspondence_distance
        self.type, self.max_iter, self.threads = registration_type, max_iterations, num_threads

    def _prep(self, points, voxel):
        return small_gicp.preprocess_points(np.ascontiguousarray(points, dtype=np.float64), downsampling_resolution=voxel,
                                            num_neighbors=self.num_neighbors, num_threads=self.threads)

    def prepare_source(self, points):
        return self._prep(points, self.source_voxel)

    def prepare_target(self, points):
        return self._prep(points, self.target_voxel)

    def align(self, source, target, T_init):
        (src, _), (tgt, tree) = source, target
        res = small_gicp.align(tgt, src, tree, init_T_target_source=np.asarray(T_init, dtype=np.float64),
                               registration_type=self.type, max_correspondence_distance=self.max_corr,
                               num_threads=self.threads, max_iterations=self.max_iter)
        return RegistrationResult(T=np.array(res.T_target_source), converged=bool(res.converged), H=np.array(res.H),
                                  num_inliers=int(res.num_inliers), num_source=int(src.size()), error=float(res.error),
                                  iterations=int(res.iterations))
