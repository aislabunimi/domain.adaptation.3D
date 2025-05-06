from scipy.ndimage import distance_transform_edt
from .voxel_utils import get_semantic_map, parse_proto_to_numpy_array
import numpy as np

EPS = 1e-4
class VoxelMap:
    def __init__(self, map_path, image_size, r_sub=4):
        H, W = image_size
        self.H, self.W = H, W
        self._r_sub = r_sub

        semantic_map = get_semantic_map(map_path)
        voxels_np, mi = parse_proto_to_numpy_array(semantic_map)

        self._voxels = voxels_np
        self._mi = mi
        self._voxel_size = semantic_map.semantic_blocks[0].voxel_size

        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        self._vo = v
        self._uo = u

        self._vr = v[::r_sub, ::r_sub].ravel()
        self._ur = u[::r_sub, ::r_sub].ravel()

        self._probs = np.zeros((H, W, self._voxels.shape[3]), dtype=np.float32)

        self.mask = np.zeros((H, W), dtype=bool)
        self.mask[::r_sub, ::r_sub] = True
        self.points = np.stack(np.where(self.mask), axis=1)

    def ray_cast_results_to_probs(self, locations, index_ray):
      self._probs.fill(0)

      if locations.shape[0] == 0:
          return self._probs

      locations = np.asarray(locations, dtype=np.float32)
      index_ray = np.asarray(index_ray, dtype=np.int32)

      idx_tmp = np.floor((locations - self._mi + EPS) / self._voxel_size).astype(np.int32)
      for i in range(3):
          idx_tmp[:, i] = np.clip(idx_tmp[:, i], 0, self._voxels.shape[i] - 1)

      probs_selected = self._voxels[idx_tmp[:, 0], idx_tmp[:, 1], idx_tmp[:, 2]]

      v_idx = self._vr[index_ray]
      u_idx = self._ur[index_ray]
      self._probs[v_idx, u_idx] = probs_selected

      min_vals = np.min(self._probs, axis=2, keepdims=True)
      probs_shifted = self._probs - min_vals
      probs_sum = np.sum(probs_shifted, axis=2, keepdims=True)

      valid = probs_sum.squeeze(-1) > EPS
      probs_final = np.zeros_like(self._probs)
      probs_final[valid] = probs_shifted[valid] / probs_sum[valid]
      probs_final[~valid, 0] = 1

      # ✅ Optimize NN fill: single call
      dist, (inds0, inds1) = distance_transform_edt(~self.mask, return_indices=True)
      interp_probs = probs_final[inds0, inds1]

      self._probs = interp_probs
      return self._probs

      def _nearest_fill(self, img):
        mask = img != 0
        if not np.any(mask):
            return img

        dist, (inds0, inds1) = distance_transform_edt(~mask, return_indices=True)
        return img[inds0, inds1]
