from scipy.ndimage import distance_transform_edt
from .voxel_utils import get_semantic_map, parse_proto_to_numpy_array
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.ndimage import zoom
import numpy as np

EPS = 1e-4
class VoxelMap:
    def __init__(self, map_path, image_size, r_sub=4):
        H, W = image_size
        self.H, self.W = H, W
        self.r_sub = r_sub

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

        # Softmax normalization over channels
        logits = self._probs
        exp_logits = np.exp(logits - np.max(logits, axis=2, keepdims=True))
        sum_exp = np.sum(exp_logits, axis=2, keepdims=True)
        softmax_probs = exp_logits / (sum_exp + EPS)

        # Fill missing pixels using nearest neighbor
        dist, (inds0, inds1) = distance_transform_edt(~self.mask, return_indices=True)
        probs_upsampled = zoom(softmax_probs, (self.r_sub, self.r_sub, 1), order=1)

        # Gaussian smoothing to reduce blockiness
        interp_probs = gaussian_filter(probs_upsampled, sigma=(1.0, 1.0, 0), mode='nearest')

        self._probs = interp_probs
        return self._probs

    def _nearest_fill(self, img):
        mask = img != 0
        if not np.any(mask):
            return img

        dist, (inds0, inds1) = distance_transform_edt(~mask, return_indices=True)
        return img[inds0, inds1]
