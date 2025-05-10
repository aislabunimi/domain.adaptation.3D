from .voxel_map import VoxelMap
from .ray_caster import RayCaster
from .visu3d import Visualizer3D
import time
class LabelGenerator:
    def __init__(self, mesh_path, map_serialized_path,k_color, image_shape, r_sub=1, visu3d=True):
        """
        Initialize the label generator with direct input data.
        
        Parameters:
        - mesh_path: path to the predicted mesh
        - map_serialized_path: path to serialized semantic map
        - k_color: 3x3 color camera intrinsics
        - image_shape: (H, W) delle immagini (coerente con rgb)
        - r_sub: voxel subsampling ratio
        - visu3d: whether to visualize rays in 3D
        """
        self.r_sub = r_sub
        self._voxel_map = VoxelMap(map_serialized_path, image_shape, r_sub)
        self._ray_caster = RayCaster(mesh_path, k_color, image_shape, r_sub, 10)

        self._visu_active = visu3d
        if visu3d:
            self._visu3d = Visualizer3D(image_shape, k_color, mesh_path)

    def get_label(self, H_cam):
        start_raycast = time.time()
        locations, index_ray, index_tri, ray_origins = self._ray_caster.raycast(H_cam)
        end_raycast = time.time()
        #print(f"[Raycast] Took {end_raycast - start_raycast:.4f} seconds")

        # Probability mapping timing
        start_probs = time.time()
        probs = self._voxel_map.ray_cast_results_to_probs(locations, index_ray)
        end_probs = time.time()
        #print(f"[Probs Mapping] Took {end_probs - start_probs:.4f} seconds")
        if self._visu_active:
            self._visu3d.visu(locations, ray_origins)

        return probs
