import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from .helper import get_rays, transform_points
import open3d as o3d
from tqdm import tqdm
import rospy

def laplacian_smoothing(mesh, iterations=10, lambda_smooth=0.5):
    import numpy as np
    from tqdm import tqdm

    vertices = mesh.vertices.copy()
    faces = mesh.faces

    # Build adjacency map (as list of numpy arrays for speed)
    vertex_neighbors = [set() for _ in range(len(vertices))]
    for face in faces:
        for i in range(3):
            vi = face[i]
            vj = face[(i + 1) % 3]
            vk = face[(i + 2) % 3]
            vertex_neighbors[vi].update([vj, vk])

    # Convert to list of numpy arrays for fast access
    vertex_neighbors = [np.array(list(neigh)) for neigh in vertex_neighbors]

    for _ in tqdm(range(iterations), desc="Smoothing", unit="iteration"):
        new_vertices = vertices.copy()
        for i, neighbors in enumerate(vertex_neighbors):
            if len(neighbors) == 0:
                continue
            avg_pos = np.mean(vertices[neighbors], axis=0)
            new_vertices[i] = (1 - lambda_smooth) * vertices[i] + lambda_smooth * avg_pos
        vertices = new_vertices

    smoothed_mesh = mesh.copy()
    smoothed_mesh.vertices = vertices
    return smoothed_mesh

def preprocess_mesh(mesh):
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.fix_normals()
    return mesh


class RayCaster:
    def __init__(self, mesh_path, k_image, size, r_sub=4, smoothing_iters=3):
        H, W = size
        mesh=trimesh.load_mesh(mesh_path)
        # Load and clean mesh
        #rospy.loginfo("Loading and preprocessing mesh")
        #mesh = preprocess_mesh(trimesh.load_mesh(mesh_path))

        # Smoothing
        #rospy.loginfo("Applying Laplacian smoothing")
        #mesh = laplacian_smoothing(mesh, iterations=smoothing_iters, lambda_smooth=0.1)

        #rospy.loginfo("RayCaster initialization completed")

        # Costruisci intersector
        self._rmi = RayMeshIntersector(mesh)

        # Genera raggi della camera
        self._start, stop, self._dir = get_rays(k_image, size, extrinsic=None, d_min=0.3, d_max=1.4)
        self.r_sub = r_sub
        self._start = self._start[::self.r_sub, ::self.r_sub]
        self._dir = self._dir[::self.r_sub, ::self.r_sub]

    def raycast(self, H_cam):
        ray_origins = transform_points(self._start.reshape((-1, 3)), H_cam)
        H_turn = np.eye(4)
        H_turn[:3, :3] = H_cam[:3, :3]
        ray_directions = transform_points(self._dir.reshape((-1, 3)), H_turn)

        locations, index_ray, index_tri = self._rmi.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )

        return locations, index_ray, index_tri, ray_origins
