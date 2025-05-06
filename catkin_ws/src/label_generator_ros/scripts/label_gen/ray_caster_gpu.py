import numpy as np
import trimesh
import taichi as ti
import rospy
from tqdm import tqdm

from .helper import get_rays, transform_points

ti.init(arch=ti.cuda)

def laplacian_smoothing(mesh, iterations=10, lambda_smooth=0.5):
    vertices = mesh.vertices
    faces = mesh.faces
    vertex_neighbors = {i: set() for i in range(len(vertices))}
    for face in faces:
        for i, vi in enumerate(face):
            for j, vj in enumerate(face):
                if i != j:
                    vertex_neighbors[vi].add(vj)

    def smooth_vertices(vertices):
        new_vertices = vertices.copy()
        for i in range(len(vertices)):
            neighbors = vertex_neighbors[i]
            if neighbors:
                avg_position = np.mean(vertices[list(neighbors)], axis=0)
                new_vertices[i] = (1 - lambda_smooth) * vertices[i] + lambda_smooth * avg_position
        return new_vertices

    for _ in tqdm(range(iterations), desc="Smoothing", unit="iteration"):
        vertices = smooth_vertices(vertices)

    mesh.vertices = vertices
    return mesh

def preprocess_mesh(mesh):
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.fix_normals()
    return mesh


@ti.data_oriented
class RayTracingGPU:
    def __init__(self, mesh_path, k_image, size, r_sub=4, smoothing_iters=30):
        H, W = size
        self.H, self.W = H, W
        self.r_sub = r_sub
        self.k_image = k_image

        rospy.loginfo("Loading and smoothing mesh")
        mesh = preprocess_mesh(trimesh.load(mesh_path))
        #mesh = laplacian_smoothing(mesh, iterations=smoothing_iters, lambda_smooth=0.1)
        rospy.loginfo("Finished mesh preparation")

        # Store triangles
        self.triangles_np = mesh.triangles  # (N, 3, 3)
        self.num_triangles = self.triangles_np.shape[0]

        # Subsample rays
        self._start, _, self._dir = get_rays(k_image, size, extrinsic=None, d_min=0.3, d_max=1.4)
        self._start = self._start[::r_sub, ::r_sub]
        self._dir = self._dir[::r_sub, ::r_sub]
        self.num_rays = self._start.shape[0] * self._start.shape[1]

        # Taichi buffers
        self.ray_origins = ti.Vector.field(3, dtype=ti.f32, shape=self.num_rays)
        self.ray_directions = ti.Vector.field(3, dtype=ti.f32, shape=self.num_rays)
        self.hit_mask = ti.field(dtype=ti.i32, shape=self.num_rays)
        self.hit_points = ti.Vector.field(3, dtype=ti.f32, shape=self.num_rays)
        self.triangles = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_triangles, 3))

        # Upload triangles to GPU
        for i in range(self.num_triangles):
            for j in range(3):
                self.triangles[i, j] = self.triangles_np[i, j]

    def raycast(self, H_cam):
        rays_o = transform_points(self._start.reshape((-1, 3)), H_cam)
        H_rot = np.eye(4)
        H_rot[:3, :3] = H_cam[:3, :3]
        rays_d = transform_points(self._dir.reshape((-1, 3)), H_rot)

        for i in range(self.num_rays):
            self.ray_origins[i] = rays_o[i]
            self.ray_directions[i] = rays_d[i]

        self.trace_kernel(self.num_rays, self.num_triangles)
        hit_mask_np = self.hit_mask.to_numpy()
        hit_points_np = self.hit_points.to_numpy()

        index_ray = np.where(hit_mask_np == 1)[0]
        locations = hit_points_np[index_ray]
        index_tri = np.zeros_like(index_ray)  # Placeholder: no triangle IDs returned

        return locations, index_ray, index_tri, rays_o

    @ti.kernel
    def trace_kernel(self, num_rays: int, num_triangles: int):
        for i in range(num_rays):
            origin = self.ray_origins[i]
            direction = self.ray_directions[i].normalized()
            min_dist = 1e10
            hit = 0
            hit_pos = ti.Vector([0.0, 0.0, 0.0])

            for t in range(num_triangles):
                v0 = self.triangles[t, 0]
                v1 = self.triangles[t, 1]
                v2 = self.triangles[t, 2]

                edge1 = v1 - v0
                edge2 = v2 - v0
                h = direction.cross(edge2)
                a = edge1.dot(h)

                if ti.abs(a) > 1e-5:
                    f = 1.0 / a
                    s = origin - v0
                    u = f * s.dot(h)
                    if 0.0 <= u <= 1.0:
                        q = s.cross(edge1)
                        v = f * direction.dot(q)
                        if 0.0 <= v and u + v <= 1.0:
                            t_ = f * edge2.dot(q)
                            if t_ > 1e-4 and t_ < min_dist:
                                min_dist = t_
                                hit = 1
                                hit_pos = origin + direction * t_

            self.hit_mask[i] = hit
            if hit:
                self.hit_points[i] = hit_pos
