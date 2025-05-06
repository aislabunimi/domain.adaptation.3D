import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from .helper import get_rays, transform_points
import open3d as o3d
import trimesh
from tqdm import tqdm
import rospy

import trimesh
import numpy as np
import time

def laplacian_smoothing(mesh, iterations=10, lambda_smooth=0.5):
    """
    Esegui il Laplacian smoothing su una mesh 3D.
    
    :param mesh: la mesh Trimesh
    :param iterations: numero di iterazioni di smoothing
    :param lambda_smooth: peso della levigatura (tra 0 e 1)
    :return: la mesh lisciata
    """
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Costruisci una mappa di adiacenza dei vertici
    vertex_neighbors = {i: set() for i in range(len(vertices))}
    for face in faces:
        for i, vi in enumerate(face):
            for j, vj in enumerate(face):
                if i != j:
                    vertex_neighbors[vi].add(vj)

    # Funzione per il calcolo del nuovo vertice levigato
    def smooth_vertices(vertices):
        new_vertices = vertices.copy()
        for i in range(len(vertices)):
            neighbors = vertex_neighbors[i]
            if neighbors:
                # Calcola la media dei vicini
                avg_position = np.mean(vertices[list(neighbors)], axis=0)
                # Applicare la levigatura (Laplaciano)
                new_vertices[i] = (1 - lambda_smooth) * vertices[i] + lambda_smooth * avg_position
        return new_vertices

    # Itera per il numero di iterazioni desiderato
    for _ in tqdm(range(iterations), desc="Smoothing", unit="iteration"):
        vertices = smooth_vertices(vertices)

    # Restituisci la mesh lisciata
    mesh.vertices = vertices
    return mesh

def preprocess_mesh(mesh):
    mesh.remove_unreferenced_vertices()
    # Clean up invalid values
    mesh.remove_infinite_values()
    mesh.fix_normals()
    return mesh


class RayCaster:
    def __init__(self, mesh_path, k_image, size, r_sub=4, smoothing_iters=30):
        H, W = size

        # Prepare mesh
        rospy.loginfo("Starting smoothing")
        mesh = preprocess_mesh(trimesh.load_mesh(mesh_path))
        mesh = laplacian_smoothing(mesh, iterations=smoothing_iters, lambda_smooth=0.1)
        # -- Normalizzazione (scala unitaria e centratura) --

        rospy.loginfo("finished")

        # Build the ray intersector
        self._rmi = RayMeshIntersector(mesh)

        # Generate camera rays
        self._start, stop, self._dir = get_rays(
            k_image, size, extrinsic=None, d_min=0.3, d_max=1.4
        )

        self.r_sub = r_sub
        self._start = self._start[::self.r_sub, ::self.r_sub]
        self._dir = self._dir[::self.r_sub, ::self.r_sub]

    def raycast(self, H_cam):
        # Move Camera Rays
        ray_origins = transform_points(self._start.reshape((-1, 3)), H_cam)
        H_turn = np.eye(4)
        H_turn[:3, :3] = H_cam[:3, :3]
        ray_directions = transform_points(self._dir.reshape((-1, 3)), H_turn)
        

        # Perform Raytracing
        locations, index_ray, index_tri = self._rmi.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )

        return locations, index_ray, index_tri, ray_origins
