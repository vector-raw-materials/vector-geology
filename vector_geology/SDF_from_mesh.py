import numpy as np
import trimesh
from scipy.spatial import cKDTree

def load_mesh(file_path):
    # Load a mesh from the given file path
    return trimesh.load(file_path)

def create_signed_distance_function(mesh, grid_resolution=100):
    # Create a grid of points
    min_bound = mesh.bounds[0]
    max_bound = mesh.bounds[1]
    x = np.linspace(min_bound[0], max_bound[0], grid_resolution)
    y = np.linspace(min_bound[1], max_bound[1], grid_resolution)
    z = np.linspace(min_bound[2], max_bound[2], grid_resolution)
    grid = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack(map(np.ravel, grid)).T

    # Create a KD-tree for efficient nearest-neighbor queries
    kdtree = cKDTree(mesh.vertices)

    # Find the nearest points on the mesh for each grid point
    distances, indices = kdtree.query(points)

    # Determine inside/outside using ray casting
    # For each point, we cast a ray and count the number of intersections with the mesh.
    # An odd number of intersections means the point is inside; even means outside.
    signs = np.array([mesh.ray.intersects_location([p], [[1, 0, 0]])[0].shape[0] % 2 for p in points])
    signs = signs * 2 - 1  # Convert to -1 for inside, 1 for outside

    # Combine distances and signs to get the signed distance
    signed_distances = distances * signs

    # Reshape to the grid
    sdf = signed_distances.reshape(grid_resolution, grid_resolution, grid_resolution)

    return sdf

# Example usage (assuming you have a mesh file)
# mesh = load_mesh("path_to_your_mesh_file.obj")
# sdf = create_signed_distance_function(mesh)
