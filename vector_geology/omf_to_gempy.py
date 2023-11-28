import os
import xarray as xr
import numpy as np
import gempy as gp
import subsurface
# Function to calculate the extent (ROI) for a single dataset
def calculate_extent(dataset):
    x_coord, y_coord, z_coord = dataset.vertex[:, 0], dataset.vertex[:, 1], dataset.vertex[:, 2]
    return [x_coord.min().values, x_coord.max().values, y_coord.min().values, y_coord.max().values, z_coord.min().values, z_coord.max().values]

# Function to extract surface points and orientations from the dataset

def extract_surface_points_and_orientations(dataset, name, color_generator) -> gp.data.StructuralElement:
    # Extract surface points and orientations
    unstruct = subsurface.UnstructuredData(dataset)
    ts = subsurface.TriSurf(mesh=unstruct)
    triangulated_mesh = subsurface.visualization.to_pyvista_mesh(ts)
    # print(name)
    # subsurface.visualization.pv_plot([triangulated_mesh])
    

    # Decimate the mesh and compute normals
    decimated_mesh = triangulated_mesh.decimate_pro(0.98)
    normals = decimated_mesh.compute_normals(point_normals=False, cell_normals=True, consistent_normals=True)
    normals_array = np.array(normals.cell_data["Normals"])

    # Filter points
    vertex_sp = normals.cell_centers().points
    vertex_grads = normals_array
    positive_z = vertex_grads[:, 2] > 0
    largest_component_z = np.all(vertex_grads[:, 2:3] >= np.abs(vertex_grads[:, :2]), axis=1)
    filter_mask = np.logical_and(positive_z, largest_component_z)

    surface_points_xyz = vertex_sp[filter_mask]
    orientations_gxyz = vertex_grads[filter_mask]
    nuggets = np.ones(len(surface_points_xyz)) * 0.000001

    # Create SurfacePointsTable and OrientationsTable
    surface_points = gp.data.SurfacePointsTable.from_arrays(
        x=surface_points_xyz[:, 0],
        y=surface_points_xyz[:, 1],
        z=surface_points_xyz[:, 2],
        names=name,
        nugget=nuggets
    )

    every = 10
    orientations = gp.data.OrientationsTable.from_arrays(
        x=surface_points_xyz[::every, 0],
        y=surface_points_xyz[::every, 1],
        z=surface_points_xyz[::every, 2],
        G_x=orientations_gxyz[::every, 0],
        G_y=orientations_gxyz[::every, 1],
        G_z=orientations_gxyz[::every, 2],
        nugget=4,
        names=name
    )
    
    structural_element = gp.data.StructuralElement(
        name=name,
        surface_points=surface_points,
        orientations=orientations,
        color=next(color_generator)
    )
    
    return structural_element


# Function to update the global extent based on each file's extent
def update_global_extent(global_extent, file_extent):
    if not global_extent:
        return file_extent
    else:
        return [min(global_extent[0], file_extent[0]), max(global_extent[1], file_extent[1]),
                min(global_extent[2], file_extent[2]), max(global_extent[3], file_extent[3]),
                min(global_extent[4], file_extent[4]), max(global_extent[5], file_extent[5])]


def process_file(filename, global_extent, color_generator: gp.data.ColorsGenerator):
    dataset: xr.Dataset = xr.open_dataset(filename)
    file_extent = calculate_extent(dataset)
    global_extent = update_global_extent(global_extent, file_extent)

    base, ext = os.path.splitext(filename)
    base = os.path.basename(base)
    structural_element = extract_surface_points_and_orientations(dataset, base, color_generator)
    
    return structural_element, global_extent
