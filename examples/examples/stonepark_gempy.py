"""
Construct a 3D geological model of the Stonepark deposit using GemPy.


"""
import numpy as np
# %%
# Read nc from subsurface


# %%
import xarray as xr
import os
from dotenv import dotenv_values

import subsurface

config = dotenv_values()
path = config.get("PATH_TO_STONEPARK_Subsurface")
for e, filename in enumerate(os.listdir(path)):
    base, ext = os.path.splitext(filename)
    if ext == '.nc':
        dataset: xr.Dataset = xr.open_dataset(path + "/" + filename)
        print(dataset)
        break

# %% 
# Operate in the dataset

# %%
# Get the extent from the dataset

# Assuming dataset.coords["XYZ"] is a 2D DataArray with shape (n, 3)
# where n is the number of points and each row is a point with [X, Y, Z] coordinates

# Extract X, Y, Z coordinates
x_coord = dataset.vertex[:, 0]  # Assuming the first column is X
y_coord = dataset.vertex[:, 1]  # Assuming the second column is Y
z_coord = dataset.vertex[:, 2]  # Assuming the third column is Z

# Calculate min and max for each
x_min = x_coord.min().values
x_max = x_coord.max().values
y_min = y_coord.min().values
y_max = y_coord.max().values
z_min = z_coord.min().values
z_max = z_coord.max().values

# Region of Interest
roi = [x_min, x_max, y_min, y_max, z_min, z_max]
# Print or use the region of interest
print(roi)

# %%
# Setup gempy object
import gempy as gp

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Tutorial_ch1_1_Basics',
    extent=roi,
    refinement=4,  # * Here we define the number of octree levels. If octree levels are defined, the resolution is ignored.
    structural_frame=gp.data.StructuralFrame.initialize_default_structure()
)

# %% Extract surface points from the dataset
geo_model

# %% 
import gempy_viewer as gpv

unstruct = subsurface.UnstructuredData(dataset)
ts = subsurface.TriSurf(mesh=unstruct)
triangulated_mesh = subsurface.visualization.to_pyvista_mesh(ts)

# %%
# Decimate the mesh to reduce the number of points
decimated_mesh = triangulated_mesh.decimate_pro(0.98)

# Compute normals
normals = decimated_mesh.compute_normals(point_normals=False, cell_normals=True, consistent_normals=True)

normals_array = np.array(normals.cell_data["Normals"])

# Extract the points and normals from the decimated mesh
vertex_sp = normals.cell_centers().points
vertex_grads = normals_array

# Filter out the points where the z-component of the normal is positive
# and is the largest component of the normal vector
positive_z = vertex_grads[:, 2] > 0
largest_component_z = np.all(vertex_grads[:, 2:3] >= np.abs(vertex_grads[:, :2]), axis=1)

# Combine both conditions
filter_mask = np.logical_and(positive_z, largest_component_z)

# Apply the filter to points and normals
surface_points_xyz = vertex_sp[filter_mask]
orientations_gxyz = vertex_grads[filter_mask]

surface_points = gp.data.SurfacePointsTable.from_arrays(
    x=surface_points_xyz[:, 0],
    y=surface_points_xyz[:, 1],
    z=surface_points_xyz[:, 2],
    names="channel_1"
)

orientations = gp.data.OrientationsTable.from_arrays(
    x=surface_points_xyz[:, 0],
    y=surface_points_xyz[:, 1],
    z=surface_points_xyz[:, 2],
    G_x=orientations_gxyz[:, 0],
    G_y=orientations_gxyz[:, 1],
    G_z=orientations_gxyz[:, 2],
    nugget=1,
    names="channel_1"
)

structural_frame = geo_model.structural_frame

structural_frame.structural_elements[0].surface_points = surface_points
structural_frame.structural_elements[0].orientations = orientations

geo_model.interpolation_options.mesh_extraction = True
geo_model.update_transform()

gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH
    )
)

gempy_vista = gpv.plot_3d(geo_model, show=False)
if ADD_ORIGINAL_MESH := False:
    gempy_vista.p.add_mesh(triangulated_mesh, color="red", opacity=0.5)
# gempy_vista.p.show()
