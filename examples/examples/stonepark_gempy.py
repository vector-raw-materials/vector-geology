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
import matplotlib.pyplot as plt
import pyvista as pv

from vector_geology.omf_to_gempy import process_file
import gempy as gp
import gempy_viewer as gpv

config = dotenv_values()
path = config.get("PATH_TO_STONEPARK_Subsurface")
structural_elements = []
accumulated_roi = []
global_extent = None
color_gen = gp.data.ColorsGenerator()

for e, filename in enumerate(os.listdir(path)):
    base, ext = os.path.splitext(filename)
    if ext == '.nc':
        structural_element, global_extent = process_file(os.path.join(path, filename), global_extent, color_gen)
        structural_elements.append(structural_element)
# 
# # %% 
# # Operate in the dataset
# 
# # %%
# # Get the extent from the dataset
# 
# # Assuming dataset.coords["XYZ"] is a 2D DataArray with shape (n, 3)
# # where n is the number of points and each row is a point with [X, Y, Z] coordinates
# 
# # Extract X, Y, Z coordinates
# x_coord = dataset.vertex[:, 0]  # Assuming the first column is X
# y_coord = dataset.vertex[:, 1]  # Assuming the second column is Y
# z_coord = dataset.vertex[:, 2]  # Assuming the third column is Z
# 
# # Calculate min and max for each
# x_min = x_coord.min().values
# x_max = x_coord.max().values
# y_min = y_coord.min().values
# y_max = y_coord.max().values
# z_min = z_coord.min().values
# z_max = z_coord.max().values
# 
# # Region of Interest
# roi = [x_min, x_max, y_min, y_max, z_min, z_max]
# # Print or use the region of interest
# print(roi)
# 
# # %%
# Setup gempy object

structural_group = gp.data.StructuralGroup(
    name="Stonepark",
    elements=structural_elements,
    structural_relation=gp.data.StackRelationType.ERODE
)

structural_frame = gp.data.StructuralFrame(
    structural_groups=[structural_group],
    color_gen = color_gen
)
# TODO: If elements do not have color maybe loop them on structural frame constructor?

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Tutorial_ch1_1_Basics',
    extent=global_extent,
    refinement=4,  # * Here we define the number of octree levels. If octree levels are defined, the resolution is ignored.
    structural_frame=structural_frame
)

# %% Extract surface points from the dataset
geo_model

# %% 
gpv.plot_3d(geo_model)

geo_model.interpolation_options.mesh_extraction = True
geo_model.interpolation_options.kernel_options.compute_condition_number = True
geo_model.update_transform()

gp.API.compute_API.optimize_and_compute(
    geo_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
    )
)

nugget_effect = geo_model.taped_interpolation_input.surface_points.nugget_effect_scalar
nugget_numpy = nugget_effect.detach().numpy()[:]

array_to_plot = nugget_numpy

plt.hist(nugget_numpy, bins=50, color='black', alpha=0.7, log=True)
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.title('Histogram of Eigenvalues (nugget-grad)')
plt.show()
clean_sp = surface_points_xyz[1:]

gempy_vista = gpv.plot_3d(geo_model, show=False)
if ADD_ORIGINAL_MESH := False:
    gempy_vista.p.add_mesh(triangulated_mesh, color="red", opacity=0.5)

# Create a point cloud mesh
point_cloud = pv.PolyData(surface_points_xyz[0:])
point_cloud['values'] = array_to_plot  # Add the log values as a scalar array

gempy_vista.p.add_mesh(
    point_cloud,
    scalars='values',
    cmap='inferno',
    point_size=25,
)

gempy_vista.p.show()
