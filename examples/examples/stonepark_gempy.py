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
from gempy_engine.core.data.continue_epoch import ContinueEpoch

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



claaned_sp_grab = np.array(surface_points_xyz)[:, 2]  
# Get the values larger than -300
grab_ = claaned_sp_grab < -600

# claaned_sp_grab = np.where(grab_)
# cleaned_supoints = surface_points_xyz[claaned_sp_grab]

cleaned_supoints = surface_points_xyz

nuggets = np.ones(len(cleaned_supoints)) * 0.001
nuggets[grab_] = .000001

surface_points = gp.data.SurfacePointsTable.from_arrays(
    x=cleaned_supoints[:, 0],
    y=cleaned_supoints[:, 1],
    z=cleaned_supoints[:, 2],
    names="channel_1",
    nugget=nuggets
)

n = 0
orientations = gp.data.OrientationsTable.from_arrays(
    x=cleaned_supoints[n:, 0],
    y=cleaned_supoints[n:, 1],
    z=cleaned_supoints[n:, 2],
    G_x=orientations_gxyz[n:, 0],
    G_y=orientations_gxyz[n:, 1],
    G_z=orientations_gxyz[n:, 2],
    nugget=1,
    names="channel_1"
)

structural_frame = geo_model.structural_frame

structural_frame.structural_elements[0].surface_points = surface_points
structural_frame.structural_elements[0].orientations = orientations

geo_model.interpolation_options.mesh_extraction = True
geo_model.interpolation_options.kernel_options.compute_condition_number = True
geo_model.update_transform()


import torch
# Define the optimizer
gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
    )
)

param = geo_model.taped_interpolation_input.surface_points.nugget_effect_scalar
optimizer = torch.optim.Adam([param], lr=0.01)
max_epochs = 10

# Optimization loop
def check_convergence_criterion():
    pass

import gempy_engine
geo_model.interpolation_options.kernel_options.optimizing_condition_number = True
for epoch in range(max_epochs):
    optimizer.zero_grad()
    
    try:
        geo_model.taped_interpolation_input.grid = geo_model.interpolation_input.grid

        gempy_engine.compute_model(
            interpolation_input=geo_model.taped_interpolation_input,
            options=geo_model.interpolation_options,
            data_descriptor=geo_model.input_data_descriptor,
            geophysics_input=geo_model.geophysics_input,
        )
    except ContinueEpoch:
        # Update the vector
        optimizer.step()
        param.data = param.data.clamp_(min=0)  # Replace negative values with 0
        continue

    # Optional: Apply constraints to the vector

    # Monitor progress
    if epoch % 100 == 0:
        # print(f"Epoch {epoch}: Condition Number = {condition_number.item()}")
        print(f"Epoch {epoch}: Condition Number =") 

    # Check for convergence
    if check_convergence_criterion():
        break

geo_model.interpolation_options.kernel_options.optimizing_condition_number = False

sp_gradients = geo_model.taped_interpolation_input.surface_points.sp_coords.grad
nugget_effect = param
orientations_gradients_pos = geo_model.taped_interpolation_input.orientations.dip_positions
orientations_gradients_pos = geo_model.taped_interpolation_input.orientations.dip_gradients.grad
orientations_gradients_gxyz = geo_model.taped_interpolation_input.orientations.dip_gradients.grad

import matplotlib.pyplot as plt


sp_gradients_numpy = sp_gradients.detach().numpy()[1:]
nugget_numpy = nugget_effect.detach().numpy()[:]

array_to_plot = nugget_numpy

x_grad = sp_gradients_numpy[:, 0]
y_grad = sp_gradients_numpy[:, 1]
z_grad = sp_gradients_numpy[:, 2]
bool_z = z_grad > 0
bool_y = y_grad > 0
bool_x = x_grad > 0
bool_ = np.logical_and(bool_z, bool_y, bool_x)

plt.hist(z_grad, bins=50, color='blue', alpha=0.7, log=True)
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.title('Histogram of Eigenvalues (Z-grad)')
plt.show()

plt.hist(x_grad, bins=50, color='red', alpha=0.7, log=True)
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.title('Histogram of Eigenvalues (X-grad)')
plt.show()

plt.hist(y_grad, bins=50, color='green', alpha=0.7, log=True)
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.title('Histogram of Eigenvalues (Y-grad)')
plt.show()


plt.hist(nugget_numpy, bins=50, color='black', alpha=0.7, log=True)
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.title('Histogram of Eigenvalues (nugget-grad)')
plt.show()
clean_sp = cleaned_supoints[1:][bool_]

# surface_points = gp.data.SurfacePointsTable.from_arrays(
#     x=clean_sp[:, 0],
#     y=clean_sp[:, 1],
#     z=clean_sp[:, 2],
#     names="channel_1"
# )
# 
# 
# structural_frame.structural_elements[0].surface_points = surface_points

gempy_vista = gpv.plot_3d(geo_model, show=False)
if ADD_ORIGINAL_MESH := False:
    gempy_vista.p.add_mesh(triangulated_mesh, color="red", opacity=0.5)



import pyvista as pv
# Create a point cloud mesh
point_cloud = pv.PolyData(cleaned_supoints[0:])

# Ensure there are no non-positive values in 'vales' before taking the logarithm
values =np.linalg.norm(sp_gradients_numpy[bool_], axis=1)
values = np.maximum(values, 1e-6)

log_values = np.log(np.abs(array_to_plot)) # Apply logarithmic transformation

point_cloud['log_values'] = log_values  # Add the log values as a scalar array

point_cloud['values'] = array_to_plot  # Add the log values as a scalar array


gempy_vista.p.add_mesh(
    point_cloud,
    # scalars='log_values',
    scalars='values',
    cmap='inferno',
    point_size=25,
)
# gempy_vista.p.add_mesh(surface_points_xyz[bool_x], color="red", point_size=30)
# gempy_vista.p.add_mesh(surface_points_xyz[bool_y], color="green", point_size=25)
# gempy_vista.p.add_mesh(surface_points_xyz[bool_z], color="blue", point_size=20)

gempy_vista.p.show()