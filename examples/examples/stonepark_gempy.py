"""
Construct a 3D geological model of the Stonepark deposit using GemPy.


"""
import numpy as np
# %%
# Read nc from subsurface


# %%
import os
from dotenv import dotenv_values

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
# %%
# Element 1 is an intrusion


#  %%
# Setup gempy object
# structural_elements.pop(1)

structural_group_red = gp.data.StructuralGroup(
    name="Red",
    # elements=[structural_elements[i] for i in [0, 4, 6, 8]],
    elements=[structural_elements[i] for i in [0, 4, 8]],
    structural_relation=gp.data.StackRelationType.ERODE
)

structural_group_green = gp.data.StructuralGroup(
    name="Green",
    elements=[structural_elements[i] for i in [5]],
    structural_relation=gp.data.StackRelationType.ERODE
)

structural_group_blue = gp.data.StructuralGroup(
    name="Blue",
    elements=[structural_elements[i] for i in [2, 3]],
    structural_relation=gp.data.StackRelationType.ERODE
)

structural_group_intrusion = gp.data.StructuralGroup(
    name="Intrusion",
    elements=[structural_elements[i] for i in [1]],
    structural_relation=gp.data.StackRelationType.ERODE
)

structural_groups = [structural_group_intrusion, structural_group_green, structural_group_blue, structural_group_red]
structural_frame = gp.data.StructuralFrame(
    structural_groups=structural_groups[3:],
    color_gen=color_gen
)
# TODO: If elements do not have color maybe loop them on structural frame constructor?

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Tutorial_ch1_1_Basics',
    extent=global_extent,
    resolution=[20, 10, 20],
    refinement=4,  # * Here we define the number of octree levels. If octree levels are defined, the resolution is ignored.
    structural_frame=structural_frame
)

# %% Extract surface points from the dataset
geo_model

# %% 
# gpv.plot_3d(geo_model)

geo_model.interpolation_options.mesh_extraction = True
geo_model.interpolation_options.kernel_options.compute_condition_number = True
geo_model.interpolation_options.kernel_options.range = 1
geo_model.interpolation_options.kernel_options.c_o = 4

geo_model.update_transform()

gp.API.compute_API.optimize_and_compute(
    geo_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
    ),
    max_epochs=100,
    convergence_criteria=1e5

)

gpv.plot_2d(geo_model, show_scalar=True)

# 
# gp.compute_model(
#     geo_model,
#     engine_config=gp.data.GemPyEngineConfig(
#         backend=gp.data.AvailableBackends.PYTORCH,
#     ),
# )

nugget_effect = geo_model.taped_interpolation_input.surface_points.nugget_effect_scalar
surface_points_xyz = geo_model.surface_points.df[['X', 'Y', 'Z']].to_numpy()

nugget_numpy = nugget_effect.detach().numpy()[:]

array_to_plot = nugget_numpy

plt.hist(nugget_numpy, bins=50, color='black', alpha=0.7, log=True)
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.title('Histogram of Eigenvalues (nugget-grad)')
plt.show()
clean_sp = surface_points_xyz[1:]

gempy_vista = gpv.plot_3d(geo_model, show=False,
                          kwargs_plot_structured_grid={'opacity': 0.3})

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
