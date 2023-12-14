"""
Constructing Structural Geological Model
----------------------------------------

This example illustrates how to construct a 3D geological model of a deposit using GemPy.
"""

# Import necessary libraries
import time
import numpy as np
import os
import xarray as xr
from dotenv import dotenv_values
from vector_geology.omf_to_gempy import process_file
import gempy as gp
import gempy_viewer as gpv
from vector_geology.model_building_functions import optimize_nuggets_for_group
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions

# Start timer
start_time = time.time()

# %%
# Loading and Processing Data
# ---------------------------
# Here we load the data necessary for the model construction, stored in nc files.
# The data is processed and stored in xarray files for easier post-processing.

config = dotenv_values()
path = config.get("PATH_TO_MODEL_1_Subsurface")
structural_elements = []
global_extent = None
color_gen = gp.data.ColorsGenerator()

for filename in os.listdir(path):
    base, ext = os.path.splitext(filename)
    if ext == '.nc':
        structural_element, global_extent = process_file(os.path.join(path, filename), global_extent, color_gen)
        structural_elements.append(structural_element)

# %%
# Setting Up GemPy Model
# -----------------------
# Here we set up the GemPy model object, defining structural groups and their relations.
# We also configure various properties of the structural model.

# Define structural groups
structural_group_red = gp.data.StructuralGroup(
    name="Red",
    elements=[structural_elements[i] for i in [0, 4, 8]],
    structural_relation=gp.data.StackRelationType.ERODE
)

# Any, Probably we can decimize this an extra notch
structural_group_green = gp.data.StructuralGroup(
    name="Green",
    elements=[structural_elements[i] for i in [5]],
    structural_relation=gp.data.StackRelationType.ERODE
)

# Blue range 2 cov 4
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


# Combine structural groups into a frame
structural_groups = [structural_group_intrusion, structural_group_green, structural_group_blue, structural_group_red]
structural_frame = gp.data.StructuralFrame(
    structural_groups=structural_groups[2:],
    color_gen=color_gen
)

# Create the GeoModel object
geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Tutorial_ch1_1_Basics',
    extent=global_extent,
    resolution=[20, 10, 20],
    refinement=6,  # * Here we define the number of octree levels. If octree levels are defined, the resolution is ignored.
    structural_frame=structural_frame
)


# %%
# Adding Topography
# -----------------
# Here we add topography to our model using a dataset in the nc format.

import xarray as xr

dataset: xr.Dataset = xr.open_dataset(os.path.join(path, "Topography.nc"))

gp.set_topography_from_arrays(
    grid=geo_model.grid,
    xyz_vertices=dataset.vertex.values
)


# %%
# Optimizing Nuggets
# ------------------

# In such a complex geometries often data does not fit perfectly. In order to account for this, GemPy allows to add a
# small random noise to the data. This is done by adding a small random value to the diagonal of the covariance matrix.
# We can optimize this value with respect to the condition number of the matrix. The condition number is a measure of
# how well the matrix can be inverted. The higher the condition number, the worse the matrix can be inverted. This
# means that the data is not well conditioned and the nugget should be increased. On the other hand, if the condition
# number is too low, the data is overfitted and the nugget should be decreased. The optimal value is the one that
# minimizes the condition number. This can be done with the following function:

TRIGGER_OPTIMIZE_NUGGETS = False
APPLY_OPTIMIZED_NUGGETS = True
if TRIGGER_OPTIMIZE_NUGGETS:

    geo_model.interpolation_options.kernel_options.range = 0.7
    geo_model.interpolation_options.kernel_options.c_o = 4
    optimize_nuggets_for_group(
        geo_model=geo_model,
        structural_group=structural_group_red,
        plot_evaluation=False,
        plot_result=True
    )

    geo_model.interpolation_options.kernel_options.range = 2
    geo_model.interpolation_options.kernel_options.c_o = 4

    optimize_nuggets_for_group(
        geo_model=geo_model,
        structural_group=structural_group_blue,
        plot_evaluation=False,
        plot_result=False
    )

    optimize_nuggets_for_group(
        geo_model=geo_model,
        structural_group=structural_group_green,
        plot_evaluation=False,
        plot_result=True
    )

if APPLY_OPTIMIZED_NUGGETS:
    loaded_nuggets_red = np.load("../temp/nuggets_Red.npy")
    loaded_nuggets_green = np.load("../temp/nuggets_Green.npy")
    loaded_nuggets_blue = np.load("../temp/nuggets_Blue.npy")

    gp.modify_surface_points(
        geo_model,
        slice=None,
        elements_names=[element.name for element in geo_model.structural_frame.get_group_by_name('Red').elements],
        nugget=loaded_nuggets_red
    )

    if True:  # Ignore OB
        gp.modify_surface_points(
            geo_model,
            slice=None,
            elements_names=[element.name for element in geo_model.structural_frame.get_group_by_name('Blue').elements],
            nugget=loaded_nuggets_blue
        )
    if False:
        gp.modify_surface_points(
            geo_model,
            slice=None,
            elements_names=[element.name for element in geo_model.structural_frame.get_group_by_name('Green').elements],
            nugget=loaded_nuggets_green
        )


# %%
# Model Computation
# -----------------
# Finally, we compute the model. This involves setting various interpolation options
# and computing the model using GemPy.

geo_model.interpolation_options.mesh_extraction = True
geo_model.interpolation_options.kernel_options.range = .7
geo_model.interpolation_options.kernel_options.c_o = 3
geo_model.interpolation_options.kernel_options.compute_condition_number = True
geo_model.interpolation_options.kernel_options.kernel_function = AvailableKernelFunctions.cubic

# Update color and transformation settings
geo_model.structural_frame.get_element_by_name("KKR").color = "#A46283"
geo_model.structural_frame.get_element_by_name("LGR").color = "#6394A4"
geo_model.structural_frame.get_element_by_name("WAL").color = "#72A473"
geo_model.structural_frame.get_element_by_name("ABL").color = "#1D3943"
geo_model.structural_frame.basement_color = "#8B4220"

geo_model.update_transform()

# Compute the model
gp.compute_model(
    geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype="float64"
    ),
)
# %%
# Visualization
# -------------
# Here we visualize the output of our model both in 2D and 3D.
gpv.plot_2d(geo_model, show_scalar=False)

# %%
gempy_vista = gpv.plot_3d(
    model=geo_model,
    show=True,
    kwargs_plot_structured_grid={'opacity': 0.8}
)

# %%
# Calculate Execution Time
# ------------------------
# Finally, we calculate the total execution time of the script.
end_time = time.time()
execution_time = end_time - start_time

print(f"The function executed in {execution_time} seconds.")

# sphinx_gallery_thumbnail_number = -1
