"""
Construct Model 1 with Helper Functions
---------------------------------------

This example demonstrates how to construct a 3D geological model of the Model 1 deposit using GemPy. 
It leverages custom APIs to streamline the modeling process.
"""

import time
import os
import xarray as xr
from dotenv import dotenv_values
import gempy as gp
import gempy_viewer as gpv

from vector_geology.model_1_builder import initialize_geo_model
from vector_geology.omf_to_gempy import process_file

# %%
# Start the timer to measure execution time
start_time = time.time()

# %%
# Load environment variables and path configurations
# This step is crucial for setting up the directory paths for the model data.
config = dotenv_values()
path = config.get("PATH_TO_MODEL_1_Subsurface")

# %%
# Initialize lists to store data
# Here we prepare the containers for the structural elements and the global extent.
structural_elements = []
global_extent = None
color_gen = gp.data.ColorsGenerator()

# %%
# Process each .nc file in the specified directory
# This loop reads the necessary data from each .nc file for modeling.
for filename in os.listdir(path):
    base, ext = os.path.splitext(filename)
    if ext == '.nc':
        structural_element, global_extent = process_file(os.path.join(path, filename), global_extent, color_gen)
        structural_elements.append(structural_element)

# %%
# Initialize the GemPy model
# We use the data processed above to initialize the geological model with GemPy.
geo_model = initialize_geo_model(
    structural_elements=structural_elements,
    extent=global_extent,
    topography=(xr.open_dataset(os.path.join(path, "Topography.nc"))),
    load_nuggets=True
)

# %%
# Display the initialized model
# It's helpful to print the model to verify its initialization.
print(geo_model)

# %%
# Modify the interpolation options
# This step configures the model's interpolation settings for better accuracy.
interpolation_options = geo_model.interpolation_options
interpolation_options.mesh_extraction = True
interpolation_options.kernel_options.range = 0.7
interpolation_options.kernel_options.c_o = 3
interpolation_options.kernel_options.compute_condition_number = True

# %%
# Modify surface points
# Adjustments to the surface points can influence the resulting geological features.
gp.modify_surface_points(
    geo_model,
    slice=0,
    X=geo_model.surface_points_copy.data[0][0] + 130,
)

# %%
# Compute the model
# This is where the model computation happens, utilizing the GemPy engine.
before_compute_time = time.time()
gp.compute_model(
    geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype="float32"
    ),
)

# %%
# Plot 2D model visualization
# Visualizing the model in 2D helps in understanding the geological layers.
gpv.plot_2d(geo_model, show_scalar=False)

# %%
# 3D visualization with gempy_viewer
# A 3D view provides a comprehensive look at the model's structure.
gempy_vista = gpv.plot_3d(
    model=geo_model,
    show=True,
    kwargs_plot_structured_grid={'opacity': 0.8}
)

# %%
# Calculate and print execution times
# It's useful to know how long different parts of the process take.
end_time = time.time()
prep_time = before_compute_time - start_time
compute_time = end_time - before_compute_time
execution_time = end_time - start_time

print(f"Preparation time: {prep_time} seconds.")
print(f"Computation time: {compute_time} seconds.")
print(f"Total execution time: {execution_time} seconds.")

# sphinx_gallery_thumbnail_number = -2  
