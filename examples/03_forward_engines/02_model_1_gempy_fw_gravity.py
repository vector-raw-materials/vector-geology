"""
Model 1 Forward Gravity
-----------------------  

This script demonstrates the calculation of forward gravity for model 1.
It utilizes libraries such as GemPy, NumPy, and others to handle and process geophysical data.
"""

import os
import time
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from dotenv import dotenv_values
import gempy as gp
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor
from vector_geology.bayesian_helpers import calculate_scale_shift
from vector_geology.model_1_builder import initialize_geo_model
from vector_geology.omf_to_gempy import process_file
from vector_geology.utils import extend_box

# Start the timer for execution time tracking
start_time = time.time()

# Load configuration from .env file
config = dotenv_values()
path = config.get("PATH_TO_MODEL_1_Subsurface")

# Initialize variables
structural_elements = []
accumulated_roi = []
global_extent = None
color_gen = gp.data.ColorsGenerator()

# Process .nc files in the specified directory
for filename in os.listdir(path):
    base, ext = os.path.splitext(filename)
    if ext == '.nc':
        file_path = os.path.join(path, filename)
        structural_element, global_extent = process_file(file_path, global_extent, color_gen)
        structural_elements.append(structural_element)

# Setup GemPy model
geo_model = initialize_geo_model(
    structural_elements=structural_elements,
    extent=(np.array(global_extent)),
    topography=(xr.open_dataset(os.path.join(path, "Topography.nc"))),
    load_nuggets=True
)

# Display the GeoModel object
print(geo_model)

# Read Bouguer gravity data from CSV
df = pd.read_csv(
    filepath_or_buffer=config.get("PATH_TO_MODEL_1_BOUGUER"),
    sep=',',
    header=0
)

# Filter data based on X coordinate
df = df[df['X'] < 565000]
interesting_columns = df[['X', 'Y', 'Bouguer_267_complete']]

# Set up interpolation options for the model
interpolation_options = geo_model.interpolation_options
interpolation_options.mesh_extraction = True
interpolation_options.kernel_options.range = .7
interpolation_options.kernel_options.c_o = 3
interpolation_options.kernel_options.compute_condition_number = True

# Plot the 2D representation of the model
plot2d = gpv.plot_2d(geo_model, show_topography=True, section_names=["topography"], show=False)
plot2d.axes[0].scatter(
    interesting_columns['X'],
    interesting_columns['Y'],
    c=interesting_columns['Bouguer_267_complete'],
    cmap='viridis',
    s=100,
    zorder=10000
)
plot2d.fig.show()

# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"The function executed in {execution_time} seconds.")

# 3D Visualization
gempy_vista = gpv.plot_3d(
    model=geo_model,
    show=True,
    kwargs_plot_structured_grid={'opacity': 0.8},
    image=True
)

# Prepare data for geophysical calculations
device_location = interesting_columns[['X', 'Y']]
device_location['Z'] = 0  # stack 0 to the z-axis

# Set up a centered grid for the calculations
gp.set_centered_grid(
    grid=geo_model.grid,
    centers=device_location,
    resolution=np.array([10, 10, 15]),
    radius=np.array([5000, 5000, 5000])
)

# Change backend for GemPy
BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH, dtype="float64")

# Calculate gravity gradient
gravity_gradient = gp.calculate_gravity_gradient(geo_model.grid.centered_grid)

# Define densities tensor for the calculation
densities_tensor = BackendTensor.t.array([2.61, 2.92, 3.1, 2.92, 2.61, 2.61])
densities_tensor.requires_grad = True

# Set geophysics input for the model
geo_model.geophysics_input = gp.data.GeophysicsInput(
    tz=BackendTensor.t.array(gravity_gradient),
    densities=densities_tensor
)

# Compute the model with geophysical data
sol = gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype='float64'
    )
)
grav = - sol.gravity
grav[0].backward()

# Output gradient information
print(densities_tensor.grad)

# Scale and shift calculations
s, c = calculate_scale_shift(
    a=interesting_columns["Bouguer_267_complete"].values,
    b=(grav.detach().numpy())
)

# Display scale and shift values
print("Scale (s):", s)
print("Shift (c):", c)

# Adapt gravity data
adapted_grav = s * interesting_columns["Bouguer_267_complete"] + c
diff = adapted_grav - grav.detach().numpy()

# Visualization of adapted gravity data
plot2d = gpv.plot_2d(geo_model, show_topography=True, section_names=["topography"], show=False)
plot2d.axes[0].scatter(
    interesting_columns['X'],
    interesting_columns['Y'],
    c=grav.detach().numpy(),
    cmap='viridis',
    s=100,
    zorder=10000
)
plt.show()

# Calculate symmetric vmin and vmax for the colorbar
max_diff = np.max(np.abs(diff))  # Get the maximum absolute value from diff
vmin, vmax = -max_diff, max_diff  # Set vmin and vmax

# Plotting the difference
plot2d = gpv.plot_2d(geo_model, show_topography=True, section_names=["topography"], show=False)
sc = plot2d.axes[0].scatter(
    interesting_columns['X'],
    interesting_columns['Y'],
    c=diff,
    cmap='bwr',
    s=100,
    zorder=10000,
    vmin=vmin,
    vmax=vmax
)
plt.colorbar(sc, label="Difference (mGal)")
plt.show()

# sphinx_gallery_thumbnail_number = -1
