
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

# %%
# Start the timer to measure the execution time of the script
start_time = time.time()

# %%
# Load environment configuration
# This step reads configurations from a .env file, crucial for setting up file paths.
config = dotenv_values()
path = config.get("PATH_TO_MODEL_1_Subsurface")

# %%
# Initialize structural elements for the geological model
# These will be used to build the model using GemPy.
structural_elements = []
global_extent = None
color_gen = gp.data.ColorsGenerator()

# %%
# Process .nc files for geological model construction
# This loop reads and processes each .nc file to extract necessary data for the model.
for filename in os.listdir(path):
    base, ext = os.path.splitext(filename)
    if ext == '.nc':
        file_path = os.path.join(path, filename)
        structural_element, global_extent = process_file(file_path, global_extent, color_gen)
        structural_elements.append(structural_element)

# %%
# Setup GemPy geological model
# Here, the model is initialized with the processed data.
geo_model = initialize_geo_model(
    structural_elements=structural_elements,
    extent=(np.array(global_extent)),
    topography=(xr.open_dataset(os.path.join(path, "Topography.nc"))),
    load_nuggets=True
)

# %%
# Display the initialized GemPy model
# It's always good practice to verify the model's initialization.
print(geo_model)

# %%
# Read Bouguer gravity data from a CSV file
# This data is used for geophysical calculations later in the script.
df = pd.read_csv(
    filepath_or_buffer=config.get("PATH_TO_MODEL_1_BOUGUER"),
    sep=',',
    header=0
)

# %%
# Filter and prepare the gravity data for further processing
# This step ensures we use the relevant subset of the data.
df = df[df['X'] < 565000]
interesting_columns = df[['X', 'Y', 'Bouguer_267_complete']]

# %%
# Set up interpolation options for the GemPy model
# Configuring interpolation is crucial for accurate geological modeling.
interpolation_options = geo_model.interpolation_options
interpolation_options.mesh_extraction = True
interpolation_options.kernel_options.range = .7
interpolation_options.kernel_options.c_o = 3
interpolation_options.kernel_options.compute_condition_number = True

# %%
# Plot the 2D representation of the geological model with gravity data
# This visualization helps in understanding the spatial distribution of the data.
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

# %%
# Calculate and display the script execution time so far
# Monitoring execution time is useful for performance optimization.
end_time = time.time()
execution_time = end_time - start_time
print(f"The function executed in {execution_time} seconds.")

# %%
# 3D visualization of the geological model
# This 3D view provides a comprehensive perspective of the model's structure.
gempy_vista = gpv.plot_3d(
    model=geo_model,
    show=True,
    kwargs_plot_structured_grid={'opacity': 0.8},
    image=True
)

# %%
# Prepare and set up data for geophysical calculations
# Configuring the data correctly is key for accurate gravity calculations.
device_location = interesting_columns[['X', 'Y']]
device_location['Z'] = 0  # Add a Z-coordinate

# %%
# Set up a centered grid for geophysical calculations
# This grid will be used for gravity gradient calculations.
gp.set_centered_grid(
    grid=geo_model.grid,
    centers=device_location,
    resolution=np.array([10, 10, 15]),
    radius=np.array([5000, 5000, 5000])
)

# %%
# Change backend for GemPy to support tensor operations
# This is necessary for integrating GemPy with PyTorch.
BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH, dtype="float64")

# %%
# Calculate the gravity gradient using GemPy
# Gravity gradient data is critical for geophysical modeling and inversion.
gravity_gradient = gp.calculate_gravity_gradient(geo_model.grid.centered_grid)

# %%
# Define and set up densities tensor for the gravity calculation
# Densities are a fundamental part of the gravity modeling process.
densities_tensor = BackendTensor.t.array([2.61, 2.92, 3.1, 2.92, 2.61, 2.61])
densities_tensor.requires_grad = True

# %%
# Set geophysics input for the GemPy model
# Configuring this input is crucial for the forward gravity calculation.
geo_model.geophysics_input = gp.data.GeophysicsInput(
    tz=BackendTensor.t.array(gravity_gradient),
    densities=densities_tensor
)

# %%
# Compute the geological model with geophysical data
# This computation integrates the geological model with gravity data.
sol = gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype='float64'
    )
)
grav = - sol.gravity
grav[0].backward()

# %%
# Output gradient information for analysis
# The gradient data can provide insights into the density distribution.
print(densities_tensor.grad)

# %%
# Perform scale and shift calculations on the gravity data
# These calculations align the model's gravity data with observed values.
s, c = calculate_scale_shift(
    a=interesting_columns["Bouguer_267_complete"].values,
    b=(grav.detach().numpy())
)

# %%
# Display the calculated scale and shift values
# Understanding these values is important for interpreting the results.
print("Scale (s):", s)
print("Shift (c):", c)

# %%
# Adapt the gravity data based on scale and shift calculations
# This step adjusts the model's gravity data to match observed values.
adapted_grav = s * interesting_columns["Bouguer_267_complete"] + c
diff = adapted_grav - grav.detach().numpy()

# %%
# Visualization of adapted gravity data
# This visualization helps in comparing the model's gravity data with observations.
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

# %%
# Calculate symmetric vmin and vmax for the colorbar in the difference plot
# This step ensures a balanced color representation of positive and negative differences.
max_diff = np.max(np.abs(diff))  # Get the maximum absolute value from diff
vmin, vmax = -max_diff, max_diff  # Set vmin and vmax

# %%
# Plotting the difference between adapted and computed gravity data
# This plot highlights the discrepancies between the model and observed data.
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
