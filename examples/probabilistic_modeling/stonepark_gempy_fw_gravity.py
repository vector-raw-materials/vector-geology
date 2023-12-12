"""
Stonepark Geological Model
--------------------------  


Construct a 3D geological model of the Stonepark deposit using GemPy.


"""

import time

import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt

from gempy_engine.core.backend_tensor import BackendTensor
from vector_geology.bayesian_helpers import calculate_scale_shift
from vector_geology.model_1_builder import initialize_geo_model

# %%
# Read nc from subsurface


# %%
import os
from dotenv import dotenv_values

from vector_geology.omf_to_gempy import process_file
import gempy as gp
import gempy_viewer as gpv
from vector_geology.utils import extend_box

start_time = time.time()  # start timer
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
geo_model = initialize_geo_model(
    structural_elements=structural_elements,
    extent=(np.array(global_extent)),
    topography=(xr.open_dataset(os.path.join(path, "Topography.nc"))),
    load_nuggets=True
)

# %%

geo_model

# %% 
df = pd.read_csv(
    filepath_or_buffer=config.get("PATH_TO_STONEPARK_BOUGUER"),
    sep=',',
    header=0
)

# Remove the items that have X > 5650000
df = df[df['X'] < 565000]

interesting_columns = df[['X', 'Y', 'Bouguer_267_complete']]

# %% 
interpolation_options = geo_model.interpolation_options

interpolation_options.mesh_extraction = True
interpolation_options.kernel_options.range = .7
interpolation_options.kernel_options.c_o = 3
interpolation_options.kernel_options.compute_condition_number = True


gpv.plot_2d(geo_model, show_scalar=False)
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

end_time = time.time()
execution_time = end_time - start_time

print(f"The function executed in {execution_time} seconds.")

gempy_vista = gpv.plot_3d(
    model=geo_model,
    show=True,
    kwargs_plot_structured_grid={'opacity': 0.8},
    image=True
)



# %%
device_location = interesting_columns[['X', 'Y']]
# stack 0 to the z axis
device_location['Z'] = 0

gp.set_centered_grid(
    grid=geo_model.grid,
    centers=device_location,
    resolution=np.array([10, 10, 15]),
    radius=np.array([5000, 5000, 5000])
)

BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH, dtype="float64")
gravity_gradient = gp.calculate_gravity_gradient(geo_model.grid.centered_grid)
gravity_gradient

# %%
densities_tensor = BackendTensor.t.array([2.61, 2.92, 3.1, 2.92, 2.61, 2.61])
densities_tensor.requires_grad = True
geo_model.geophysics_input = gp.data.GeophysicsInput(
    tz=BackendTensor.t.array(gravity_gradient),
    densities=densities_tensor
)

# %% 
sol = gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype='float64'
    )
)
grav = - sol.gravity

grav[0].backward()
print(densities_tensor.grad)

# %%
# Your tensors

# Use the function
s, c = calculate_scale_shift(
    a=interesting_columns["Bouguer_267_complete"].values,
    b=(grav.detach().numpy())
)

print("Scale (s):", s)
print("Shift (c):", c)

adapted_grav = s * interesting_columns["Bouguer_267_complete"] + c

diff = adapted_grav - grav.detach().numpy().detach().numpy()

# %%
# TODO: Scale the gravity data to the same scale as the model

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
plt.colorbar(sc, label="Difference (mGal)" )
plt.show()

