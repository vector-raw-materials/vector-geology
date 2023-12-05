"""
Stonepark Geological Model
--------------------------  


Construct a 3D geological model of the Stonepark deposit using GemPy.


"""

import time

from vector_geology.stonepark_builder import initialize_geo_model, optimize_nuggets_for_whole_project, apply_optimized_nuggets

start_time = time.time()  # start timer

import numpy as np
# %%
# Read nc from subsurface


# %%
import os
from dotenv import dotenv_values

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

# %% 
# Add topography
import xarray as xr
dataset: xr.Dataset = xr.open_dataset(os.path.join(path, "Topography.nc"))

geo_model = initialize_geo_model(structural_elements, global_extent)

# %%

geo_model

# %% 
interpolation_options = geo_model.interpolation_options

interpolation_options.mesh_extraction = True
interpolation_options.kernel_options.range = .7
interpolation_options.kernel_options.c_o = 3
interpolation_options.kernel_options.compute_condition_number = True

# %% 
gp.compute_model(
    geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype="float64"
    ),
)

gpv.plot_2d(geo_model, show_scalar=False)

end_time = time.time()
execution_time = end_time - start_time

print(f"The function executed in {execution_time} seconds.")

gempy_vista = gpv.plot_3d(
    model=geo_model,
    show=True,
    kwargs_plot_structured_grid={'opacity': 0.8}
)