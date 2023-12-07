"""
Stonepark Geological Model
--------------------------  


Construct a 3D geological model of the Stonepark deposit using GemPy.


"""

import time

import xarray as xr

from vector_geology.stonepark_builder import initialize_geo_model

# %%
# Read nc from subsurface


# %%
import os
from dotenv import dotenv_values

from vector_geology.omf_to_gempy import process_file
import gempy as gp
import gempy_viewer as gpv

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
    extent=global_extent,
    topography=(xr.open_dataset(os.path.join(path, "Topography.nc"))),
    load_nuggets=True
)

# %%

geo_model

# %% 
interpolation_options = geo_model.interpolation_options

interpolation_options.mesh_extraction = True
interpolation_options.kernel_options.range = .7
interpolation_options.kernel_options.c_o = 3
interpolation_options.kernel_options.compute_condition_number = True

# %% 
gp.modify_surface_points(
    geo_model,
    slice=0,
    X = geo_model.surface_points.data[0][0] + 130,
)

before_compute_time = time.time()
gp.compute_model(
    geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype="float32"
    ),
)

gpv.plot_2d(geo_model, show_scalar=False)

end_time = time.time()
prep_time = before_compute_time - start_time
compute_time = end_time - before_compute_time
execution_time = end_time - start_time

print(f"The function executed in {prep_time} seconds.")
print(f"The function executed in {compute_time} seconds.")
print(f"The function executed in {execution_time} seconds.")

gempy_vista = gpv.plot_3d(
    model=geo_model,
    show=True,
    kwargs_plot_structured_grid={'opacity': 0.8}
)
