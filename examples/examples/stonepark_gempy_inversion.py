"""
Stonepark Geological Model
--------------------------  


Construct a 3D geological model of the Stonepark deposit using GemPy.


"""

import time

import numpy as np
import xarray as xr
import pandas as pd

from vector_geology.stonepark_builder import initialize_geo_model

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

#  %%j
# Setup gempy object
extent___ = np.array(global_extent)
# extent___[1] = 570000 # move the extent far enough to include the last device
geo_model = initialize_geo_model(
    structural_elements=structural_elements,
    extent=extent___,
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

interesting_columns = df[['X', 'Y', 'Bouguer_267_complete']]

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
    kwargs_plot_structured_grid={'opacity': 0.8}
)
