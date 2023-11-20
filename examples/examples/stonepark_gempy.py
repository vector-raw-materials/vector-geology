"""
Construct a 3D geological model of the Stonepark deposit using GemPy.


"""

# %%
# Read nc from subsurface


# %%
import xarray as xr
import os
from dotenv import dotenv_values

import subsurface

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

gempy_vista = gpv.plot_3d(geo_model, show=False)
unstruct = subsurface.UnstructuredData(dataset)
ts = subsurface.TriSurf(mesh=unstruct)
s = subsurface.visualization.to_pyvista_mesh(ts)
gempy_vista.p.add_mesh(s, color="red", opacity=0.5)
gempy_vista.p.show()
