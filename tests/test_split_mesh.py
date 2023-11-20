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


def test_foo():
    unstruct = subsurface.UnstructuredData(dataset)
    
    
    
    ts = subsurface.TriSurf(mesh=unstruct)
    s = subsurface.visualization.to_pyvista_mesh(ts)
    subsurface.visualization.pv_plot([s])
