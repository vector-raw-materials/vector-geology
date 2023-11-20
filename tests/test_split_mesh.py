import numpy as np
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
    triangulated_mesh = subsurface.visualization.to_pyvista_mesh(ts)
    # Decimate the mesh to reduce the number of points
    decimated_mesh = triangulated_mesh.decimate_pro(0.95)

    # Compute normals
    normals = decimated_mesh.compute_normals(point_normals=False, cell_normals=True, consistent_normals=True)

    normals_array = np.array(normals.cell_data["Normals"])

    # Extract the points and normals from the decimated mesh
    sampled_points = normals.cell_centers().points
    sampled_normals = normals_array

    subsurface.visualization.pv_plot([triangulated_mesh, sampled_normals])
