import os

import dotenv
import numpy as np
import pandas as pd

from subsurface import TriSurf, UnstructuredData, PointSet
from subsurface.modules.reader import read_unstructured_topography, read_structured_topography
from subsurface.modules.reader.mesh.dxf_reader import DXFEntityType
from subsurface.modules.visualization import to_pyvista_mesh, pv_plot, to_pyvista_points

dotenv.load_dotenv()


def test_read_topography_I():
    file_path = os.getenv("PATH_TO_TOPOGRAPHY")
    unstruct = read_unstructured_topography(
        path=file_path,
        additional_reader_kwargs={'entity_type': DXFEntityType.POLYLINE}
    )
    ts = TriSurf(mesh=unstruct)
    s = to_pyvista_mesh(ts)
    pv_plot([s], image_2d=False)


def test_read_sensors():
    file_path = os.getenv("PATH_TO_SENSORS")
    # 1. Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path)

    # 2. Select the columns for X, Y, and Z — in your case, "Local Easting", "Local Northing", and "DEM Height"
    xyz_columns = ["Local Easting", "Local Northing", "DEM Height"]
    xyz_df = df[xyz_columns]

    # remove rows with NaN values
    xyz_df = xyz_df.dropna()

    # 3. Convert the selected columns to a NumPy array
    xyz_array = xyz_df.to_numpy()

    print(xyz_array)

    # 4. Create a PointSet object
    unstruc = UnstructuredData.from_array(
        vertex=xyz_array,
        cells="points"
    )

    point_set = PointSet(data=unstruc)
    s = to_pyvista_points(point_set)
    pv_plot(
        meshes=[s],
        image_2d=False,
        add_mesh_kwargs={
                "point_size"              : 10,
                "render_points_as_spheres": True
        }
    )


def test_read_section():
    file_path = os.getenv("PATH_TO_SECTION")
    df = pd.read_csv(
        filepath_or_buffer=file_path,
        skiprows=4,  # Skip the header lines above 'CDP'
        delim_whitespace=True,  # Treat consecutive spaces as separators
        names=["CDP", "X_COORD", "Y_COORD"]  # Assign column names
    )

    # Extract only the X and Y columns (ignoring the CDP column)
    xy_array = df[["X_COORD", "Y_COORD"]].to_numpy()
    
    # Add a Z column with zeros
    Z = 0
    xyz_array = np.column_stack((xy_array, Z * np.ones(xy_array.shape[0])))
    unstruct = UnstructuredData.from_array(
        vertex=xyz_array,
        cells="points"
    )
    
    point_set = PointSet(data=unstruct)
    s = to_pyvista_points(point_set)
    pv_plot(
        meshes=[s],
        image_2d=False,
        add_mesh_kwargs={
                "point_size"              : 10,
                "render_points_as_spheres": True
        }
    )
    

    print(xy_array)
