"""
Reading OMF project and exporting it to Subsurface: Example 1
==============================================================

This tutorial demonstrates how to read an OMF project file and export it for use with Subsurface.

"""

# %%
# Import Required Libraries
# -------------------------
# 
# Import the necessary libraries for reading an OMF file and processing it.

import os

import dotenv

import subsurface
from subsurface import StructuredGrid
from subsurface.modules.reader.volume.read_volume import read_VTK_structured_grid
from subsurface.modules.visualization import init_plotter, to_pyvista_grid
from vector_geology.model_contructor.spremberg_reader import load_spremberg_meshes, process_borehole_data, read_topography, read_seismic_profiles, read_magnetic_profiles

dotenv.load_dotenv()
meshes, lines = load_spremberg_meshes(os.getenv('PATH_TO_SPREMBERG_OMF'))

well_mesh, collars = process_borehole_data(
    path_to_stratigraphy=(os.getenv("PATH_TO_SPREMBERG_STRATIGRAPHY")),
    path_to_survey=os.getenv("PATH_TO_SPREMBERG_SURVEY"),
    path_to_collar=os.getenv("PATH_TO_SPREMBERG_COLLAR")
)

s1 = read_topography(os.getenv("PATH_TO_TOPOGRAPHY"))

seismic_profiles = read_seismic_profiles(
    interpretation_path=os.getenv("PATH_TO_SEISMIC_INTERPRETATION"),
    section_path=os.getenv("PATH_TO_SEISMIC_SECTION"),
    crop_coords={
        "x_start": 60,
        "x_end": 2080,
        "y_start": 83,
        "y_end": 730
    },
    zmin=-450,
    zmax=140,
)

magnetic_profiles = read_magnetic_profiles(
    interpretation_path=os.getenv("PATH_TO_MAGNETIC_INTERPRETATION"),
    section_path=os.getenv("PATH_TO_MAGNETIC_SECTION"),
    crop_coords={
        "x_start": 250,
        "x_end": 1535,
        "y_start": 120,
        "y_end": 1495
    },
    zmin=-2500,
    zmax=500,
    profile_number=1
)

structured_data: subsurface.StructuredData = read_VTK_structured_grid(
    file_or_buffer= os.getenv("PATH_TO_SPREMBERG_FAKE_GEOPHYSICS"),
    active_scalars="model_name"
)

sg: subsurface.StructuredGrid = StructuredGrid(structured_data)
pyvista_structured_data = to_pyvista_grid(sg)

# %%
# Visualize Unstructured Data
# ---------------------------
# 
# Visualize the unstructured data using Subsurface and PyVista.

plotter = init_plotter()

if True:
    for mesh in meshes:
        plotter.add_mesh(mesh, cmap="magma", opacity=0.1)

for line in lines:
    plotter.add_mesh(line, cmap="viridis", opacity=1)

    file_or_buffer = os.getenv("PATH_TO_SPREMBERG_STRATIGRAPHY"),

# Initialize the PyVista plotter.
pyvista_plotter = plotter

# Define the units limit for thresholding the well mesh.
units_limit = [0, 20]

# Add the well mesh and collars to the plotter and display.
pyvista_plotter.add_mesh(
    well_mesh.threshold(units_limit),
    cmap="tab20c",
    clim=units_limit
)

pyvista_plotter.add_mesh(
    collars,
    point_size=10,
    render_points_as_spheres=True
)


pyvista_plotter.add_mesh(s1, opacity=1)

pyvista_plotter.add_mesh(pyvista_structured_data, opacity=.5)

pyvista_plotter.add_mesh(
    mesh=seismic_profiles,
    texture=seismic_profiles._textures.get(0, None),
    opacity=1
)

pyvista_plotter.add_mesh(
    mesh=magnetic_profiles,
    texture=magnetic_profiles._textures.get(0, None),
    opacity=1
)

pyvista_plotter.show()
