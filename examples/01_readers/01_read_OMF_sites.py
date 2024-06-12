"""
Reading OMF Project Into Python
===============================

This tutorial demonstrates how to read an OMF (Open Mining Format) project file in Python, visualize it using PyVista, and convert it to a format suitable for further analysis with Subsurface.
"""

# %%
# Required Libraries
# ------------------
# First, we import the necessary libraries for handling and visualizing OMF files.

import omfvista
import pyvista
import subsurface
from subsurface import TriSurf
from dotenv import dotenv_values

from subsurface.modules.visualization import to_pyvista_mesh, pv_plot
from subsurface.modules.writer import base_structs_to_binary_file


# %%
# Load OMF Project
# ----------------
# Here, we define a function to load an OMF project using a path specified in a .env file.

def load_omf():
    config = dotenv_values()
    path = config.get('PATH_TO_MODEL_2')
    omf_project = omfvista.load_project(path)
    return omf_project

omf_project = load_omf()

# %%
# Visualize OMF with PyVista
# --------------------------
# Utilize PyVista for an interactive visualization of the OMF project.

omf_project.plot(multi_colors=True, show_edges=True, notebook=False)

# %%
# Convert OMF to Unstructured Single Block
# ----------------------------------------
# Convert the loaded OMF project into an unstructured single block for further processing.

block_name = omf_project.get_block_name(4)
polydata_obj: pyvista.PolyData = omf_project[block_name]
unstruct_pyvista: pyvista.UnstructuredGrid = polydata_obj.cast_to_unstructured_grid()
cells_pyvista = unstruct_pyvista.cells.reshape(-1, 4)[:, 1:]

unstruct: subsurface.UnstructuredData = subsurface.UnstructuredData.from_array(
    vertex=unstruct_pyvista.points,
    cells=cells_pyvista,
)

# Optional: Export to Liquid Earth if required
TO_LIQUID_EARTH = False  # Replace with actual condition
if TO_LIQUID_EARTH:
    base_structs_to_binary_file("leapfrog1", unstruct)

# %%
# Visualize Unstructured Data
# ---------------------------
# Finally, visualize the converted unstructured data using Subsurface and PyVista.

ts = TriSurf(mesh=unstruct)
subsurface_mesh = to_pyvista_mesh(ts)
pv_plot([subsurface_mesh], image_2d=False)
