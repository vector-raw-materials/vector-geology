"""
Reading COLLINSTOWN OMF project
================================

This tutorial demonstrates how to read an OMF project file in COLLINSTOWN.

"""

# %%
# Required Libraries:
# ~~~~~~~~~~~~~~~~~~~
#
# Import the required libraries. 

import omfvista
import pyvista
import subsurface
from subsurface import TriSurf
from subsurface.visualization import to_pyvista_mesh, pv_plot
from subsurface.writer import base_structs_to_binary_file
from dotenv import dotenv_values

# %%
# Load OMF Project:
# ~~~~~~~~~~~~~~~~~
#
# Load the OMF project using a fixture.

def load_omf():
    config = dotenv_values()
    path = config.get('PATH_TO_MODEL_2')
    omf = omfvista.load_project(path)
    return omf

omf = load_omf()

# %%
# Read OMF with PyVista:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Visualize the OMF project with PyVista.

omf.plot(multi_colors=True, show_edges=True, notebook=False)

# %%
# Convert OMF to Unstructured Single Block:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Convert the loaded OMF project into an unstructured single block for further analysis.

block_name = omf.get_block_name(4)
polydata_obj: pyvista.PolyData = omf[block_name]
unstruct_pyvista: pyvista.UnstructuredGrid = polydata_obj.cast_to_unstructured_grid()
cells_pyvista = unstruct_pyvista.cells.reshape(-1, 4)[:, 1:]

unstruct: subsurface.UnstructuredData = subsurface.UnstructuredData.from_array(
    vertex=unstruct_pyvista.points,
    cells=cells_pyvista,
)

if TO_LIQUID_EARTH := False:  # Replace with condition for exporting to Liquid Earth
    base_structs_to_binary_file("leapfrog1", unstruct)

# %%
# Visualize Unstructured Data:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Use Subsurface and PyVista to visualize the unstructured data.

ts = TriSurf(mesh=unstruct)
s = to_pyvista_mesh(ts)
pv_plot([s], image_2d=False)

