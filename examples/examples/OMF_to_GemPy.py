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
    path = config.get('PATH_TO_COLLINSTOWN')
    omf = omfvista.load_project(path)
    return omf

omf = load_omf()

# %%
# Read OMF with PyVista:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Visualize the OMF project with PyVista.

if False:
    omf.plot(multi_colors=True, show_edges=True, notebook=False)

# %%
# Convert OMF to Unstructured Single Block:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Convert the loaded OMF project into an unstructured single block for further analysis.

meshes = []
for e in range(omf.n_blocks):

    block_name = omf.get_block_name(e)
    polydata_obj: pyvista.PolyData = omf[block_name]
    # Check if the polydata is a mesh and if is not continue
    print(polydata_obj.cell_type(0))
    if polydata_obj.cell_type(0) != pyvista.CellType.TRIANGLE:
        continue
        
    unstruct_pyvista: pyvista.UnstructuredGrid = polydata_obj.cast_to_unstructured_grid()
    cells_pyvista = unstruct_pyvista.cells.reshape(-1, 4)[:, 1:]

    unstruct: subsurface.UnstructuredData = subsurface.UnstructuredData.from_array(
        vertex=unstruct_pyvista.points,
        cells=cells_pyvista,
    )

    ts = TriSurf(mesh=unstruct)
    s = to_pyvista_mesh(ts)
    
    meshes.append(s)
    
if False:  # Replace with condition for exporting to Liquid Earth
    base_structs_to_binary_file("leapfrog1", unstruct)

# %%
# Visualize Unstructured Data:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use Subsurface and PyVista to visualize the unstructured data.

import random
def random_color_generator():
    while True:
        yield (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Usage Example:

color_gen = random_color_generator()

pv_plot(
    meshes, 
    image_2d=False, 
    add_mesh_kwargs={ }
)


