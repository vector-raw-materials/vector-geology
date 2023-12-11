"""
Reading OMF project and export it to Subsurface: Example 1
==========================================================

This tutorial demonstrates how to read an OMF project file.

"""

# %%
# Required Libraries:
# ~~~~~~~~~~~~~~~~~~~
#
# Import the required libraries. 

import omfvista
import pandas as pd
import pyvista
from dotenv import dotenv_values

import subsurface
from subsurface import TriSurf, LineSet
from subsurface.visualization import to_pyvista_mesh, to_pyvista_line, init_plotter


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

if False:
    omf.plot(multi_colors=True, show_edges=True, notebook=False)

# %%
# Convert OMF to Unstructured Single Block:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Convert the loaded OMF project into an unstructured single block for further analysis.

meshes = []
lines = []
for e in range(omf.n_blocks):

    block_name = omf.get_block_name(e)
    polydata_obj: pyvista.PolyData = omf[block_name]
    # Check if the polydata is a mesh and if is not continue
    print(polydata_obj.cell_type(0))
    unstruct_pyvista: pyvista.UnstructuredGrid = polydata_obj.cast_to_unstructured_grid()

    grid = unstruct_pyvista 
    cell_data = {name: grid.cell_data[name] for name in grid.cell_data}
    match polydata_obj.cell_type(0):
        case pyvista.CellType.TRIANGLE:
            if False: continue # TODO: Remove this
            
            cells_pyvista = unstruct_pyvista.cells.reshape(-1, 4)[:, 1:]
            new_cell_data = {
                **{
                    "Formation_Major_": e,
                },
                **cell_data
            }
            unstruct: subsurface.UnstructuredData = subsurface.UnstructuredData.from_array(
                vertex=unstruct_pyvista.points,
                cells=cells_pyvista,
                cells_attr=pd.DataFrame(new_cell_data)
            )
            
            ts = TriSurf(mesh=unstruct)
            s = to_pyvista_mesh(ts)
            meshes.append(s)

           
        case pyvista.CellType.LINE:
            if "Formation_Major" not in cell_data.keys(): continue
            cells_pyvista = unstruct_pyvista.cells.reshape(-1, 3)[:, 1:]
            unstruct: subsurface.UnstructuredData = subsurface.UnstructuredData.from_array(
                vertex=unstruct_pyvista.points,
                cells=cells_pyvista,
                cells_attr=pd.DataFrame(cell_data)
            )
            line = LineSet(data=unstruct)
            s = to_pyvista_line(line, radius=100, as_tube=True, spline=False)
            
            lines.append(s)

if False:  # Replace with condition for exporting to Liquid Earth
    base_structs_to_binary_file("leapfrog1", unstruct)

# %%
# Visualize Unstructured Data:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use Subsurface and PyVista to visualize the unstructured data.

plotter = init_plotter()
for mesh in meshes[3:]:
    plotter.add_mesh(mesh, cmap="magma", opacity=0.7)

for line in lines:
    plotter.add_mesh(line, cmap="viridis", opacity=1)

plotter.show()



