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

import omfvista
import pandas as pd
import pyvista
from dotenv import dotenv_values
import subsurface
from subsurface import TriSurf, LineSet
from subsurface.modules.visualization import to_pyvista_mesh, to_pyvista_line, init_plotter

# %%
# Load OMF Project
# ----------------
# 
# Load the OMF project using a fixture.

def load_omf():
    config = dotenv_values()
    path = config.get('PATH_TO_MODEL_2')
    omf_project = omfvista.load_project(path)
    return omf_project

omf_project = load_omf()

# %%
# Visualize OMF Project with PyVista (Optional)
# ---------------------------------------------
# 
# Optionally, visualize the OMF project using PyVista. This step can be skipped or modified as needed.

if False:  # Change to True to enable visualization
    omf_project.plot(multi_colors=True, show_edges=True, notebook=False)

# %%
# Convert OMF to Unstructured Single Block
# ----------------------------------------
# 
# Convert the loaded OMF project into an unstructured single block for further analysis.

meshes = []
lines = []

for block_index in range(omf_project.n_blocks):
    block_name = omf_project.get_block_name(block_index)
    polydata_obj: pyvista.PolyData = omf_project[block_name]

    # Skip if the polydata is not a mesh
    if not isinstance(polydata_obj, pyvista.PolyData):
        continue

    unstruct_pyvista: pyvista.UnstructuredGrid = polydata_obj.cast_to_unstructured_grid()
    cell_data = {name: unstruct_pyvista.cell_data[name] for name in unstruct_pyvista.cell_data}

    # Process based on cell type
    match polydata_obj.cell_type(0):
        case pyvista.CellType.TRIANGLE:
            # Process triangle mesh
            cells_pyvista = unstruct_pyvista.cells.reshape(-1, 4)[:, 1:]
            new_cell_data = {"Formation_Major_": block_index, **cell_data}
            unstruct = subsurface.UnstructuredData.from_array(
                vertex=unstruct_pyvista.points,
                cells=cells_pyvista,
                cells_attr=pd.DataFrame(new_cell_data)
            )
            ts = TriSurf(mesh=unstruct)
            meshes.append(to_pyvista_mesh(ts))

        case pyvista.CellType.LINE:
            # Process line data
            if "Formation_Major" not in cell_data.keys():
                continue
            cells_pyvista = unstruct_pyvista.cells.reshape(-1, 3)[:, 1:]
            unstruct = subsurface.UnstructuredData.from_array(
                vertex=unstruct_pyvista.points,
                cells=cells_pyvista,
                cells_attr=pd.DataFrame(cell_data)
            )
            line = LineSet(data=unstruct)
            lines.append(to_pyvista_line(line, radius=100, as_tube=True, spline=False))

# Export to desired format here if necessary

# %%
# Visualize Unstructured Data
# ---------------------------
# 
# Visualize the unstructured data using Subsurface and PyVista.

plotter = init_plotter()

for mesh in meshes:
    plotter.add_mesh(mesh, cmap="magma", opacity=0.7)

for line in lines:
    plotter.add_mesh(line, cmap="viridis", opacity=1)

plotter.show()

