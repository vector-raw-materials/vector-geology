
"""
Reading Model 1 OMF Project into Subsurface
===========================================

This tutorial demonstrates how to read an OMF (Open Mining Format) project file and convert it into a Subsurface format for enhanced analysis and visualization.
"""
# %%
# Importing Required Libraries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Begin by importing the necessary libraries for this tutorial.

import pandas as pd
import pyvista
import xarray

import subsurface
from subsurface import TriSurf, LineSet
from subsurface.modules.visualization import to_pyvista_mesh, to_pyvista_line, init_plotter
from vector_geology.utils import load_omf

# %%
# Loading the OMF Project
# ~~~~~~~~~~~~~~~~~~~~~~~
# Load the OMF project file using a custom function.

omf = load_omf("PATH_TO_MODEL_1")

# %%
# Visualizing the OMF Project with PyVista
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next, we use PyVista for an initial visualization of the OMF project.

# Replace `False` with a condition or toggle to enable plotting.
if False:
    omf.plot(multi_colors=True, show_edges=True, notebook=False)

# %%
# Visualizing Unstructured Data with Subsurface and PyVista
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use Subsurface and PyVista to visualize the unstructured data.


meshes_far = []

meshes = []
lines_1 = []
lines_far = []

dataset: xarray.Dataset = None

for e in range(omf.n_blocks):
    block_name = omf.get_block_name(e)
    print(block_name)
    polydata_obj: pyvista.PolyData = omf[block_name]
    # Check if the polydata is a mesh and if is not continue
    print(polydata_obj.get_cell(0).type)
    unstruct_pyvista: pyvista.UnstructuredGrid = polydata_obj.cast_to_unstructured_grid()

    grid = unstruct_pyvista
    cell_data = {name: grid.cell_data[name] for name in grid.cell_data}
    match polydata_obj.get_cell(0).type:
        case pyvista.CellType.TRIANGLE:
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
            if e == 5:
                meshes_far.append(s)  # * This mesh is far from the rest. I am still unsure what is meant to represent.
            else:
                if False:
                    to_netcdf(
                        base_data=unstruct,
                        path=f"./{block_name}.nc",
                    )
                meshes.append(s)
                

        case pyvista.CellType.LINE:
            if e > 11: continue
            continue # To ignore wells for now
            cells_pyvista = unstruct_pyvista.cells.reshape(-1, 3)[:, 1:]
            unstruct: subsurface.UnstructuredData = subsurface.UnstructuredData.from_array(
                vertex=unstruct_pyvista.points,
                cells=cells_pyvista,
                cells_attr=pd.DataFrame(cell_data)
            )
            line = LineSet(data=unstruct)
            s = to_pyvista_line(line, radius=100, as_tube=True, spline=False)
            if e <= 10:
                lines_1.append(s)
            elif e <= 11:
                lines_far.append(s)

if False:  # Replace with condition for exporting to Liquid Earth
    base_structs_to_binary_file("leapfrog1", unstruct)

# %%
# Visualize Unstructured Data:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use Subsurface and PyVista to visualize the unstructured data.

plotter = init_plotter()
if plot_model_area := True:
    for mesh in meshes[:1]:
        plotter.add_mesh(mesh, cmap="magma", opacity=1)

    for line in lines_far:
        plotter.add_mesh(line, cmap="viridis", opacity=1)
else:
    # * This seems to go together
    for mesh in meshes_far:
        plotter.add_mesh(mesh, cmap="magma", opacity=0.7)

    for line in lines_1:
        plotter.add_mesh(line, cmap="viridis", opacity=1)

plotter.show()

# %% 
# Conclusions
# ~~~~~~~~~~~
# It seems that there are two areas in the OMF but the second one does not match the Legacy report.
# On the second area, wells do not have lithological data so there is not much we can do with it.
# For now we will use the interpreted meshes to reconstruct the gempy model
# Lithology data and gravity seems to be confidential so how much we can share in this documentation will be limited.
