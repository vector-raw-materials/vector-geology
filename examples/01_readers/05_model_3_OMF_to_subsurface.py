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
import omfvista
import pandas as pd
import pyvista
from dotenv import dotenv_values
import subsurface
from subsurface import TriSurf, LineSet
from subsurface.modules.visualization import to_pyvista_mesh, to_pyvista_line, init_plotter
from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.modules.reader.wells.read_borehole_interface import read_lith, read_survey, read_collar
from subsurface.modules.visualization import to_pyvista_line, to_pyvista_points, init_plotter
from subsurface.modules.reader import read_unstructured_topography, read_structured_topography
from subsurface.modules.reader.mesh.dxf_reader import DXFEntityType

# %%
# Load OMF Project
# ----------------
# 
# Load the OMF project using a fixture.

def load_omf():
    config = dotenv_values()
    path = config.get('PATH_TO_SPREMBERG_OMF')
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
    match polydata_obj.get_cell(0).type:
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
    plotter.add_mesh(mesh, cmap="magma", opacity=0.3)

for line in lines:
    plotter.add_mesh(line, cmap="viridis", opacity=1)




# %% [markdown]
# Initialize the reader for the lithological data. Specify the file path and column mappings.
import dotenv
dotenv.load_dotenv()
reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
    file_or_buffer=os.getenv("PATH_TO_SPREMBERG_STRATIGRAPHY"),
    columns_map={
            'hole_id'   : 'id',
            'depth_from': 'top',
            'depth_to'  : 'base',
            'lit_code'  : 'component lith'
    }
)

# Read the lithological data into a DataFrame.
lith: pd.DataFrame = read_lith(reader)

# %% [markdown]
# Initialize the reader for the survey data. Specify the file path and column mappings.
reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
    file_or_buffer=os.getenv("PATH_TO_SPREMBERG_SURVEY"),
    columns_map={
            'depth'  : 'md',
            'dip'    : 'dip',
            'azimuth': 'azi'
    },
)

# Read the survey data into a DataFrame.
df = read_survey(reader)

# %% [markdown]
# Create a Survey object from the DataFrame and update it with lithological data.
survey: Survey = Survey.from_df(
    survey_df=df,
    attr_df=None
)
survey.update_survey_with_lith(lith)

# %% [markdown]
# Initialize the reader for the collar data. Specify the file path, header, and column mappings.
reader_collar: GenericReaderFilesHelper = GenericReaderFilesHelper(
    file_or_buffer=os.getenv("PATH_TO_SPREMBERG_COLLAR"),
    header=0,
    usecols=[0, 1, 2, 4],
    columns_map={
            "hole_id"            : "id",
            "X_GK5_incl_inserted": "x",
            "Y__incl_inserted"   : "y",
            "Z_GK"               : "z"
    }
)

# Read the collar data into a DataFrame and create a Collars object.
df_collar = read_collar(reader_collar)
collar = Collars.from_df(df_collar)

# %% [markdown]
# Combine the collar and survey data into a BoreholeSet.
borehole_set = BoreholeSet(
    collars=collar,
    survey=survey,
    merge_option=MergeOptions.INTERSECT
)

# %% [markdown]
# Visualize the borehole trajectories and collars using PyVista.
well_mesh = to_pyvista_line(
    line_set=borehole_set.combined_trajectory,
    active_scalar="lith_ids",
    radius=10
)

collars = to_pyvista_points(borehole_set.collars.collar_loc)

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

file_path = os.getenv("PATH_TO_TOPOGRAPHY")
unstruct = read_unstructured_topography(
    path=file_path,
    additional_reader_kwargs={'entity_type': DXFEntityType.POLYLINE}
)
ts = TriSurf(mesh=unstruct)
s1 = to_pyvista_mesh(ts)

pyvista_plotter.add_mesh(s1,  opacity=1)

pyvista_plotter.show()
