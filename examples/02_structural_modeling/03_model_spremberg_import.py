"""
Construct Spremberg: Importing borehole data
--------------------------------------------

This example demonstrates how to construct a 3D geological model of the Model 1 deposit using GemPy. 
It leverages custom APIs to streamline the modeling process.
"""
# %% [markdown]
# Import the necessary libraries for geological modeling and visualization.
import os
import pandas as pd
import pyvista

import gempy as gp
import gempy_viewer as gpv
from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.modules.reader.wells.read_borehole_interface import read_lith, read_survey, read_collar
from subsurface.modules.visualization import to_pyvista_line, to_pyvista_points, init_plotter

# %% [markdown]
# Initialize the reader for the lithological data. Specify the file path and column mappings.
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
survey: Survey = Survey.from_df(df)
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
    radius=2
)

collars = to_pyvista_points(
    borehole_set.collars.collar_loc,
)

# Initialize the PyVista plotter.
pyvista_plotter = init_plotter()

# Define the units limit for thresholding the well mesh.
units_limit = [-1, 32]

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

pyvista_plotter.show()

# %% [markdown]
# Create structural elements from the borehole set for different lithological units.
elements: list[gp.data.StructuralElement] = gp.structural_elements_from_borehole_set(
    borehole_set=borehole_set,
    elements_dict={
            "Buntsandstein"       : {
                    "id"   : 53_300,
                    "color": "#983999"
            },
            "Werra-Anhydrit"      : {
                    "id"   : 61_730,
                    "color": "#00923f"
            },
            "Kupfershiefer"       : {
                    "id"   : 61_760,
                    "color": "#da251d"
            },
            "Zechsteinkonglomerat": {
                    "id"   : 61_770,
                    "color": "#f8c300"
            },
            "Rotliegend"          : {
                    "id"   : 62_000,
                    "color": "#bb825b"
            }
    }
)

# %% [markdown]
# Group the structural elements into a StructuralGroup and create a StructuralFrame.
group = gp.data.StructuralGroup(
    name="Stratigraphic Pile",
    elements=elements,
    structural_relation=gp.data.StackRelationType.ERODE
)
structural_frame = gp.data.StructuralFrame(
    structural_groups=[group],
    color_gen=gp.data.ColorsGenerator()
)

# %% [markdown]
# Determine the extent of the geological model from the surface points coordinates.
all_surface_points_coords: gp.data.SurfacePointsTable = structural_frame.surface_points_copy
extent_from_data = all_surface_points_coords.xyz.min(axis=0), all_surface_points_coords.xyz.max(axis=0)

# %% [markdown]
# Create a GeoModel with the specified extent, grid resolution, and interpolation options.
geo_model = gp.data.GeoModel(
    name="Stratigraphic Pile",
    structural_frame=structural_frame,
    grid=gp.data.Grid(
        extent=[extent_from_data[0][0], extent_from_data[1][0], extent_from_data[0][1], extent_from_data[1][1], extent_from_data[0][2], extent_from_data[1][2]],
        resolution=(50, 50, 50)
    ),
    interpolation_options=gp.data.InterpolationOptions(
        range=5,
        c_o=10,
        mesh_extraction=True,
        number_octree_levels=3,
    ),
)

# %% [markdown]
# Visualize the 3D geological model using GemPy's plot_3d function.
gempy_plot = gpv.plot_3d(
    model=geo_model,
    kwargs_pyvista_bounds={
            'show_xlabels': False,
            'show_ylabels': False,
    },
    show=True,
    image=True
)

# %% [markdown]
# Combine all visual elements and display them together.
sp_mesh: pyvista.PolyData = gempy_plot.surface_points_mesh

pyvista_plotter = init_plotter()
pyvista_plotter.show_bounds(all_edges=True)

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

pyvista_plotter.add_actor(gempy_plot.surface_points_actor)

pyvista_plotter.show()
