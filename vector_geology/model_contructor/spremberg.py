import numpy as np
import os
import pandas as pd
import pyvista

import gempy as gp
import gempy_viewer as gpv
import subsurface as ss
from subsurface.modules.visualization import init_plotter, to_pyvista_points, to_pyvista_line


def generate_spremberg_model(
        borehole_set: ss.core.geological_formats.BoreholeSet,
        elements_to_gempy: dict[str, dict[str, str]], plot: bool = False) -> gp.data.GeoModel:

    elements: list[gp.data.StructuralElement] = gp.structural_elements_from_borehole_set(
        borehole_set=borehole_set,
        elements_dict=elements_to_gempy
    )

    group = gp.data.StructuralGroup(
        name="Stratigraphic Pile",
        elements=elements,
        structural_relation=gp.data.StackRelationType.ERODE
    )
    structural_frame = gp.data.StructuralFrame(
        structural_groups=[group],
        color_gen=gp.data.ColorsGenerator()
    )

    # %%
    # Get the extent from the borehole set
    all_surface_points_coords: gp.data.SurfacePointsTable = structural_frame.surface_points_copy
    extent_from_data = all_surface_points_coords.xyz.min(axis=0), all_surface_points_coords.xyz.max(axis=0)

    extent_from_data_ = [extent_from_data[0][0],
                         extent_from_data[1][0],
                         extent_from_data[0][1],
                         extent_from_data[1][1],
                         extent_from_data[0][2],
                         extent_from_data[1][2]]


    # Calculate point_y_axis
    n_octree_levels = 6
    regular_grid = gp.data.grid.RegularGrid.from_corners_box(
        pivot=(5_478_256.5, 5_698_528.946534388),
        point_x_axis=((5_483_077.527386775, 5_710_030.2446156405)),
        distance_point3=35_000,
        zmin=extent_from_data[0][2],
        zmax=extent_from_data[1][2],
        resolution=np.array([2 ** n_octree_levels] * 3),
        plot=True
    )
    
    
    interpolation_options = gp.data.InterpolationOptions(
        range=5,
        c_o=10,
        mesh_extraction=True,
        number_octree_levels=n_octree_levels,
    )

    grid = gp.data.grid.Grid()
    grid.set_octree_grid(regular_grid, interpolation_options.evaluation_options)

    geo_model = gp.data.GeoModel(
        name="Stratigraphic Pile",
        structural_frame=structural_frame,
        grid=grid,
        interpolation_options=interpolation_options,
    )

    gempy_plot = gpv.plot_3d(
        model=geo_model,
        kwargs_pyvista_bounds={
                'show_xlabels': False,
                'show_ylabels': False,
        },
        show=True,
        image=True
    )

    if plot:
        add_wells_plot(borehole_set, gempy_plot, geo_model.grid)

    return geo_model


def get_spremberg_borehole_set() -> ss.core.geological_formats.BoreholeSet:
    borehole_set: ss.core.geological_formats.BoreholeSet = _read_spremberg_borehole_set()
    return borehole_set


def add_wells_plot(borehole_set, gempy_plot, grid: gp.data.grid.Grid):
    sp_mesh: pyvista.PolyData = gempy_plot.surface_points_mesh
    well_mesh = to_pyvista_line(
        line_set=borehole_set.combined_trajectory,
        active_scalar="lith_ids",
        radius=10
    )
    units_limit = [0, 20]
    collars = to_pyvista_points(
        borehole_set.collars.collar_loc,
    )
    pyvista_plotter = init_plotter(ve=10)
    pyvista_plotter.show_bounds(all_edges=False)
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
    pyvista_plotter.add_point_labels(
        points=borehole_set.collars.collar_loc.points,
        labels=borehole_set.collars.ids,
        point_size=3,
        shape_opacity=0.5,
        font_size=12,
        bold=True
    )
    pyvista_plotter.add_actor(gempy_plot.surface_points_actor)

    if ADD_GRID:=False:
        grid_points = pyvista.PointSet(grid.values)
        pyvista_plotter.add_mesh(grid_points, color="red", point_size=1)

    pyvista_plotter.show()


def _read_spremberg_borehole_set() -> ss.core.geological_formats.BoreholeSet:
    from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
    from subsurface.core.geological_formats import Survey
    from subsurface.core.geological_formats import Collars
    from subsurface.core.geological_formats.boreholes.boreholes import MergeOptions
    from subsurface.core.geological_formats import BoreholeSet
    from subsurface.modules.reader.wells.read_borehole_interface import read_lith, read_survey, read_collar

    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_STRATIGRAPHY"),
        columns_map={
                'hole_id'   : 'id',
                'depth_from': 'top',
                'depth_to'  : 'base',
                'lit_code'  : 'component lith'
        }
    )
    lith: pd.DataFrame = read_lith(reader)
    reader: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_SURVEY"),
        columns_map={
                'depth'  : 'md',
                'dip'    : 'dip',
                'azimuth': 'azi'
        },
    )
    df = read_survey(reader)
    survey: Survey = Survey.from_df(df)
    survey.update_survey_with_lith(lith)
    reader_collar: GenericReaderFilesHelper = GenericReaderFilesHelper(
        file_or_buffer=os.getenv("PATH_TO_SPREMBERG_COLLAR"),
        header=0,
        usecols=[0, 1, 2, 4],
        columns_map={
                "hole_id"            : "id",  # ? Index name is not mapped
                "X_GK5_incl_inserted": "x",
                "Y__incl_inserted"   : "y",
                "Z_GK"               : "z"
        }
    )

    df_collar = read_collar(reader_collar)
    collar = Collars.from_df(df_collar)
    borehole_set = BoreholeSet(
        collars=collar,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )

    return borehole_set
