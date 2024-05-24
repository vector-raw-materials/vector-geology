import os
import pandas as pd

import gempy as gp
import subsurface as ss


def generate_spremberg_model(elements_to_gempy: dict[str, dict[str, str]] ) -> gp.data.GeoModel:
    borehole_set: ss.core.geological_formats.BoreholeSet = _read_spremberg_borehole_set()

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

    return geo_model


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
