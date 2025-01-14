import omfvista
import omfvista
import pandas as pd
import pyvista

import subsurface
from subsurface import TriSurf, LineSet
from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.modules.reader import read_unstructured_topography
from subsurface.modules.reader.mesh.dxf_reader import DXFEntityType
from subsurface.modules.reader.wells.read_borehole_interface import read_lith, read_survey, read_collar
from subsurface.modules.visualization import to_pyvista_line, to_pyvista_points
from subsurface.modules.visualization import to_pyvista_mesh


def load_spremberg_meshes(path_to_files: str) -> tuple[list[pyvista.PolyData], list[pyvista.PolyData]]:
    # %%
    # Load OMF Project
    # ----------------

    project = omfvista.load_project(path_to_files)
    omf_project = project
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
            
    return meshes, lines


def process_borehole_data(
        path_to_stratigraphy: str,
        path_to_survey: str,
        path_to_collar: str
) -> tuple[pyvista.PolyData, pyvista.PolyData]:
    # Initialize the reader for the lithological data. Specify the file path and column mappings.
    reader_lith = GenericReaderFilesHelper(
        file_or_buffer=path_to_stratigraphy,
        columns_map={
                'hole_id'   : 'id',
                'depth_from': 'top',
                'depth_to'  : 'base',
                'lit_code'  : 'component lith'
        }
    )
    # Read the lithological data into a DataFrame.
    lith = read_lith(reader_lith)

    # Initialize the reader for the survey data. Specify the file path and column mappings.
    reader_survey = GenericReaderFilesHelper(
        file_or_buffer=path_to_survey,
        columns_map={
                'depth'  : 'md',
                'dip'    : 'dip',
                'azimuth': 'azi'
        },
    )
    # Read the survey data into a DataFrame.
    df_survey = read_survey(reader_survey)

    # Create a Survey object from the DataFrame and update it with lithological data.
    survey = Survey.from_df(
        survey_df=df_survey,
        attr_df=None
    )
    survey.update_survey_with_lith(lith)

    # Initialize the reader for the collar data. Specify the file path, header, and column mappings.
    reader_collar = GenericReaderFilesHelper(
        file_or_buffer=path_to_collar,
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

    # Combine the collar and survey data into a BoreholeSet.
    borehole_set = BoreholeSet(
        collars=collar,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )

    # Visualize the borehole trajectories and collars using PyVista.
    well_mesh = to_pyvista_line(
        line_set=borehole_set.combined_trajectory,
        active_scalar="lith_ids",
        radius=10
    )
    collars = to_pyvista_points(borehole_set.collars.collar_loc)

    return well_mesh, collars


def read_topography(file_path):
    unstruct = read_unstructured_topography(
        path=file_path,
        additional_reader_kwargs={'entity_type': DXFEntityType.POLYLINE}
    )
    ts = TriSurf(mesh=unstruct)
    s1 = to_pyvista_mesh(ts)
    return s1

def read_seismic_profiles(
            interpretation_path: str,
            section_path: str,
            crop_coords: dict,
            zmin: float,
            zmax: float
) -> pyvista.PolyData:
        import tifffile as tiff  # Install with pip install tifffile
        import pandas as pd
        from subsurface import TriSurf, UnstructuredData, StructuredData
        from subsurface.modules.reader.profiles.profiles_core import create_vertical_mesh
        from subsurface.modules.visualization import to_pyvista_mesh

        image = tiff.imread(interpretation_path)

        # Perform the crop
        cropped_image = image[crop_coords['y_start']:crop_coords['y_end'], crop_coords['x_start']:crop_coords['x_end']]
        texture = StructuredData.from_numpy(cropped_image)

        # Read coordinates
        df = pd.read_csv(
            filepath_or_buffer=section_path,
            skiprows=4,  # Skip the header lines above 'CDP'
            delim_whitespace=True,  # Treat consecutive spaces as separators
            names=["CDP", "X_COORD", "Y_COORD"]  # Assign column names
        )
        coords = df[["X_COORD", "Y_COORD"]].to_numpy()

        vertices, faces = create_vertical_mesh(coords, zmin, zmax)
        geometry: UnstructuredData = UnstructuredData.from_array(vertices, faces)

        ts = TriSurf(
            mesh=geometry,
            texture=texture,
            texture_origin=[coords[0][0], coords[0][1], zmin],
            texture_point_u=[coords[-1][0], coords[-1][1], zmin],
            texture_point_v=[coords[0][0], coords[0][1], zmax]
        )
        
        return to_pyvista_mesh(ts)


