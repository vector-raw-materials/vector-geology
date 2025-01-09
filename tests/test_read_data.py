import os

import dotenv

from subsurface import TriSurf, UnstructuredData, PointSet
from subsurface.modules.reader import read_unstructured_topography, read_structured_topography
from subsurface.modules.reader.mesh.dxf_reader import DXFEntityType
from subsurface.modules.visualization import to_pyvista_mesh, pv_plot, to_pyvista_points

dotenv.load_dotenv()


def test_read_topography_I():
    file_path = os.getenv("PATH_TO_TOPOGRAPHY")
    unstruct = read_unstructured_topography(
        path=file_path,
        additional_reader_kwargs={'entity_type': DXFEntityType.POLYLINE}
    )
    ts = TriSurf(mesh=unstruct)
    s = to_pyvista_mesh(ts)
    pv_plot([s], image_2d=False)
