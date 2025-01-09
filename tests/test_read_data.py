import os

import dotenv

from subsurface import TriSurf, UnstructuredData, PointSet
from subsurface.modules.reader import read_unstructured_topography, read_structured_topography
from subsurface.modules.visualization import to_pyvista_mesh, pv_plot, to_pyvista_points

dotenv.load_dotenv()

def test_read_topography():
    file_path = os.getenv("PATH_TO_TOPOGRAPHY")
    import ezdxf
    import numpy as np
    doc = ezdxf.readfile(file_path)
    vertices = []

    # Access the ENTITIES section
    for entity in doc.entities:
        # Check if the entity is a POLYLINE
        if entity.dxftype() == "POLYLINE":
            # Iterate over all vertices in the POLYLINE
            for vertex in entity.vertices:
                # Extract X, Y, Z coordinates
                x, y, z = vertex.dxf.location.xyz
                vertices.append([x, y, z])

    # Convert to a NumPy array
    foo = np.array(vertices)
    unstruct = UnstructuredData.from_array(
        vertex=foo,
        cells="points"
    )
    
    # unstruct = read_unstructured_topography(filepath)
    # # struct = read_structured_topography(topo_path)
    # 
    # ts = TriSurf(mesh=unstruct)
    s = to_pyvista_points(PointSet(data=unstruct))
    pv_plot([s], image_2d=False)
