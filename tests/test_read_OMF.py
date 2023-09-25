import numpy as np
import pandas as pd
import pyvista
import omfvista
from dotenv import dotenv_values

import pytest

import subsurface
from subsurface import TriSurf
from subsurface.visualization import to_pyvista_mesh, pv_plot
from subsurface.writer import base_structs_to_binary_file


@pytest.fixture(scope="module")
def load_omf():
    config = dotenv_values()
    path = config.get('PATH_TO_OMF2')
    omf = omfvista.load_project(path)
    return omf


def test_read_omf_with_pyvista(load_omf):
    omf = load_omf
    omf.plot(multi_colors=True, show_edges=True, notebook=False)


def test_omf_to_unstruct_single_block(load_omf):
    omf = load_omf
    polydata_obj: pyvista.PolyData = omf["GSB - BIF contacts"]
    unstruct_pyvista: pyvista.UnstructuredGrid = polydata_obj.cast_to_unstructured_grid()
    cells_pyvista = unstruct_pyvista.cells.reshape(-1, 4)[:, 1:]

    unstruct: subsurface.UnstructuredData = subsurface.UnstructuredData.from_array(
        vertex=unstruct_pyvista.points,
        cells=cells_pyvista,
    )

    base_structs_to_binary_file("leapfrog1", unstruct)

    ts = TriSurf(mesh=unstruct)
    s = to_pyvista_mesh(ts)
    pv_plot([s], image_2d=True)
