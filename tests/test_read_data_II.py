import os
import pathlib

import dotenv
import numpy as np
from matplotlib import pyplot as plt

from subsurface import optional_requirements
from subsurface.modules.reader.volume.read_volume import pv_cast_to_explicit_structured_grid
import subsurface.modules.tools.mocking_aux as mocking_aux

dotenv.load_dotenv()


def test_mock_gravity_volume_read():
    pv = optional_requirements.require_pyvista()
    filepath = os.getenv("PATH_TO_SPREMBERG_FAKE_GEOPHYSICS")

    pyvista_obj: pv.DataSet = pv.read(filepath)
    pyvista_struct: pv.ExplicitStructuredGrid = pv_cast_to_explicit_structured_grid(pyvista_obj)
    pv.plot(
        pyvista_struct,
        show_grid=True,
    )

def test_mock_gravity_volume():
    pv = optional_requirements.require_pyvista()
    filepath = os.getenv("PATH_TO_SPREMBERG_FAKE_GEOPHYSICS_SOURCE")

    pyvista_obj: pv.DataSet = pv.read(filepath)
    pyvista_struct: pv.ExplicitStructuredGrid = pv_cast_to_explicit_structured_grid(pyvista_obj)

    mocking_aux.update_extent(
        pyvista_struct,
        new_extent=[
                5.45334920e+06, 5.47499100e+06,
                5.70195450e+06, 5.71927041e+06,
                -1.28654213e+03, -4.76860133e+01
        ]
    )

    # Set your threshold limits.
    lower_threshold = -2_000
    upper_threshold = 2_000
    
    transformers = [
            mocking_aux.transform_subtract_mean,
            mocking_aux.transform_scale,
            mocking_aux.transform_gaussian_blur,  # Apply as a grid operation
            lambda values: mocking_aux.transform_sinusoidal(values, amplitude=200, frequency=0.005),
    ]

    # Apply the transformation to obfuscate the data.
    pyvista_struct: pv.ExplicitStructuredGrid = mocking_aux.obfuscate_model_name(
        grid=pyvista_struct, 
        transform_functions=transformers,
        attr="model_name"
    )

    if False:
        output_path = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            'spremberg_gravity_volume.vtk'
        )

        pyvista_struct.save(output_path)

    pyvista_struct: pv.UnstructuredGrid = pyvista_struct.threshold([lower_threshold, upper_threshold], scalars='model_name')
    values = np.array(pyvista_struct['model_name'])

    # Create a histogram plot with a specified number of bins.
    plt.hist(values, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel(r'Value')
    plt.ylabel(r'Frequency')
    plt.title(r'Histogram of Dataset Values')
    plt.grid(True)
    plt.show()

    # show
    pv.plot(
        pyvista_struct,
        show_grid=True,
    )

