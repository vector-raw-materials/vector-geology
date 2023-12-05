"""
Stonepark Geological Model
--------------------------  


Construct a 3D geological model of the Stonepark deposit using GemPy.


"""

import time

start_time = time.time()  # start timer

import numpy as np
# %%
# Read nc from subsurface


# %%
import os
from dotenv import dotenv_values

import matplotlib.pyplot as plt
import pyvista as pv

from vector_geology.omf_to_gempy import process_file
import gempy as gp
import gempy_viewer as gpv

config = dotenv_values()
path = config.get("PATH_TO_STONEPARK_Subsurface")
structural_elements = []
accumulated_roi = []
global_extent = None
color_gen = gp.data.ColorsGenerator()

for e, filename in enumerate(os.listdir(path)):
    base, ext = os.path.splitext(filename)
    if ext == '.nc':
        structural_element, global_extent = process_file(os.path.join(path, filename), global_extent, color_gen)
        structural_elements.append(structural_element)
# %%
# Element 1 is an intrusion


#  %%
# Setup gempy object
# structural_elements.pop(1)


# Red range 0.7 cov 4
structural_group_red = gp.data.StructuralGroup(
    name="Red",
    # elements=[structural_elements[i] for i in [0, 4, 6, 8]],
    elements=[structural_elements[i] for i in [0, 4, 8]],
    structural_relation=gp.data.StackRelationType.ERODE
)

# Any, Probably we can decimize this an extra notch
structural_group_green = gp.data.StructuralGroup(
    name="Green",
    elements=[structural_elements[i] for i in [5]],
    structural_relation=gp.data.StackRelationType.ERODE
)

# Blue range 2 cov 4
structural_group_blue = gp.data.StructuralGroup(
    name="Blue",
    elements=[structural_elements[i] for i in [2, 3]],
    structural_relation=gp.data.StackRelationType.ERODE
)

structural_group_intrusion = gp.data.StructuralGroup(
    name="Intrusion",
    elements=[structural_elements[i] for i in [1]],
    structural_relation=gp.data.StackRelationType.ERODE
)

structural_groups = [structural_group_intrusion, structural_group_green, structural_group_blue, structural_group_red]
structural_frame = gp.data.StructuralFrame(
    structural_groups=structural_groups[2:],
    color_gen=color_gen
)
# TODO: If elements do not have color maybe loop them on structural frame constructor?

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Tutorial_ch1_1_Basics',
    extent=global_extent,
    resolution=[20, 10, 20],
    refinement=4,  # * Here we define the number of octree levels. If octree levels are defined, the resolution is ignored.
    structural_frame=structural_frame
)

# gpv.plot_2d(geo_model, show_data=True)
# gempy_vista = gpv.plot_3d(geo_model, show_data=True, show=False)


# %% 
# ## Optimize nuggets
# In such a complex geometries often data does not fit perfectly. In order to account for this, GemPy allows to add a
# small random noise to the data. This is done by adding a small random value to the diagonal of the covariance matrix.
# We can optimize this value with respect to the condition number of the matrix. The condition number is a measure of
# how well the matrix can be inverted. The higher the condition number, the worse the matrix can be inverted. This
# means that the data is not well conditioned and the nugget should be increased. On the other hand, if the condition
# number is too low, the data is overfitted and the nugget should be decreased. The optimal value is the one that
# minimizes the condition number. This can be done with the following function:

# %%
from vector_geology.model_building_functions import optimize_nuggets_for_group

TRIGGER_OPTIMIZE_NUGGETS = False
APPLY_OPTIMIZED_NUGGETS = True
if TRIGGER_OPTIMIZE_NUGGETS:
    
    geo_model.interpolation_options.kernel_options.range = 0.7
    geo_model.interpolation_options.kernel_options.c_o = 4
    optimize_nuggets_for_group(
        geo_model=geo_model,
        structural_group=structural_group_red,
        plot_evaluation=False,
        plot_result=True
    )
    
    geo_model.interpolation_options.kernel_options.range = 2
    geo_model.interpolation_options.kernel_options.c_o = 4
    optimize_nuggets_for_group(
        geo_model=geo_model,
        structural_group=structural_group_green,
        plot_evaluation=False,
        plot_result=True
    )

    optimize_nuggets_for_group(
        geo_model=geo_model,
        structural_group=structural_group_blue,
        plot_evaluation=False,
        plot_result=False
    )

# %%
if APPLY_OPTIMIZED_NUGGETS:
    loaded_nuggets_red = np.load("nuggets_Red.npy")
    loaded_nuggets_green = np.load("nuggets_Green.npy")
    loaded_nuggets_blue = np.load("nuggets_Blue.npy")

    gp.modify_surface_points(
        geo_model,
        slice=None,
        elements_names=[element.name for element in geo_model.structural_frame.get_group_by_name('Red').elements],
        nugget=loaded_nuggets_red
    )
    if False:
        gp.modify_surface_points(
            geo_model,
            slice=None,
            elements_names=[element.name for element in geo_model.structural_frame.get_group_by_name('Green').elements],
            nugget=loaded_nuggets_green
        )

    if True: # Ignore OB
        gp.modify_surface_points(
                geo_model,
                slice=None,
                elements_names=[element.name for element in geo_model.structural_frame.get_group_by_name('Blue').elements],
                nugget=loaded_nuggets_blue
            )

geo_model

# %% 
geo_model.interpolation_options.mesh_extraction = True
geo_model.interpolation_options.kernel_options.range = .7
geo_model.interpolation_options.kernel_options.c_o = 3
geo_model.interpolation_options.kernel_options.compute_condition_number = True

from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
geo_model.interpolation_options.kernel_options.kernel_function = AvailableKernelFunctions.cubic

# %% 
# Refine each layer
# dark green: Stonepark_kkr to pink Knockroe
# purple: Stonepark_LGR to cyan: Lough Gur Fm
# light green: Stonepark_WAL light green: Waulsortian Limestone
# blue: Stonepark_ABL to dark green: Ballysteen Fm
# very light green basement to brown: Old red Sandstone

geo_model.structural_frame.get_element_by_name("Stonepark_KKR").color = "#A46283"
geo_model.structural_frame.get_element_by_name("Stonepark_LGR").color = "#6394A4"
geo_model.structural_frame.get_element_by_name("Stonepark_WAL").color = "#72A473"
geo_model.structural_frame.get_element_by_name("Stonepark_ABL").color = "#1D3943"
geo_model.structural_frame.basement_color = "#8B4220"

geo_model.update_transform()

gp.compute_model(
    geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype="float64"
    ),
)

gpv.plot_2d(geo_model, show_scalar=False)

end_time = time.time()
execution_time = end_time - start_time

print(f"The function executed in {execution_time} seconds.")

gempy_vista = gpv.plot_3d(
    model=geo_model,
    show=True,
    kwargs_plot_structured_grid={'opacity': 0.1}
)

if ADD_ORIGINAL_MESH := False:
    gempy_vista.p.add_mesh(triangulated_mesh, color="red", opacity=0.5)
