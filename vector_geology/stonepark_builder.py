import numpy as np

import gempy as gp
import xarray as xr
from vector_geology.model_building_functions import optimize_nuggets_for_group


def initialize_geo_model(structural_elements: list[gp.data.StructuralElement], extent: list[float],
                         topography: xr.Dataset, load_nuggets: bool = False
                         ) -> gp.data.GeoModel:

    structural_group_red = gp.data.StructuralGroup(
        name="Red",
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
        color_gen=gp.data.ColorsGenerator()
    )
    # TODO: If elements do not have color maybe loop them on structural frame constructor?

    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Tutorial_ch1_1_Basics',
        extent=extent,
        resolution=[20, 10, 20],
        refinement=6,  # * Here we define the number of octree levels. If octree levels are defined, the resolution is ignored.
        structural_frame=structural_frame
    )
    
    if topography is not None:
        gp.set_topography_from_arrays(
            grid=geo_model.grid,
            xyz_vertices=topography.vertex.values
        )
    
    if load_nuggets:
        apply_optimized_nuggets(
            geo_model=geo_model,
            loaded_nuggets_red=(np.load("nuggets_Red.npy")),
            loaded_nuggets_blue=(np.load("nuggets_Blue.npy")),
            loaded_nuggets_green=(np.load("nuggets_Green.npy"))
        )

    geo_model.structural_frame.get_element_by_name("Stonepark_KKR").color = "#A46283"
    geo_model.structural_frame.get_element_by_name("Stonepark_LGR").color = "#6394A4"
    geo_model.structural_frame.get_element_by_name("Stonepark_WAL").color = "#72A473"
    geo_model.structural_frame.get_element_by_name("Stonepark_ABL").color = "#1D3943"
    geo_model.structural_frame.basement_color = "#8B4220"

    geo_model.update_transform()

    return geo_model


def optimize_nuggets_for_whole_project(geo_model: gp.data.GeoModel):
    geo_model.interpolation_options.kernel_options.range = 0.7
    geo_model.interpolation_options.kernel_options.c_o = 4
    optimize_nuggets_for_group(
        geo_model=geo_model,
        structural_group=geo_model.structural_frame.get_group_by_name('Red'),
        plot_evaluation=False,
        plot_result=True
    )
    geo_model.interpolation_options.kernel_options.range = 2
    geo_model.interpolation_options.kernel_options.c_o = 4
    optimize_nuggets_for_group(
        geo_model=geo_model,
        structural_group=geo_model.structural_frame.get_group_by_name('Blue'),
        plot_evaluation=False,
        plot_result=False
    )
    if False:
        optimize_nuggets_for_group(
            geo_model=geo_model,
            structural_group=geo_model.structural_frame.get_group_by_name('Green'),
            plot_evaluation=False,
            plot_result=True
        )


def apply_optimized_nuggets(geo_model: gp.data.GeoModel, loaded_nuggets_red, loaded_nuggets_blue, loaded_nuggets_green):
    gp.modify_surface_points(
        geo_model,
        slice=None,
        elements_names=[element.name for element in geo_model.structural_frame.get_group_by_name('Red').elements],
        nugget=loaded_nuggets_red
    )
    if True:  # Ignore OB
        gp.modify_surface_points(
            geo_model,
            slice=None,
            elements_names=[element.name for element in geo_model.structural_frame.get_group_by_name('Blue').elements],
            nugget=loaded_nuggets_blue
        )
    if False:
        gp.modify_surface_points(
            geo_model,
            slice=None,
            elements_names=[element.name for element in geo_model.structural_frame.get_group_by_name('Green').elements],
            nugget=loaded_nuggets_green
        )

