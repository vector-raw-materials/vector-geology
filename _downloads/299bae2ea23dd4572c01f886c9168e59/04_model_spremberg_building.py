"""
Construct Spremberg: Building initial model
--------------------------------------------

This example demonstrates...

"""

# sphinx_gallery_thumbnail_number = -2
import numpy as np

import gempy as gp
import gempy_viewer as gpv
from subsurface.modules.visualization import to_pyvista_line, to_pyvista_points
from vector_geology.model_contructor.spremberg import generate_spremberg_model, get_spremberg_borehole_set, add_wells_plot

# %%
elements_to_gempy = {
        "Buntsandstein"       : {
                "id"   : 53_300,
                "color": "#983999"
        },
        "Werra-Anhydrit"      : {
                "id"   : 61_730,
                "color": "#00923f"
        },
        # "Kupferschiefer"      : {
        #         "id"   : 61_760,
        #         "color": "#da251d"
        # },
        "Zechsteinkonglomerat": {
                "id"   : 61_770,
                "color": "#f8c300"
        },
        "Rotliegend"          : {
                "id"   : 62_000,
                "color": "#bb825b"
        }
}

spremberg_boreholes = get_spremberg_borehole_set()
geo_model: gp.data.GeoModel = generate_spremberg_model(
    borehole_set=spremberg_boreholes,
    elements_to_gempy=elements_to_gempy,
    plot=False
)

# %%
# Add one orientation to the model
rotliegend: gp.data.StructuralElement = geo_model.structural_frame.get_element_by_name("Rotliegend")
gp.add_orientations(
    geo_model=geo_model,
    x=[5_460_077.527386775, 5_450_077.527386775],
    y=[5_720_030.2446156405, 5_710_030.2446156405],
    z=[-600, -600],
    elements_names=["Rotliegend", "Rotliegend"],
    pole_vector=[
            np.array([.7, 0.7, 0.2]),
            np.array([-1, 0, 0.2])
    ],
)

# %%
pivot = [5_478_256.5, 5_698_528.946534388]
point_2 = [5_483_077.527386775, 5_710_030.2446156405]
point_3 = [5_474_977.5974836275, 5_712_059.373443342]
section_dict = {
        'section1': (pivot, point_2, [100, 100]),
        'section2': (pivot, point_3, [100, 100]),
        'section3': (point_2, point_3, [100, 100])
}


# %% 
gp.set_section_grid(geo_model.grid, section_dict)
gpv.plot_section_traces(geo_model)

# %%
_ = gpv.plot_3d(
    model=geo_model,
    ve=10,
    image=True,
    transformed_data=True,
    kwargs_pyvista_bounds={
            'show_xlabels': False,
            'show_ylabels': False,
    },
    kwargs_plot_data={
            'arrow_size': 0.001000
    }
)

# %%

# * Ignore curvature for now
geo_model.interpolation_options.kernel_options.range = 3
geo_model.interpolation_options.compute_scalar_gradient = False
geo_model.interpolation_options.evaluation_options.curvature_threshold = 0.4
geo_model.interpolation_options.evaluation_options.number_octree_levels_surface = 5

geo_model.interpolation_options.evaluation_options.error_threshold = 1
geo_model.interpolation_options.evaluation_options.verbose = True


gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype="float64",
    )
)

# %%
gpv.plot_2d(
    model=geo_model,
    figsize=(15, 15),
    ve=10,
    section_names=['section1', 'section2', 'section3']
)

# %%
gempy_plot = gpv.plot_3d(
    model=geo_model,
    show_lith=False,
    ve=10,
    show=False,
    image=False,
    transformed_data=False,
    kwargs_pyvista_bounds={
            'show_xlabels': True,
            'show_ylabels': True,
            'show_zlabels': False,
    },
    kwargs_plot_data={
            'arrow_size': 100.001000
    }
)

well_mesh = to_pyvista_line(
    line_set=spremberg_boreholes.combined_trajectory,
    active_scalar="lith_ids",
    radius=10
)
units_limit = [0, 20]
collars = to_pyvista_points(spremberg_boreholes.collars.collar_loc)
gempy_plot.p.add_mesh(
    well_mesh.threshold(units_limit),
    cmap="tab20c",
    clim=units_limit
)

gempy_plot.p.add_mesh(
    collars,
    point_size=10,
    render_points_as_spheres=True
)

gempy_plot.p.add_point_labels(
    points=spremberg_boreholes.collars.collar_loc.points,
    labels=spremberg_boreholes.collars.ids,
    point_size=3,
    shape_opacity=0.5,
    font_size=12,
    bold=True
)

gempy_plot.p.show()


# %%
# LiquidEarth Integration
# ~~~~~~~~~~~~~~~~~~~~~~~
# Beyond the classical plotting capabilities introduced in GemPy v3, users can now also upload models to LiquidEarth. 
# `LiquidEarth <https://www.terranigma-solutions.com/liquidearth>`_ is a collaborative platform designed for 3D visualization,
# developed by many of the main `gempy` maintainers,  with a strong focus on collaboration and sharing. 
# This makes it an excellent tool for sharing your models with others and viewing them across different platforms.
# To upload a model to LiquidEarth, you must have an account and a user token. Once your model is uploaded, 
# you can easily share the link with anyone.

# # %%
if True:
    link = gpv.plot_to_liquid_earth(
        geo_model=geo_model,
        space_name="Spremberg",
        file_name="gempy_model",
        user_token=None,  # If None, it will try to grab it from the environment
        grab_link=True,
    )

    print(f"Generated Link: {link}")
