"""
Construct Spremberg: Building initial model
--------------------------------------------

This example demonstrates...
"""
import numpy as np

import gempy as gp
import gempy_viewer as gpv
from vector_geology.model_contructor.spremberg import generate_spremberg_model

# %%
elements_to_gempy = {
        # "Buntsandstein"       : {
        #         "id"   : 53_300,
        #         "color": "#983999"
        # },
        "Werra-Anhydrit": {
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
        "Rotliegend"    : {
                "id"   : 62_000,
                "color": "#bb825b"
        }
}

geo_model: gp.data.GeoModel = generate_spremberg_model(
    elements_to_gempy=elements_to_gempy,
    plot=False
)

# %%
# Add one orientation to the model
rotliegend: gp.data.StructuralElement = geo_model.structural_frame.get_element_by_name("Rotliegend")
gp.add_orientations(
    geo_model=geo_model,
    x=[5_460_077.527386775],
    y=[5_720_030.2446156405],
    z=[0.946534388],
    elements_names=["Rotliegend"],
    pole_vector=[np.array([.7, 0.7, 0.2])],
)

# %%
gempy_plot = gpv.plot_3d(
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
geo_model.interpolation_options.kernel_options.range = 1
geo_model.interpolation_options.compute_scalar_gradient = False
gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype="float64",
    )
)

# %%
gpv.plot_3d(
    model=geo_model,
    ve=10,
    image=False,
    transformed_data=False,
    kwargs_pyvista_bounds={
            'show_xlabels': False,
            'show_ylabels': False,
    },
    kwargs_plot_data={
            'arrow_size': 100.001000
    }
)
