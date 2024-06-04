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
        # "Zechsteinkonglomerat": {
        #         "id"   : 61_770,
        #         "color": "#f8c300"
        # },
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
    x=[5_483_077.527386775],
    y=[5_710_030.2446156405],
    z=[0.946534388],
    elements_names=["Rotliegend"],
    pole_vector=[np.array([0, 0, 1])],
)

# %%
gempy_plot = gpv.plot_3d(
    model=geo_model,
    ve=10,
    image=True,
    kwargs_pyvista_bounds={
            'show_xlabels': False,
            'show_ylabels': False,
    }
)

# %%
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
    kwargs_pyvista_bounds={
            'show_xlabels': False,
            'show_ylabels': False,
    }
)
