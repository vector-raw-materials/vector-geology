"""
Construct Spremberg: Building initial model
--------------------------------------------

This example demonstrates...
"""
import gempy as gp
import gempy_viewer as gpv
from vector_geology.model_contructor.spremberg import generate_spremberg_model

# %%
elements_to_gempy = {
        # "Buntsandstein"       : {
        #         "id"   : 53_300,
        #         "color": "#983999"
        # },
        "Werra-Anhydrit"      : {
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
        "Rotliegend"          : {
                "id"   : 62_000,
                "color": "#bb825b"
        }
}

geo_model: gp.data.GeoModel = generate_spremberg_model(
        elements_to_gempy=elements_to_gempy,
        plot = True
)

gempy_plot = gpv.plot_3d(geo_model,ve=10)

