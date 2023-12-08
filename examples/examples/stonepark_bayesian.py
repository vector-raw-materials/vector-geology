"""
Stonepark Geological Model
--------------------------  


Construct a 3D geological model of the Stonepark deposit using GemPy.


"""

import time

import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt

from gempy_probability.plot_posterior import default_red, default_blue
from vector_geology.stonepark_builder import initialize_geo_model, setup_geophysics

# %%
# Read nc from subsurface


# %%
import os
from dotenv import dotenv_values

from vector_geology.omf_to_gempy import process_file
import gempy as gp
import gempy_viewer as gpv
from vector_geology.utils import extend_box

start_time = time.time()  # start timer
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

from gempy_engine.core.backend_tensor import BackendTensor

BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH, dtype="float64")

#  %%
# Setup gempy object
geo_model = initialize_geo_model(
    structural_elements=structural_elements,
    extent=(np.array(global_extent)),
    topography=(xr.open_dataset(os.path.join(path, "Topography.nc"))),
    load_nuggets=True
)

# %%
geophysics_input = setup_geophysics(
    env_path="PATH_TO_STONEPARK_BOUGUER",
    geo_model=geo_model
)

# %% 
interpolation_options = geo_model.interpolation_options

interpolation_options.kernel_options.range = .7
interpolation_options.kernel_options.c_o = 3
interpolation_options.kernel_options.compute_condition_number = True

sol = gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype='float64'
    )
)

gempy_vista = gpv.plot_3d(
    model=geo_model,
    show=True,
    kwargs_plot_structured_grid={'opacity': 0.8},
    image=True
)
grav = - sol.gravity

# %%
# TODO: Scale the gravity data to the same scale as the model

plot2d = gpv.plot_2d(geo_model, show_topography=True, section_names=["topography"], show=False)
plot2d.axes[0].scatter(
    geophysics_input['X'],
    geophysics_input['Y'],
    c=grav,
    cmap='viridis',
    s=100,
    zorder=10000
)

plt.show()

# %% Calculate shift and scale


# %%
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.autoguide import init_to_mean
import torch
import gempy_engine
from gempy_engine.core.data.interpolation_input import InterpolationInput


prior_tensor = BackendTensor.t.array([2.61, 2.92, 3.1, 2.92, 2.61, 2.61]).to(torch.float64)

geo_model.geophysics_input = gp.data.GeophysicsInput(
    tz=geo_model.geophysics_input.tz,
    densities=prior_tensor,
)

def model(y_obs_list, interpolation_input):
    """
    This Pyro model represents the probabilistic aspects of the geological model.
    It defines a prior distribution for the top layer's location and 
    computes the thickness of the geological layer as an observed variable.
    """
    # Define prior for the top layer's location
    prior_mean = 2.62
    mu_density = pyro.sample(
        name=r'$\mu_{density}$', 
        fn=dist.Normal(prior_mean, torch.tensor(0.02, dtype=torch.float64))
    )

    # Update the model with the new top layer's location
    # interpolation_input: InterpolationInput = geo_model.interpolation_input
    geo_model.geophysics_input.densities = torch.index_put(
        input=prior_tensor,
        indices=(torch.tensor([0]),),
        values=mu_density
    )

    # Compute the geological model
    geo_model.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model.interpolation_options,
        data_descriptor=geo_model.input_data_descriptor,
        geophysics_input=geo_model.geophysics_input
    )

    simulated_geophysics = geo_model.solutions.gravity
    # pyro.deterministic(r'$\mu_{gravity}$', simulated_geophysics.detach())
    y_gravity = pyro.sample(
        name=r'$y_{gravity}$', 
        fn=dist.Normal(simulated_geophysics[0], 50),
        obs=y_obs_list
    )


# y_obs_list = torch.tensor(geophysics_input['Bouguer_267_complete'].values[0])
y_obs_list = torch.tensor([-436])

# %%
# Optimize mesh
interpolation_options.mesh_extraction = False
interpolation_options.number_octree_levels = 1
geo_model.grid.set_inactive("topography")
geo_model.grid.set_inactive("regular")


# %%
import arviz as az

# Run prior sampling and visualization
if False:
    prior = Predictive(model, num_samples=50)(y_obs_list, interpolation_input=geo_model.interpolation_input)
    data = az.from_pyro(prior=prior)
    az.plot_trace(data.prior)
    plt.show()


# Running MCMC using the NUTS algorithm

pyro.primitives.enable_validation(is_validate=True)
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=20)
mcmc.run(y_obs_list, interpolation_input=geo_model.interpolation_input)


posterior_samples = mcmc.get_samples(50)
posterior_predictive = Predictive(model, posterior_samples)(y_obs_list, interpolation_input=geo_model.interpolation_input)

# Creating a data object for ArviZ
data = az.from_pyro(
    posterior=mcmc,
    prior=prior,
    posterior_predictive=posterior_predictive
)

# Plotting trace of the sampled parameters
az.plot_trace(data)
plt.show()

# %%
# Density Plots of Posterior and Prior
# ------------------------------------
# Density plots provide a visual representation of the distribution of the sampled parameters.
# Comparing the posterior and prior distributions allows us to assess the impact of the observed data.

# Plotting density of posterior and prior distributions
az.plot_density(
    data=[data, data.prior],
    shade=.9,
    data_labels=["Posterior", "Prior"],
    colors=[default_red, default_blue],
)
plt.show()


az.plot_density(
    data=[data.posterior_predictive, data.prior_predictive],
    shade=.9,
    var_names=[r'$\mu_{density}$'],
    data_labels=["Posterior Predictive", "Prior Predictive"],
    colors=[default_red, default_blue],
)
plt.show()