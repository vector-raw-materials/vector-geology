"""
2.2 - Including GemPy
=====================

Complex probabilistic model
---------------------------
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.autoguide import init_to_mean

import gempy as gp
import gempy_engine
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor
import arviz as az
from gempy_probability.plot_posterior import default_red, default_blue

from gempy_viewer.modules.plot_2d.drawer_contours_2d import plot_regular_grid_contacts

# sphinx_gallery_thumbnail_number = -1

# %%
# Config
seed = 123456
torch.manual_seed(seed)
pyro.set_rng_seed(seed)

# %%
# Set the data path
data_path = os.path.abspath('../')


# %%
# Define a function for plotting geological settings with wells
def plot_geo_setting_well(geo_model, alpha=1., show_lith=True):
    """
    This function plots the geological settings along with the well locations.
    It uses gempy_viewer to create 2D plots of the model.
    """
    # Define well and device locations
    device_loc = np.array([[6e3, 0, 3700]])
    well_1 = 3.41e3
    well_2 = 3.6e3

    # Create a 2D plot
    p2d = gpv.plot_2d(
        geo_model,
        show_topography=False,
        legend=False,
        show=False,
        show_lith=show_lith,
        kwargs_boundaries={'alpha': alpha}
    )

    # Add well and device markers to the plot
    p2d.axes[0].scatter([3e3], [well_1], marker='^', s=400, c='#71a4b3', zorder=10)
    p2d.axes[0].scatter([9e3], [well_2], marker='^', s=400, c='#71a4b3', zorder=10)
    p2d.axes[0].scatter(device_loc[:, 0], device_loc[:, 2], marker='x', s=400, c='#DA8886', zorder=10)

    # Add vertical lines to represent wells
    p2d.axes[0].vlines(3e3, .5e3, well_1, linewidth=4, color='gray')
    p2d.axes[0].vlines(9e3, .5e3, well_2, linewidth=4, color='gray')

    # Show the plot
    p2d.fig.show()
    return p2d


# %%
# Creating the Geological Model
# -----------------------------
# Here we create a geological model using GemPy. The model defines the spatial extent,
# resolution, and geological information derived from orientations and surface points data.
geo_model = gp.create_geomodel(
    project_name='Wells',
    extent=[0, 12000, -500, 500, 0, 4000],
    resolution=[100, 2, 100],
    # refinement=1,
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/2-layers/2-layers_orientations.csv",
        path_to_surface_points=data_path + "/data/2-layers/2-layers_surface_points.csv"
    )
)

# %%
# Configuring the Model
# ---------------------
# We configure the interpolation options for the geological model. 
# These options determine how the model interpolates between data points.

geo_model.interpolation_options.uni_degree = 1
geo_model.interpolation_options.mesh_extraction = False
geo_model.interpolation_options.sigmoid_slope = 1000  # ! Temporary fix to set the hard sigmoid 

# %%
# Setting up a Custom Grid
# ------------------------
# A custom grid is set for the model, defining specific points in space
# where geological formations will be evaluated.

x_loc = 6000
y_loc = 0
z_loc = np.linspace(0, 4000, 100)
xyz_coord = np.array([[x_loc, y_loc, z] for z in z_loc])
gp.set_custom_grid(geo_model.grid, xyz_coord=xyz_coord)

# %%
# Plotting the Initial Geological Setting
# ---------------------------------------
# Before running any probabilistic analysis, we first visualize the initial geological setting.
# This step ensures that our model is correctly set up with the initial data.

# Plot initial geological settings
plot_geo_setting_well(geo_model=geo_model)

# %%
# Interpolating the Initial Guess
# -------------------------------
# The model interpolates an initial guess for the geological formations.
# This step is crucial to provide a starting point for further probabilistic analysis.

gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.numpy)
)
plot_geo_setting_well(geo_model=geo_model)

# %%
# Probabilistic Geomodeling with Pyro
# -----------------------------------
# In this section, we introduce a probabilistic approach to geological modeling.
# By using Pyro, a probabilistic programming language, we define a model that integrates
# geological data with uncertainty quantification.

sp_coords_copy = geo_model.interpolation_input_copy.surface_points.sp_coords.copy()
# Change the backend to PyTorch for probabilistic modeling
BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)


# %%
# Defining the Probabilistic Model
# --------------------------------
# The Pyro model represents the probabilistic aspects of the geological model.
# It defines a prior distribution for the top layer's location and computes the thickness
# of the geological layer as an observed variable.

def model(y_obs_list):
    """
    This Pyro model represents the probabilistic aspects of the geological model.
    It defines a prior distribution for the top layer's location and 
    computes the thickness of the geological layer as an observed variable.
    """
    # Define prior for the top layer's location
    prior_mean = sp_coords_copy[0, 2]
    mu_top = pyro.sample(
        name=r'$\mu_{top}$',
        fn=dist.Normal(prior_mean, torch.tensor(0.02, dtype=torch.float64))
    )

    # Update the model with the new top layer's location
    interpolation_input = geo_model.interpolation_input_copy
    interpolation_input.surface_points.sp_coords = torch.index_put(
        interpolation_input.surface_points.sp_coords,
        (torch.tensor([0]), torch.tensor([2])),
        mu_top
    )

    # Compute the geological model
    geo_model.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model.interpolation_options,
        data_descriptor=geo_model.input_data_descriptor,
        geophysics_input=geo_model.geophysics_input,
    )

    # Compute and observe the thickness of the geological layer
    simulated_well = geo_model.solutions.octrees_output[0].last_output_center.custom_grid_values
    thickness = simulated_well.sum()
    pyro.deterministic(r'$\mu_{thickness}$', thickness.detach())
    y_thickness = pyro.sample(r'$y_{thickness}$', dist.Normal(thickness, 50), obs=y_obs_list)


# %%
# Running Prior Sampling and Visualization
# ----------------------------------------
# Prior sampling is an essential step in probabilistic modeling. 
# It helps in understanding the distribution of our prior assumptions before observing any data.

# %%
# Prepare observation data
y_obs_list = torch.tensor([200, 210, 190])

# %%
# Run prior sampling and visualization
prior = Predictive(model, num_samples=50)(y_obs_list)
data = az.from_pyro(prior=prior)
az.plot_trace(data.prior)
plt.show()

# %%
# Sampling from the Posterior using MCMC
# --------------------------------------
# We use Markov Chain Monte Carlo (MCMC) with the NUTS (No-U-Turn Sampler) algorithm 
# to sample from the posterior distribution. This gives us an understanding of the 
# distribution of our model parameters after considering the observed data.

# %%
# Run MCMC using NUTS to sample from the posterior

# Magic sauce
from gempy_engine.core.backend_tensor import BackendTensor

# import gempy_engine.config
# config.DEFAULT_PYKEOPS = False
BackendTensor._change_backend(
    engine_backend=gp.data.AvailableBackends.PYTORCH,
    dtype="float64",
    pykeops_enabled=False
)

pyro.primitives.enable_validation(is_validate=True)
nuts_kernel = NUTS(model, step_size=0.0085, adapt_step_size=True, target_accept_prob=0.9, max_tree_depth=10, init_strategy=init_to_mean)
mcmc = MCMC(nuts_kernel, num_samples=200, warmup_steps=50)
mcmc.run(y_obs_list)

# %%
# Posterior Predictive Checks
# ---------------------------
# After obtaining the posterior samples, we perform posterior predictive checks.
# This step is crucial to evaluate the performance and validity of our probabilistic model.

# %%
# Sample from posterior predictive and visualize
posterior_samples = mcmc.get_samples()
posterior_predictive = Predictive(model, posterior_samples)(y_obs_list)
data = az.from_pyro(posterior=mcmc, prior=prior, posterior_predictive=posterior_predictive)
az.plot_trace(data)
plt.show()

# %%
# Density Plot of Posterior Predictive
# ------------------------------------
# A density plot provides a visual representation of the distribution of the 
# posterior predictive checks. It helps in comparing the prior and posterior distributions 
# and in assessing the impact of our observed data on the model.

# %%
# Plot density of posterior predictive and prior predictive
az.plot_density(
    data=[data.posterior_predictive, data.prior_predictive],
    shade=.9,
    var_names=[r'$\mu_{thickness}$'],
    data_labels=["Posterior Predictive", "Prior Predictive"],
    colors=[default_red, default_blue],
)
plt.show()

# %%
# Plot spagetti
p2d = plot_geo_setting_well(geo_model, alpha=0.1, show_lith=False)
for i in range(0,199,5):
    gp.modify_surface_points(
        geo_model=geo_model,
        slice=0,
        Z= geo_model.input_transform.apply_inverse(
            points=np.array([[0, 0, data.posterior[r'$\mu_{top}$'][0][i].item()]]))[0,2]
    )

    gp.compute_model(geo_model)

    plot_regular_grid_contacts(
        gempy_model=geo_model,
        ax=p2d.axes[0],
        slicer_data=p2d.section_data_list[0].slicer_data,
        resolution=geo_model.grid.regular_grid.resolution,
        only_faults=False,
        kwargs={'alpha': 0.1}
    )

p2d.fig.show()

# sphinx_gallery_thumbnail_number = 2
