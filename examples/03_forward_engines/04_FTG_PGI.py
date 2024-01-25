"""
Inversion of Full Tensor Gravity Gradiometry Data
-------------------------------------------------

# # Inversion of Full Tensor Gravity Gradiometry Data
# 
# This notebook showcases the use of Petrophysically and Geologically guided Inversion (PGI), given by [Astic and Oldenburg (2019)](https://doi.org/10.1093/gji/ggz389), to invert a sub-surface density model from Full-Tensor Gravity Gradiometry (FTG-G) data measured over the Irish Midlands. This notebook covers the following points:
# 
# - Loading the necessary modules and data
# - Generating a mesh
# - Initializing the Gaussian Mixture Model
# - Tuning the hyper-parameters
# - Plotting the results
# 
# This has been built and modified based on the original example notebook in SimPEG's documentation (please check [here](https://docs.simpeg.xyz/content/tutorials/14-pgi/index.html)). Please check the [SimPEG docs](https://docs.simpeg.xyz/index.html) for additional information on SimPEG.

"""

import os
import shutil

import SimPEG.potential_fields as pf
# Import the SimPEG functions
import discretize as ds
import matplotlib as mpl
import matplotlib.pyplot as plt
# Import the general modules
import numpy as np
import pandas as pd
from SimPEG import (
    maps,
    data,
    utils,
    inverse_problem,
    inversion,
    optimization,
    regularization,
    data_misfit,
    directives,
)
from SimPEG.potential_fields import gravity as grav
# Import the discretize module
from discretize.utils import active_from_xyz

# Import SimpegHelper for plotting and resampling data
from vector_geology import SimpegHelper as SH

# Plot beautification
formatter = mpl.ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))

mpl.rc("axes", titlesize=14, labelsize=12)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# %%
# ### Step 1: Create Directories for storing model iterations and outputs
# 
# Since PGI is an iterative approach, the notebook has options to save the model at every iteration, since overfitting can be prevented by changing the stopping criterion, which can be inferred by viewing the model iterations.


# Name of the model
name = "Ireland_FTG"

# Names for the directories for the model iterations and output
path_to_mod_iterations = "./" + str(name) + "/Model Iterations"
path_to_output = "./" + str(name) + "/Output"

# Check if the model iterations directory exists and remove it if it does
if os.path.exists(path_to_mod_iterations):
    shutil.rmtree(path_to_mod_iterations)

# Create the model iterations directory.
os.makedirs(path_to_mod_iterations)

# Check if the output directory exists and remove it if it does
if os.path.exists(path_to_output):
    shutil.rmtree(path_to_output)

# Create the output directory
os.makedirs(path_to_output)

# %%
# ### Step 2: Load the FTG-G Data
# 
# Full Tensor Gradiometry data is a 2-Tensor, with the general structure as follows:
# 
# $$
# G = \left[\begin{array}{cc} 
# G_{xx} & G_{xy} & G_{xz}\\
# G_{xy} & G_{yy} & G_{yz}\\
# G_{xz} & G_{yz} & G_{zz}
# \end{array}\right]
# $$
# 
# The matrix $G$ is symmetric and traceless (i.e., $\sum_{i}{G_{ii}} = 0$) as $G_{ij} = \frac{\partial^2{\Phi}}{\partial{x_i}\partial{x_j}}$, and $\Phi$ is a solution to Laplace's equation (i.e., $\nabla^2\Phi = 0$), where $\Phi$ is the scalar gravitational potential field. Hence it only has five independent components, which are used for the inversion.

# In[3]:


# Read the csv file into a pandas dataframe and then convert it to a numpy array.
FTG_Data = pd.read_csv('temp/VECTOR_FTG_PGI/Ireland_FTG_Gridded_Final.csv', delimiter=",").to_numpy()

# %%
# ### Step 3: Resample the data onto a regular grid.
# 
# While not mandatory, it aids in mesh generation (since the data is used to define the mesh bounds). The ``SimpegHelper.pf_rs()`` will take in potential field data and resample it onto a new regular grid.

# In[4]:


# %%
# New sampling interval (ground units)
inc = 150

# Resample the data
[grav_new, nx_new, ny_new] = SH.pf_rs(FTG_Data, inc, bounds=[156000, 168000, 143000, 148500])

# Extract the gravity gradiometry data vectors
grav_vec = grav_new[:, 3:]

# Negate the gravity data to the opposite sign to match the coordinate system.
# NOTE: UNDER CONSIDERATION, WILL UPDATE THIS STEP.
grav_vec = grav_vec * np.array([-1.0, -1.0, 1.0, 1.0, -1.0])[None, :]
# NOTE: This step is necessary for real data since the convention followed by
# the SimPEG forward operator is the opposite of the general convention

# Extract the topography from the data
inv_topo = grav_new[:, [0, 1, 2]]

# %%
# ### Visualize the data
# 
# The ``SimpegHelper.plot_2D_data()`` function will plot the datasets using ``matplotlib``.

# In[5]:


# %%
# Plot FTG Data
SH.plot_2D_data(np.c_[grav_new[:, [0, 1, 2]], grav_vec[:, 0]], [np.nanmin(grav_vec[:, 0]), np.nanmax(grav_vec[:, 0])], cmap="jet", which_data="FTG", comp="xx", path_to_output=path_to_output, name=name)
# Plot Topo Data
SH.plot_2D_data(np.c_[grav_new[:, [0, 1, 2]], inv_topo[:, -1]], [np.nanmin(inv_topo[:, -1]), np.nanmax(inv_topo[:, -1])], cmap="terrain", which_data="Topo", comp="xx", path_to_output=path_to_output, name=name)

# %%
# ### Step 4: Create a TensorMesh object to invert the data
# 
# Using the ``discretize.TensorMesh`` utility, create a 3D Tensor Mesh to invert the gravity gradiometry data. A Tensor Mesh is a cartesian mesh, with the optional ability to have expanding or contracting cell sizes. In this specific case we make a Tensor Meshh object such that outside the core block, every successive cell will have a cell size as follows:
# 
# $$
# z_{n+1} = r z_{n}
# $$
# 
# Where $r$ is a multiplicative factor ($<1$ for contracting, $>1$ for expanding cells). $x$ and $y$ increments can be altered as well, but we choose not to, as the data are not spatially clustered (i.e., the entire region is an area of interest).

# In[6]:


# %%
# Choose the cell size in the three cartesian directions
dx = inc
dy = inc
dz = 10
# This refers to the core block, where the cell size stays constant
nz_core = 10
# The padding cells are where every successive cell will be multiplied by a factor (fact)
nz_pad = 18
fact = 1.1

# Set the cell sizes to a constant value in the x and y directions and expanding in the z direction.
inv_hx = dx * np.ones(nx_new)
inv_hy = dy * np.ones(ny_new)
inv_hz = [(dz, nz_pad, -fact), (dz, nz_core), (dz, nz_pad, fact)]

# Create the inverse tensor mesh
inv_mesh = ds.TensorMesh([inv_hx, inv_hy, inv_hz], x0=[np.min(inv_topo[:, 0]), np.min(inv_topo[:, 1]), "C"])

# Drape the topography over the mesh
actv = active_from_xyz(inv_mesh, inv_topo)
ndv = np.nan
actvMap = maps.InjectActiveCells(inv_mesh, actv, ndv)
nactv = int(actv.sum())

# Checking the mesh extent and cell sizes
inv_mesh

# %%
# ### Visualize the mesh
# 
# The ``discretize.TensorMesh.plot_slice()`` utility will plot a slice of the Tensor Mesh object generated above.

# In[7]:


# %%
# Create a background model.
bg = np.ones(nactv)
mod = actvMap * bg

# Plot the mesh slice.
fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
cplot = inv_mesh.plot_slice(mod, grid=True, ax=ax, normal="Y", ind=22)
ax.set_title('Tensor Mesh Slice')
ax.set_xlabel('x (m)')
ax.set_ylabel('z (m)')
ax.set_xlim([158000, 159000])
ax.set_ylim([-500, 100])
ax.ticklabel_format(axis="both")
ax.set_aspect('equal')
plt.savefig(path_to_output + "/" + name + "_TreeMeshSlice.pdf", bbox_inches="tight")
plt.show()

# %%
# ### Step 5: Set up the gravity inverse problem
# 
# The first requirement for a successful inversion is a working forward model. SimPEG's ``pf.gravity`` module is used to generate a forward model based on the mesh and the receiver locations, and is set to compute the components of Gravity gradients in the order of the input data. The code uses maps to keep things clean. More details about SimPEG maps and models can be found [here](https://docs.simpeg.xyz/content/tutorials/01-models_mapping/index.html).

# In[8]:


# %%
# Create wires for the density model
wires = maps.Wires(("density", nactv))

# Create a density map
density_map = actvMap * wires.density

# Create an identity map
identity_map = maps.IdentityMap(nP=nactv)

# Components of the data used as input
gravity_components = ["gxx", "gyy", "gxz", "gyz", "gxy"]

# Receiver locations (25 meters above the topography)
gravity_receiver_locations = inv_topo + 120.

# Create a gravity receiver list
gravity_receivers = pf.gravity.receivers.Point(gravity_receiver_locations, components=gravity_components)
gravity_receiver_list = [gravity_receivers]

# Create a gravity source field
gravity_source_field = pf.gravity.sources.SourceField(receiver_list=gravity_receiver_list)

# Define the gravity survey
gravity_survey = pf.gravity.survey.Survey(gravity_source_field)

# Set up the gravity simulation problem
gravity_problem = grav.simulation.Simulation3DIntegral(
    inv_mesh,
    survey=gravity_survey,
    rhoMap=wires.density,
    ind_active=actv
)

# Create a gravity data object, with the relative errors and a standard noise floor
gravity_data = data.Data(gravity_survey, dobs=grav_vec.flatten(), noise_floor=5, relative_error=0.1)

# Define the misfits associated with the gravity data
gravity_misfit = data_misfit.L2DataMisfit(data=gravity_data, simulation=gravity_problem)

# %%
# ### Step 6: Setting up the Gaussian Mixture Model (GMM) Prior
# 
# In PGI, the petrophysical data is used as a constraint in the form of a [Gaussian Mixture Model](https://scikit-learn.org/stable/modules/mixture.html). A GMM is a multimodal probabilistic distribution which is just a weighted sum of multiple gaussian distributions. Given the number of rock units ($n$), the petrophysical distribution can be displayed as an $n$-modal GMM. If you have no petrophysical information available, you can initialize the GMM as below. If you do have a petrophysical dataset, you can fit the GMM to said dataset using the ``gmmref.fit()`` method.

# In[9]:


# %%
# Number of rock units and number of physical properties
num_rock_units = 5
num_physical_props = 1

# Create a weighted Gaussian mixture model with specified parameters
gmmref = utils.WeightedGaussianMixture(
    n_components=num_rock_units,
    mesh=inv_mesh,
    actv=actv,
    covariance_type="full",
)

# Set the background density
background_density = 0.0

# Initialize the GMM fit with random samples, mesh size, and number of physical properties
gmmref.fit(np.random.randn(nactv, num_physical_props))

# Set the mean values of physical property contrasts for each rock unit
# One value (density) for each rock unit
gmmref.means_ = np.c_[
    [-0.4],
    [-0.2],
    [0.0],
    [0.2],
    [0.4],
].T

# Set the original variance for density
density_variance = 8e-5

# Set the covariances of physical properties for each rock unit
# NOTE: Since we don't have petrophysical information for this example, we keep the covariances
# same for every unit. The GMM will update itself during the inversion.
gmmref.covariances_ = np.array(
    [
        [[density_variance]],
        [[density_variance]],
        [[density_variance]],
        [[density_variance]],
        [[density_variance]]
    ]
)

# Compute the precision (inverse covariance) of each cluster
gmmref.compute_clusters_precisions()

# Set the weights for each rock unit
# NOTE: This determines the size of each peak of the n-modal GMM
gmmref.weights_ = np.ones((nactv, 1)) * np.c_[0.125, 0.125, 0.5, 0.125, 0.125]

# %%
# ### Plot the 1D-GMM

# In[10]:


# %%
fig = plt.figure(figsize=(6, 6))
ax = gmmref.plot_pdf(flag2d=False, plotting_precision=1000, padding=0.2)
ax[0].set_xlabel(r"Density Contrast (g/cc)")
ax[0].set_ylabel(r"Probability Density Values")
ax[0].get_legend().remove()
ax[0].set_title(r"Initial DelRho Distribution")
ax[0].ticklabel_format(axis="both", style="scientific", scilimits=(0, 0))
ax[0].set_aspect(1 / 20)
plt.savefig(path_to_output + "/" + name + "_Init_GMM.pdf", bbox_inches="tight")
plt.show()

# %%
# ### Step 7: Setting the hyper-parameters and the sensitivity weights
# 
# Every PGI is a set of three Maximum-A-Posteriori ([MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation)) problems, being solved iteratively. The solver tries to minimize the L2 error of an objective function containing both the FTG Data and the petrophysical GMM. In this section we tune the necessary hyper-parameters, as well as initialise the necessary weights for every voxel (as the contribution of every voxel is dependent on it's depth from the surface). The regularization smallness ($\alpha_s$) and smoothness ($\alpha_i,\ i = x, y, z$) are initialised here. Please check [here](https://giftoolscookbook.readthedocs.io/en/latest/content/fundamentals/index.html) for the physical meaning of these parameters and the fundamentals of a Tikhonov regularized inversion.
# 
# <strong> NOTE </strong> : The smoothness parameters ($\alpha_i$) are static and hence need to be fine-tuned through trial and error.
# 

# In[11]:


# %%
# Initial Model
initial_model = np.r_[background_density * np.ones(actvMap.nP)]

# %%
# Sensitivity weighting
# Compute the sensitivity weights for each cell based on the gravity problem's sensitivity matrix (G)
# This is done by computing the square root of the sum of the squared elements of G for each cell,
# and then normalizing by the cell volumes and the maximum weight value.

# %%
sensitivity_weights_gravity = np.sum(gravity_problem.G ** 2.0, axis=0) ** 0.5 / (inv_mesh.cell_volumes[actv])
sensitivity_weights_gravity = sensitivity_weights_gravity / np.nanmax(sensitivity_weights_gravity)

# Regularization multipliers
smallness_multiplier = 1e-4
smoothness_x_multiplier = 4
smoothness_y_multiplier = 4
smoothness_z_multiplier = 0.25

# Create joint PGI regularization with smoothness
regularization_term = regularization.PGI(
    gmmref=gmmref,
    mesh=inv_mesh,
    wiresmap=wires,
    maplist=[identity_map],
    active_cells=actv,
    alpha_pgi=smallness_multiplier,
    alpha_x=smoothness_x_multiplier,
    alpha_y=smoothness_y_multiplier,
    alpha_z=smoothness_z_multiplier,
    # The second derivative smoothnesses are kept to be zero, since we
    # don't want to include second derivatives in the objective function
    alpha_xx=0.0,
    alpha_yy=0.0,
    alpha_zz=0.0,
    weights_list=[sensitivity_weights_gravity]
)

# %%
# ### Step 8: Initialize the directives
# 
# The directives include a set of instructions on the bounds, the update factors and other hyperparameters for the inversion solver.

# In[12]:


# %%
# Estimate smoothness multipliers
alphas_directive = directives.AlphasSmoothEstimate_ByEig(verbose=True)

# Initialize beta and beta/alpha_s schedule
beta_directive = directives.BetaEstimate_ByEig(beta0_ratio=1e-2)
beta_schedule = directives.PGI_BetaAlphaSchedule(
    coolingFactor=16.0,
    tolerance=0.2,
    progress=0.2,
    verbose=True,
)

# Define target misfits for geophysical and petrophysical data
target_misfits = directives.MultiTargetMisfits(verbose=True)

# Add reference model once stable
mref_in_smooth = directives.PGI_AddMrefInSmooth(wait_till_stable=True, verbose=True)

# Update smallness parameters, keeping GMM fixed (L2 Approx of PGI)
update_smallness_directive = directives.PGI_UpdateParameters(
    update_gmm=True,
    kappa=0,
    nu=0.5,
    zeta=0
)

# Update preconditioner
update_preconditioner = directives.UpdatePreconditioner()

# Save iteration results
save_iteration_directive = directives.SaveOutputEveryIteration(name=name, directory=path_to_output)

# Save model iterations
save_model_directive = directives.SaveModelEveryIteration(name=name, directory=path_to_mod_iterations)

# Optimization options for the inversion
lower_bound = np.r_[-1.0 * np.ones(actvMap.nP)]
upper_bound = np.r_[1.0 * np.ones(actvMap.nP)]
optimizer = optimization.ProjectedGNCG(
    maxIter=20,
    lower=lower_bound,
    upper=upper_bound,
    maxIterLS=20,
    maxIterCG=100,
    tolCG=1e-4,
)

# Inverse problem setup
inverse_prob = inverse_problem.BaseInvProblem(gravity_misfit, regularization_term, optimizer)
inversion_algo = inversion.BaseInversion(
    inverse_prob,
    directiveList=[
        alphas_directive,
        beta_directive,
        update_smallness_directive,
        target_misfits,
        beta_schedule,
        mref_in_smooth,
        update_preconditioner,
        save_iteration_directive,
        save_model_directive
    ],
)

# %%
# ### Run the inversion! 

# In[13]:


# %%
# Run the inversion
inverted_model = inversion_algo.run(initial_model)

# %%
# ### Visualize the 3D Model

# In[14]:


# %%
set = 1
save_plots = True

# Indices of the depth sections
ind_plot_x = int(len(inv_mesh.cell_centers_x) / 6) + 25
ind_plot_y = int(len(inv_mesh.cell_centers_y) / 2)
ind_plot_z = int(len(inv_mesh.cell_centers_z) / 2) - 15

ind_plot = [ind_plot_x, ind_plot_y, ind_plot_z]

# Extract the results
inverted_density_model = density_map * inverted_model
quasi_geology_model = actvMap * regularization_term.objfcts[0].compute_quasi_geology_model()

# Plot Density Contrast Model (Z)
normal = "Z"
model_to_plot = inverted_density_model
SH.plot_model_slice(inv_mesh, actv, model_to_plot, normal, ind_plot, [-1.0, 1.0], set, sec_loc=True, gdlines=True, which_prop="Den", cmap="Spectral", save_plt=save_plots, path_to_output=path_to_output, name=name)

# Plot Inverted Model Slices (Y)
normal = "X"
model_to_plot = inverted_density_model
SH.plot_model_slice(inv_mesh, actv, model_to_plot, normal, ind_plot, [-1.0, 1.0], set, sec_loc=True, gdlines=True, which_prop="Den", cmap="Spectral", save_plt=save_plots, path_to_output=path_to_output, name=name)

# Plot Inverted Model Slices (X)
normal = "Y"
model_to_plot = inverted_density_model
SH.plot_model_slice(inv_mesh, actv, model_to_plot, normal, ind_plot, [-1.0, 1.0], set, sec_loc=True, gdlines=True, which_prop="Den", cmap="Spectral", save_plt=save_plots, path_to_output=path_to_output, name=name)

# %%
# ### Visualize the Quasi-Geological model
# 
# The quasi-geological model (as given by [Li et al., 2019](https://doi.org/10.1190/tle38010060.1)), is a way of visualizing an inverted model by using the learned GMM as a classifier for all of the voxels in the model. Therefore, instead of density values, a Quasi-Geological model shows the different rock units as separated by the GMM classifier.

# In[15]:


# %%
# Plot Density Contrast Model (Z)
normal = "Z"
model_to_plot = quasi_geology_model
SH.plot_model_slice(inv_mesh, actv, model_to_plot, normal, ind_plot, [0, 4], set, sec_loc=True, gdlines=True, which_prop="QGM", cmap="jet", save_plt=save_plots, path_to_output=path_to_output, name=name)

# Plot Inverted Model Slices (Y)
normal = "X"
model_to_plot = quasi_geology_model
SH.plot_model_slice(inv_mesh, actv, model_to_plot, normal, ind_plot, [0, 4], set, sec_loc=True, gdlines=True, which_prop="QGM", cmap="jet", save_plt=save_plots, path_to_output=path_to_output, name=name)

# Plot Inverted Model Slices (X)
normal = "Y"
model_to_plot = quasi_geology_model
SH.plot_model_slice(inv_mesh, actv, model_to_plot, normal, ind_plot, [0, 4], set, sec_loc=True, gdlines=True, which_prop="QGM", cmap="jet", save_plt=save_plots, path_to_output=path_to_output, name=name)

# %%
# ### Plot the updated 1D-GMM

# In[16]:


# %%
# Plot the learned GMM
fig = plt.figure(figsize=(6, 6))
ax = regularization_term.objfcts[0].gmm.plot_pdf(flag2d=False, plotting_precision=500, padding=0.4)
ax[0].hist(inverted_density_model[actv], density=True, bins=1000)
ax[0].set_xlabel(r"Density contrast (g/cc)")
ax[0].set_ylabel(r"Probability Density Values")
ax[0].get_legend().remove()
ax[0].set_title(r"Learned DelRho Distribution")
ax[0].ticklabel_format(axis="both", style="scientific", scilimits=(0, 0))
plt.savefig(path_to_output + "/" + name + "_Learned_GMM.pdf", bbox_inches="tight")
plt.show()
