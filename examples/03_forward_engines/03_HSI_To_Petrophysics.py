"""
Prediction of P-Wave velocity from Hyperspectral Data
-----------------------------------------------------

This notebook showcases the prediction of P-Wave velocities from hyperspectral data using machine learning. As hyperspectral data captures information on the mineralogy of the rock, which in turn controls its petrophysical properties, we aim to ascertain and quantify the link between spectra and various petrophysical properties.
"""

# In[1]:


# %%
# Import the necessary modules
import numpy as np
import sklearn
# Import Scikit-learn models
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

from vector_geology.Packages import hklearn  # !(To date, Jan 2024) These packages are not public available yet. Contact the author for more information. 
from vector_geology.Packages import hycore  # !(To date, Jan 2024) These packages are not public available yet. Contact the author for more information.

# %%
# ### Step 1: Load the ``hycore.Shed`` and extract the HyLibraries
# 
# This notebook uses the ``hycore`` python package (Not yet public) to organise its data. ``hycore`` generates data structures called Sheds, making it easier to categorize and store hyperspectral data. We store the hyperspectral data in [HyLibraries](https://hifexplo.github.io/hylite/hylite/hylibrary.html) (a data structure within [hylite](https://github.com/hifexplo/hylite) for storing spectral libraries) within a Shed, with the corresponding measurement (P-Wave velocity in this case), stored as a numpy array. 

# In[2]:


# %%
# Load the train and test sheds
# TODO: Change the path to use the .env file
train_shed = hycore.loadShed("./Data/KSL133_Sonic.shed")
test_shed = hycore.loadShed("./Data/KSL_Sonic.shed")

# Load the train spectral libraries and p-wave velocities
train_fx50 = train_shed.results.FX50Lib
train_fenix = train_shed.results.FENIXLib
train_lwir = train_shed.results.LWIRLib
train_vp = train_shed.results.Vp

# Load the test spectral libraries and p-wave velocities
test_fx50 = test_shed.results.FX50Lib
test_fenix = test_shed.results.FENIXLib
test_lwir = test_shed.results.LWIRLib
test_vp = test_shed.results.Vp

# %%
# ### Step 2: Create a ``hklearn.Stack`` object
# 
# We also use the package ``hklearn`` (Not yet public), which is built on [Scikit-Learn](https://scikit-learn.org/stable/), specifically tailored towards handling hyperspectral data. ``hklearn`` organizes spectral libraries into data structures called Stack, which are built to handle multiple spectral libraries and integrate data from multiple sensors. ``Stack`` objects have inbuilt methods for carrying out hull corrections, used here to enhance the extrema, which are features of interest.

# In[3]:


# %%
# Create Stack objects and hull correct the spectra
hsi_train = hklearn.Stack(['FENIX', 'FX50', 'LWIR'], [train_fenix, train_fx50, train_lwir]).hc(ranges={'FENIX': (450., 2500.), 'FX50': (10, -10), 'LWIR': (10, -10)},
                                                                                               hull={'FENIX': 'upper', 'FX50': 'lower', 'LWIR': 'lower'})
hsi_test = hklearn.Stack(['FENIX', 'FX50', 'LWIR'], [test_fenix, test_fx50, test_lwir]).hc(ranges={'FENIX': (450., 2500.), 'FX50': (10, -10), 'LWIR': (10, -10)},
                                                                                           hull={'FENIX': 'upper', 'FX50': 'lower', 'LWIR': 'lower'})

# %%
# ### Step 3: Scaling the Y Variable
# 
# Now that we have the independent variable (spectra) ready for training, we move on to the dependent variable (P-Wave velocity). Since the P-wave velocities are present in m.s<sup>-1</sup>, and may impede the loss function from converging, we scale the velocities using the ``StandardScaler()`` from scikit-learn.

# In[4]:


# %%
# Scale Y Variable
y_scaler = sklearn.preprocessing.StandardScaler()
# NOTE: Dummy dimension is added to make the scaler work
_ = y_scaler.fit(train_vp[:, None])

# %%
# ### Step 4: Transforming the X-Variable
# 
# Since hyperspectral data contains a large number of bands (many of which are redundant), a Principal Component Analysis (PCA) is done on the dataset (please see [here](https://en.wikipedia.org/wiki/Principal_component_analysis)) to extract the principal components. A two-step PCA is carried out, the first one extracting 10 principal components from every sensor, which are then concatenated. The second PCA then extracts the 10 principal components from these concatenated components, to make sure that there isn't any inter-sensor correlation.
# 
# The ``Stack`` object stores the PCA (since the same PCA must be applied on the test set to make the results admissible). Since some of spectra may contain NaN values, which may impede the training, the Y-Variable is set using the ``Stack.set_y()`` method, which ensures that only spectra with no NaNs are being used for training.

# In[5]:


# %%
# Fit a PCA to the X variable
PCA_X = hsi_train.fit_pca(n_components=10, normalise=True)
hsi_test.set_transform(PCA_X)

# Set the Y-Variable
hsi_train.set_y(train_vp[:, None, None])
hsi_test.set_y(test_vp[:, None, None])

# %%
# ### Step 5: Initialise and train the models
# 
# We create a dictionary with the different models we wish to train and test. In this example we use a simple linear regression (``sklearn.linear_model.LinearRegression``) and a Multilayer Perceptron (``sklearn.neural_network.MLPRegressor``). Another dictionary with the parameters for optimization is initialised. 

# In[6]:


# Initialise models
models = dict(Linear=LinearRegression(),
              MLP=MLPRegressor(hidden_layer_sizes=(180,), max_iter=1000, solver='sgd', learning_rate='adaptive'))

# Define parameter ranges for each model
params = dict(Linear={'fit_intercept': [True, False]},
              MLP={"alpha": np.linspace(1e-4, 1e0)})

# %%
# ### Train the Models
# 
# We now have both the variables ready within the Stack. We fit the models one by one, using the ``Stack.fit_model()`` method, using the parameters from the ``params`` dictionary.

# In[7]:


# %%
# Loop through the models
for name, model in tqdm(models.items(), 'Fitting models', leave=True):
    hsi_train.fit_model(name, model, xtransform=True, ytransform=y_scaler, grid_search_cv=params[name], n_jobs=-1)

# %%
# ### Get a score table
# 
# Using the best fit model, get a score table by testing it on the test Stack object. The ``Stack.get_score_table()`` prints a pretty table using ``pandas``, displaying the train, 5-Fold cross validation andd test scores of the best performing model.

# In[8]:


# %%
# Print a pretty table!
hsi_train.get_score_table(hsi_test, y_test=None, style=True)
