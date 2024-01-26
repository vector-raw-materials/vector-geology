"""
.. _example_predicting_pwave_velocity:

Predicting P-Wave Velocity from Hyperspectral Data
==================================================

This example demonstrates the prediction of P-Wave velocities from hyperspectral data using machine learning techniques. Hyperspectral data provides detailed information on rock mineralogy, which influences its petrophysical properties. This script aims to explore and quantify the relationship between spectral features and various petrophysical properties, specifically focusing on P-Wave velocity.
"""

# %%
# Importing Necessary Libraries
# -----------------------------
# 
# First, we import the required Python packages. We use `dotenv` to manage environment variables, `numpy` and `sklearn` for numerical operations and machine learning, and custom packages `hklearn` and `hycore` for handling hyperspectral data and machine learning workflows.

from dotenv import dotenv_values
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

# Custom packages for handling hyperspectral data
# Note: As of January 2024, these packages are not publicly available. Please contact the author for more information.
from vector_geology.Packages import hklearn
from vector_geology.Packages import hycore

# %%
# Loading Data with `hycore.Shed`
# -------------------------------
# 
# The `hycore` package, which organizes hyperspectral data in a structure called Shed, is used here. A Shed makes it convenient to categorize and store hyperspectral data. We use HyLibraries (a data structure within `hylite` for storing spectral libraries) within a Shed to store the hyperspectral data. The corresponding P-Wave velocity measurements are stored as numpy arrays.

# Load the train and test sheds
config = dotenv_values()
path_to_train = config.get("PATH_TO_HSI_TRAINING")
path_to_test = config.get("PATH_TO_HSI_TESTING")

train_shed = hycore.loadShed(path_to_train)
test_shed = hycore.loadShed(path_to_test)

# Load the train and test spectral libraries and P-Wave velocities
train_fx50 = train_shed.results.FX50Lib
train_fenix = train_shed.results.FENIXLib
train_lwir = train_shed.results.LWIRLib
train_vp = train_shed.results.Vp

test_fx50 = test_shed.results.FX50Lib
test_fenix = test_shed.results.FENIXLib
test_lwir = test_shed.results.LWIRLib
test_vp = test_shed.results.Vp

# %%
# Creating a `hklearn.Stack` Object
# ---------------------------------
# 
# The `hklearn` package, designed to handle hyperspectral data, is used to organize spectral libraries into a structure called Stack. A Stack can manage multiple spectral libraries and integrate data from multiple sensors. We perform hull corrections on the spectra to enhance features of interest.

# Create Stack objects and apply hull correction to the spectra
hsi_train = hklearn.Stack(
    names=['FENIX', 'FX50', 'LWIR'],
    data=[train_fenix, train_fx50, train_lwir]).hc(
    ranges={'FENIX': (450., 2500.), 'FX50': (10, -10), 'LWIR': (10, -10)},
    hull={'FENIX': 'upper', 'FX50': 'lower', 'LWIR': 'lower'}
)

hsi_test = hklearn.Stack(
    names=['FENIX', 'FX50', 'LWIR'],
    data=[test_fenix, test_fx50, test_lwir]).hc(
    ranges={'FENIX': (450., 2500.), 'FX50': (10, -10), 'LWIR': (10, -10)},
    hull={'FENIX': 'upper', 'FX50': 'lower', 'LWIR': 'lower'}
)

# %%
# Scaling the Dependent Variable (Y)
# ----------------------------------
# 
# The P-Wave velocities are scaled to facilitate the convergence of the loss function during model training. We use the `StandardScaler` from scikit-learn for this purpose.

# Scale Y Variable (P-Wave velocity)
y_scaler = sklearn.preprocessing.StandardScaler()
_ = y_scaler.fit(train_vp[:, None])

# %%
# Transforming the Independent Variable (X)
# -----------------------------------------
# 
# Hyperspectral data often contains a large number of bands, many of which might be redundant. We apply Principal Component Analysis (PCA) to extract the most significant components. A two-step PCA is performed to ensure that there isn't any inter-sensor correlation. The `Stack` object stores the PCA, which is also applied to the test set.

# Apply PCA to the X variable (spectral data)
PCA_X = hsi_train.fit_pca(n_components=10, normalise=True)
hsi_test.set_transform(PCA_X)

# Set the Y-Variable (P-Wave velocity)
hsi_train.set_y(train_vp[:, None, None])
hsi_test.set_y(test_vp[:, None, None])

# %%
# Initializing and Training Models
# --------------------------------
# 
# Different machine learning models are initialized and trained. In this example, we use a simple linear regression (`LinearRegression`) and a Multilayer Perceptron (`MLPRegressor`) from scikit-learn. A dictionary with the parameters for optimization is also initialized.

# Initialize models
models = dict(
    Linear=LinearRegression(),
    MLP=MLPRegressor(
        hidden_layer_sizes=(180,),
        max_iter=1000,
        solver='sgd',
        learning_rate='adaptive'
    )
)

# Define parameter ranges for each model
params = dict(Linear={'fit_intercept': [True, False]},
              MLP={"alpha": np.linspace(1e-4, 1e0)})

# Train the models
for name, model in tqdm(models.items(), 'Fitting models', leave=True):
    hsi_train.fit_model(name, model, xtransform=True, ytransform=y_scaler, grid_search_cv=params[name], n_jobs=-1)

# %%
# Scoring and Evaluating Models
# -----------------------------
# 
# Finally, the trained models are scored and evaluated. The `Stack.get_score_table()` method is used to display a table with the training, cross-validation, and test scores of the best-performing model.

# Evaluate models and display score table
hsi_train.get_score_table(hsi_test, y_test=None, style=True)
