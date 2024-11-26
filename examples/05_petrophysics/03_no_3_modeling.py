"""
Modelling and Propagation of Legacy Petrophysical Data for Mining Exploration (1/3)
===================================================================================
**Exploratory Data Analysis and Data Integration**

Barcelona 25/09/24
GEO3BCN
Manuel David Soto, Juan Alcalde, Adrià Hernàndez-Pineda, Ramón Carbonel

"""
import dotenv
import os
#%% md
# <img src="images\logos.png" style="width:1500px">
# 
# Barcelona 25/09/24 </br>
# GEO3BCN </br>
# Manuel David Soto, Juan Alcalde, Adrià Hernàndez-Pineda, Ramón Carbonel.
# </br>
# <h1><center> Modeling and Propagation of Petrophysical Data for Mining Exploration </h1></center>
# <h1><center> 3/3 - Modeling </h1></center>
#%% md
# ## Introduction
# 
# The dispersion and scarcity of petrophysical data are well-known challenges in the mining sector. These issues are primarily driven by economic factors, but also by geological (such as sedimentary cover, weathering, erosion, or the depth of targets), geotechnical (e.g., slope or borehole stability), and even technical limitations or availability that have been resolved in other industries (for instance, sonic logs were not previously acquired due to a lack of interest in velocity field data).
# 
# To address the challenge of sparse and incomplete petrophysical data in the mining sector we have developed three Jupyter notebooks that tackle these issues through a suite of open-source Python tools. These tools support researchers in the initial exploration, visualization, and integration of data from diverse sources, filling gaps caused by technical limitations and ultimately enabling the complete modeling of missing properties (through standard and more advanced ML-based models). We applied these tools to both, recently acquired and legacy petrophysical data of two cores northwest of Collinstown (County Westmeath, Province of Leinster), Ireland, located 26 km west-northwest of the Navan mine. However, these tools are adaptable and applicable to mining data from any location.
# 
# After the exploratory data analysis (notebook 1/3) and filling the gaps in the petrophysical dataset of Collinstown (previous notebook 2/3), this third notebook is focused on modeling by Machine Learning (ML) algorithms entire missing variables, such as the Gamma Ray (GR) measurement, which is available in only one of the petrophysical boreholes. The tasks to perform in this notebook are:
# 
# * Load and merge the legacy GR data of borehole TC-3660-008 into its petrophysical dataset.
# * Use the merged data to train and evaluate different ML algorithms capable of predicting GR data in other boreholes.
# * By using the trained best model, propagate the GR to the other petrophysical borehole.
# * Evaluate the possibility of propagating this (GR) or other properties to other boreholes with less available data.
# 
# As with the previous notebooks, these tasks are performed with open-source Python tools easily accessible by any researcher through a Python installation connected to the Internet.
#%% md
# ## Variables
# 
# The dataset used in this notebook is the imputed dataset from the previous notebook (2/3), *features3*. It contains the modelable petrophysical features of the two available boreholes; TC-1319-005 and TC-3660-008. Hole (text object) and Len (float) variables are for reference, Form (text object) is a categorical variable representing the Major Formations:
# 
# </br>
# 
# | Name | Explanation | Unit |
# | --- | --- | --- |
# | Hole | Hole name | - |
# | From | Top of the sample | m |
# | Len | Lenght of the core sample | cm |
# | Den | Density | g/cm3 
# | Vp | Compressional velocity| m/s |
# | Vs | Shear velocity | m/s |
# | Mag | Magnetic susceptibility | - |
# | Ip | Chargeability or induced polarization | mv/V |
# | Res | Resistivity | ohm·m |
# | Form | Major formations or zone along the hole |
# 
# </br>
# The GR measurement from keyhole TC-3660-008 is the property we want to transfer to the other borehole, TC-1319-005. The challenge lies in the fact that both datasets have different ranges and intervals, and is not possible to make an ML model unless both datasets (new petrophysical features and legacy GR) are integrated into a single dataframe.
# </br>
#%% md
# ## Libraries
# 
# The following are the Python libraries used along this notebook. PSL are Python Standard Libraries, UDL are User Defined Libraries, and PEL are Python External Libraries:
#%%
# PLS
import sys
import warnings

#UDL
from vector_geology import basic_stat, geo

# PEL- Basic
import numpy as np
import pandas as pd
from tabulate import tabulate
import json

# PEL - Plotting
import matplotlib.pyplot as plt

# PEL - Filtering
from scipy.signal import butter, filtfilt

# # PEL - Data selection, transformation, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# # PEL- ML algorithms
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# PEL - Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
#%% md
# ## Settings
#%%
# Activation of qt GUI
# Seed of random process
seed = 123

# Warning suppression
warnings.filterwarnings("ignore")
#%% md
# ## Data Loading
#%%
# features3, imputed features, data load

features3 = pd.read_csv('Output/features3.csv', index_col=0)
features3.head()
#%%
features3.describe()
#%%
# Columns in the features3

features3.columns
#%%
# Imputed features of borehole TC-3660-008

features3_8 = features3[features3.Hole=='TC-3660-008']
features3_8 = features3_8.reset_index(drop=True)
features3_8
#%%
# Depth information of feature3

print('Petrophysical features of TC-360-008', '\n')
print('Top:', features3_8.From[0])
print('Base:', features3_8.From[len(features3_8) - 1], '\n')

print('Length:', len(features3_8))
print('Range:', features3_8.From[len(features3_8) - 1] - features3_8.From[0], '\n')
print('First step: {:.1f}'.format(features3_8.From[1] - features3_8.From[0]))

# Mean step
steps = []
for i in range(len(features3_8) - 1):
    steps.append(features3_8.From[i+1] - features3_8.From[i])
print('Mean step: {:.2f}'.format(np.mean(steps)))
#%%
# Legacy GR data of different holes
dotenv.load_dotenv()
base_path = os.getenv("PATH_TO_COLLINSTOWN_PETRO")
gr = pd.read_csv(f'{base_path}/collinstown_Gamma.csv')
gr.head()
#%%
# Different holes in the legacy GR dataframe

gr.HOLEID.unique()
#%%
# Legacy GR of borehole TC-3660-008

gr8 = gr[gr.HOLEID == 'TC-3660-008'].reset_index(drop=True)
gr8
#%%
gr8.describe()
#%%
# Depth information of the legacy GR

print('Legacy GR of TC-360-008', '\n')

print('Top:', gr8.DEPTH[0])
print('Base:', gr8.DEPTH[len(gr8) - 1], '\n')

print('Length:', len(gr8))
print('Range:', gr8.DEPTH[len(gr8) - 1] - gr8.DEPTH[2], '\n')

print('First step: {:.1f}'.format(gr8.DEPTH[3] - gr8.DEPTH[2]))

# Mean step
steps = []
for i in range(2, len(gr8) - 1):
    steps.append(gr8.DEPTH[i+1] - gr8.DEPTH[i])
print('Mean step: {:.2f}'.format(np.mean(steps)))
#%% md
# ## Data Merging
# 
# To model a new GR in borehole TC-1319-005, the legacy GR (gr8) and petrophysical data of borehole TC-3660-008 (feature3_8) have to be integrated into a single dataframe, free of NaNs.
# 
# ### Depth Equalization
# 
# The first step in merging the process of the legacy GR into the petrophysical data of borehole TC-3660-008 is to equalize the depths of both dataframes by the use of conditional expressions:
#%%
# Equalizing depths in feature3

features3_8 = features3_8[features3_8.From <= 830.8].reset_index(drop=True)
features3_8
#%%
# New depth information of feature3

print('Petrophysical features of TC-360-008', '\n')
print('Top:', features3_8.From[0])
print('Base:', features3_8.From[len(features3_8) - 1], '\n')

print('Length:', len(features3_8))
print('Range:', features3_8.From[len(features3_8) - 1] - features3_8.From[0], '\n')
print('First step: {:.1f}'.format(features3_8.From[1] - features3_8.From[0]))

# Mean step
steps = []
for i in range(len(features3_8) - 1):
    steps.append(features3_8.From[i+1] - features3_8.From[i])
print('Mean step: {:.2f}'.format(np.mean(steps)))
#%%
# Equalizing depths in the legacy GR

gr8 = gr8[(gr8.DEPTH >= 9.6) & (gr8.DEPTH <= 828.8)].reset_index(drop=True)
gr8
#%%
# New depth information of the legacy GR

print('Legacy GR of TC-360-008', '\n')

print('Top:', gr8.DEPTH[0])
print('Base:', gr8.DEPTH[len(gr8) - 1], '\n')

print('Length:', len(gr8))
print('Range:', gr8.DEPTH[len(gr8) - 1] - gr8.DEPTH[2], '\n')

print('First step: {:.1f}'.format(gr8.DEPTH[3] - gr8.DEPTH[2]))

# Mean step
steps = []
for i in range(2, len(gr8) - 1):
    steps.append(gr8.DEPTH[i+1] - gr8.DEPTH[i])
print('Mean step: {:.2f}'.format(np.mean(steps)))
#%% md
# ### Filtering and Dowsampling
#%% md
# Now that the depths of the legacy GR and the petrophysical data of borehole TC-3660-008 have been equalized, the next challenge is to merge the legacy GR into the petrophysical data. This involves condensing the 8605 legacy GR values into a new petrophysical feature with 167 values that can be used in the models. This task has been archived by combining a Butterwort filter and a downsampling function, an ideal combination for data with fast fluctuation such as the GR log (only downsampling produces a very spiky result). The effect of the filter, downsampling, and filter followed by downsampling can be appreciated at the end of this section.
#%%
# Simplify the legacy GR

gr8 = gr8[['DEPTH', 'GeoPhys_NGAM_API']]
gr8
#%%
# Rename series in the legacy GR

gr8 = gr8.rename(columns={'DEPTH':'depth', 'GeoPhys_NGAM_API':'gr'})
gr8
#%%
# Downsampling the legacy GR but keeping the first and last value

reduc = int((len(gr8)) / 165)

gr8_middle = gr8.iloc[1:-1].iloc[::reduc, :]
gr8_down = pd.concat([gr8.iloc[[0]], gr8_middle, gr8.iloc[[-1]]])
gr8_down = gr8_down.sort_values('depth').reset_index(drop=True)
gr8_down
#%%
# Butterworth filter on the legacy GR

b, a = butter(N=2, Wn=0.02, btype='low')
filtered_gr = filtfilt(b, a, gr8.gr)

gr8['gr_fil'] = filtered_gr
gr8 = gr8.sort_values('depth').reset_index(drop=True)
gr8
#%%
# Downsampling the filtered GR

gr8_fil_down_middle = gr8.iloc[1:-1].iloc[::reduc, :]
gr8_fil_down = pd.concat([gr8.iloc[[0]], gr8_fil_down_middle, gr8.iloc[[-1]]])

# Sort by depth
gr8_fil_down = gr8_fil_down.sort_values('depth').reset_index(drop=True)
gr8_fil_down
#%%
# Plot of filter and/or downsampling

plt.figure(figsize=(15, 8))

plt.subplot(121)
plt.plot(gr8.gr, gr8.depth, label='Original')
plt.legend()
plt.grid()
plt.xlabel('GR (API)')
plt.xlabel('Depth (m)')
plt.axis([0, 120, 850, 0])
plt.ylim(850, 0)

plt.subplot(122)
plt.plot(gr8_down.gr, gr8_down.depth, label='Down Samplig')
plt.plot(gr8.gr_fil, gr8.depth, label='Butterworth Filtered')
plt.plot(gr8_fil_down.gr_fil, gr8_fil_down.depth, label='Butterworth Filtered & Down Sampling')
plt.legend()
plt.grid()
plt.xlabel('GR (API)')
plt.xlabel('Depth (m)')
plt.axis([0, 120, 850, 0])
plt.ylim(850, 0)

plt.suptitle('Borehole TC-3660-008 GR', y=0.98)
plt.tight_layout()
#%% md
# ### Integrated Dataframe
# 
# As the downsampling gave 168 values, we dropped the central value to reach 167 (the length of the petrophysical dataset) and then fused the filter and downsampled GR into the petrophysical data of borehole TC-3660-008.
#%%
# Drop of central value in the filter and downsampled GR

del_index = len(gr8_fil_down)//2
gr8_fil_down = gr8_fil_down.drop(axis=0, index=del_index).reset_index(drop=True)
gr8_fil_down
#%%
# Final integrated dataframe without NaNs

features3_8['GR'] = gr8_fil_down.gr_fil
features3_8
#%%
features3_8.info()
#%% md
# ## Modeling
# 
# Having the GR already integrated into the petrophysical data of borehole TC-3660-008 we are ready for modeling, which involves a regression analysis to generate predictive numerical outputs. The steps required are: 
# 
# * Split the data for training and test or evaluation
# * Select and run the regression models
# * Test or evaluate the results of the different models
# * With the best ML algorithm, generate a synthetic GR for borehole TC-1319-005 
#%% md
# ### Data Split
# 
# Here we selected the petrophysical features and target (GR) in borehole TC-3660-008 for modeling, leaving 20% (34) of them aside for testing or evaluation of the regression algorithms.
#%%
# features for modeling

features_model_8 = features3_8.drop(columns=['Hole', 'Len', 'Form', 'GR'])
print(features_model_8.shape)
features_model_8.head()
#%%
# target or objetive

target_8 = features3_8.GR
target_8
#%%
# Split and shape of data for training and testing

X_train, X_test, y_train, y_test = train_test_split(features_model_8, target_8, test_size=0.2, random_state=seed)

# X_train = np.array(X_train).flatten()
# y_train = np.array(y_train).flatten()

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)
#%% md
# ### Regression Models
# 
# In this analysis, we trained and evaluated nine regression models, some of which are influenced by random processes (those using the `random_state=seed` argument). After testing with different seed values (e.g., 123, 456, 789), we observed variations in the top-performing model, typically alternating between Gradient Boosting, Extreme Gradient Boosting, and Random Forest. To mitigate this variability and ensure a more reliable evaluation, we implemented cross-validation for the calculation of the metrics.
#%%
# Regression models and metrics with a fixed seed

seed = 123

models = []

models.append(('LR', LinearRegression()))
models.append(('L', Lasso()))
models.append(('R', Ridge()))
models.append(('SVR', SVR()))
models.append(('KNR', KNeighborsRegressor()))
models.append(('GB', GradientBoostingRegressor(random_state=seed)))
models.append(('DTR', DecisionTreeRegressor(random_state=seed)))
models.append(('RFR', RandomForestRegressor(random_state=seed)))
models.append(('XGB', xgb.XGBRegressor(objective ='reg:squarederror')))

headers = ['Model', 'Sum  of Residuals', 'MAE', 'MSE', 'RMSE', 'R2']
rows = []

for name, model in models:
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    
    sum_residual = sum(y_test - predict)
    mae = mean_absolute_error(y_test, predict)
    mse = mean_squared_error(y_test, predict)
    rmse = np.sqrt(mse)
    score = model.score(X_test, y_test)
    
    rows.append([name, sum_residual, mae, mse, rmse, score])

print('Seed:', seed)
print(tabulate(rows, headers=headers, tablefmt="fancy_outline"))
#%%
# Regression models and metrics with a fixed seed and cross-validation

seed = 123

models = []

models.append(('LR', LinearRegression()))
models.append(('L', Lasso()))
models.append(('R', Ridge()))
models.append(('SVR', SVR()))
models.append(('KNR', KNeighborsRegressor()))
models.append(('GB', GradientBoostingRegressor(random_state=seed)))
models.append(('DTR', DecisionTreeRegressor(random_state=seed)))
models.append(('RFR', RandomForestRegressor(random_state=seed)))
models.append(('XGB', xgb.XGBRegressor(objective ='reg:squarederror')))

headers = ['Model', 'MAE', 'MSE', 'RMSE', 'R2']
rows = []

n_folds = 5

rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))

for name, model in models:
    scores_mae = cross_val_score(model, X_train, y_train, cv=n_folds, scoring='neg_mean_absolute_error')
    scores_mse = cross_val_score(model, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')
    scores_rmse = cross_val_score(model, X_train, y_train, cv=n_folds, scoring=rmse_scorer)
    scores_r2 = cross_val_score(model, X_train, y_train, cv=n_folds, scoring='r2')
    
    mae = -np.mean(scores_mae)
    mse = -np.mean(scores_mse)
    rmse = np.mean(scores_rmse)
    r2 = np.mean(scores_r2)
    
    rows.append([name, mae, mse, rmse, r2])

print('Seed:', seed)
print(tabulate(rows, headers=headers, tablefmt="fancy_outline"))
#%% md
# ### Best Regression Model
# 
# After applying cross-validation, Linear Regression (LR) emerged as the top-performing model with an R² of 0.265, followed by Ridge Regression (R) with an R² of 0.256, and Lasso Regression (L) with an R² of 0.247.
# 
# Next, we selected features from borehole C-1319-005 to predict GR, based on the best model trained on borehole TC-3660-008. The chosen model is LR, with additional predictions generated using XGB.
#%%
# Petrophysical features of borehole TC-1319-005

features3_5 = features3[features3.Hole=='TC-1319-005']
features3_5 = features3_5.reset_index(drop=True)
features3_5
#%%
# Petrophysical features of borehole TC-1319-005 for modeling

features_model_5 = features3_5.drop(columns=['Hole', 'Len', 'Form'])
print(features_model_5.shape)
features_model_5.head()
#%%
# GR by Linear Regressor

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

gr5_lr = lr_model.predict(features_model_5)
gr5_lr
#%%
# GR by Extreme Gradient Boosting Regressor

xgb_model = xgb.XGBRegressor(objective ='reg:squarederror')
xgb_model.fit(X_train, y_train)

gr5_xgb = xgb_model.predict(features_model_5)
gr5_xgb
#%%
# Merge the two synthetic GRs into the petrophysical data of borehole TC-1319-005

features3_5['GR_lr'], features3_5['GR_xgb']  = gr5_lr, gr5_xgb
features3_5
#%%
# There are no NaNs in the merge data of borehole TC-1319-005

features3_5.info()
#%% md
# ## Understanding the Model
# 
# Beyond determining which regression model performs better, it is important to examine the coefficients of the Linear Regression (LR) model to understand the relative importance and influence of each feature. The magnitude and sign of these coefficients reveal which features contribute most to the target, whether positively or negatively.
# 
# The table and plot below (negative coefficients were inverted for the logarithm, and colored in red) show that Den has a strong negative impact, while Mag has a significant positive effect. In contrast, Vp, Vs, and Ip have much smaller impacts, and the influence of Res is insignificant. The intercept represents the baseline value when all features are zero.
#%%
X_train
#%%
# Linear regressor coefficients

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print('Features:', X_train.columns, '\n')
print("Coeficientes:", lr_model.coef_, '\n')
print("Intercepto:", lr_model.intercept_)
#%%
# Linear regressor coefficients in table

features_list = list(X_train.columns)
features_list.append('Intercepto')

coef_list = list(lr_model.coef_)
coef_list.append(lr_model.intercept_)

table = list(zip(features_list, coef_list))
print(tabulate(table, headers=['Feature', 'Coefficient'], tablefmt="fancy_outline"))
#%%
# Linear regressor coefficients in linear and log10 plot

plt.figure(figsize=(13,5))

plt.subplot(121)
plt.bar(features_list, (coef_list), label='Positive')
plt.bar(features_list[1], (coef_list[1]), color='r', label='negative')
plt.legend()
plt.ylabel('Coeficients')
plt.grid()

plt.subplot(122)
plt.bar(features_list, np.log10(coef_list))
plt.bar(features_list[1], np.log10(-coef_list[1]), color='r')
plt.bar(features_list[3], np.log10(-coef_list[3]), color='r')
plt.ylabel('Log10[Coeficients]')
plt.grid()

plt.tight_layout;
#%% md
# ## Graphic Comparison
# 
# The legacy GR data and major formation tops from borehole TC-3660-008 were integrated in the first notebook (1/3), provide a reference for comparing the new ML-generated GRs in borehole TC-1319-005. While the Linear Regression (LR) model achieved a higher R² score, the GR predictions generated by the XGB model (with an R² of only 0.170) more closely follow the patterns observed in the reference borehole TC-3660-008.
# 
# The original and ML-generated GRs are presented in the plot at the end of the section. To make the comparison easier between both ML-generated GRs, the GR generated by the XGB algorithm (in orange) was displayed to the right by adding 40 API units.
#%%
# Load of tops from borehole TC-3660-008

# Leer el archivo JSON
with open(f'{base_path}/tops_list_8.json', 'r') as json_file:
    tops_list_8 = json.load(json_file)

tops_list_8
#%%
# Tops in borehole TC-1319-005

features3_5.Form.unique()
#%%
# Extraction of top depths in borehole TC-1319-005

tops5 = pd.DataFrame(features3_5.From[features3_5.Form.ne(features3_5.Form.shift())])
tops5
#%%
# Extraction tops names in borehole TC-1319-005

tops5['Top'] = features3_5.Form[features3_5.Form.ne(features3_5.Form.shift())]
tops5
#%%
# Reset of indexes

tops5 = tops5.reset_index(drop=True)
tops5
#%%
# Colors of the formations in borehole TC-1319-005

tops5['color'] = pd.Series([ '#CCDF9A', '#8A6E9F', '#ABA16C', '#EBDE98', '#806000', '#2A7C43'])
tops5
#%%
# Bottom of the last formation

new_row5 = pd.DataFrame([{'From':1059.20, 'Top': '', 'color':''}])
new_row5
#%%
# Merge of dataframes

tops5 = pd.concat([tops5, new_row5], ignore_index=True)
tops5
#%%
# Rename of columns of the dataframe

tops5 = tops5.rename(columns={'From':'depth', 'Top':'name'})
tops5
#%%
# Convertion of the dataframe to a list of dictionaries by UDL, borehole TC-1319-005

tops_list_5 = geo.plot_tops(tops5)
tops_list_5
#%%
# Saving the tops to a json file

with open('Output/tops_list_5.json', 'w') as json_file:
    json.dump(tops_list_5, json_file, indent=4)
#%%
# Plot of GRs with formations

plt.figure(figsize=(10, 8))

plt.subplot(141)
plt.plot(gr8.iloc[:,1], gr8.iloc[:,0], label='Real', c='lightblue')
# plt.plot(gr8.iloc[:,2].rolling(window=250).mean(), gr8.iloc[:,1], label='Mean 400')
plt.xlabel('GR (API)')

# Butterworth filter
b, a = butter(N=2, Wn=0.02, btype='low')
filtered_data = filtfilt(b, a, gr8.iloc[:,1])
plt.plot(filtered_data, gr8.iloc[:,0], label='BWF')
plt.legend()
plt.grid()
plt.xlabel('GR (API)')
plt.ylabel('Depth (m)')
plt.axis([0, 120, 1100, 0])

plt.subplot(142)
for i in range(0, len(tops_list_8)):
    plt.axhspan(tops_list_8[i]['top'], tops_list_8[i]['base'], color=tops_list_8[i]['color'])
    plt.text(122, tops_list_8[i]['top'], tops_list_8[i]['name'], fontsize=6, va='center')
plt.axis([0, 120, 1100, 0])
plt.xticks([])
plt.xlabel('Formations')

plt.subplot(143)
plt.plot(features3_5.GR_lr, features3_5.From, label='LR')
plt.plot(features3_5.GR_xgb + 40, features3_5.From, label='XGB + 40')
plt.legend()
plt.grid()
plt.xlabel('GR (API)')
plt.axis([0, 120, 1100, 0])

plt.subplot(144)
for i in range(0, len(tops_list_5)):
    plt.axhspan(tops_list_5[i]['top'], tops_list_5[i]['base'], color=tops_list_5[i]['color'])
    plt.text(122, tops_list_5[i]['top'], tops_list_5[i]['name'], fontsize=7, va='center')
plt.axis([0, 120, 1100, 0])
plt.xticks([])
plt.xlabel('Formations')
 
plt.suptitle('GR and Formations in Boreholes\nTC-3660-008                                                                   TC-1319-005', y=0.98)
plt.tight_layout()
plt.savefig('Output/all_grs.png', dpi=300)
plt.show()
#%% md
# ## Observations
# 
# Here are some observations related to the tasks covered in this notebook:
# 
# * This notebook highlights the utility of open-source Python tools in addressing the issue of incomplete and sparse petrophysical data in mining. The various regression models employed demonstrate that machine learning techniques can be effective in filling data gaps, even when direct measurements are limited or unavailable.
# 
# * More complex models, such as Extreme Gradient Boosting, provided higher predictive accuracy following the pattern of the reference GR, but simpler models like Linear Regression offered better interpretability. This trade-off between accuracy and interpretability is a crucial consideration for practical applications in mining data analysis.
# 
# * The coefficients in the Linear Regression model indicated that variables such as Den and Mag significantly impact the target variable (GR), demonstrating strong correlation and critical predictive roles in petrophysical properties. In contrast, Vp and Res were less important, contributing minimally to the model's predictions.
#%% md
# ## Final Remarks
# 
# The analysis of the petrophysical dataset from Collinstown highlights challenges related to data dispersion and scarcity, as well as the potential of machine learning (ML) tools to enhance the value of mining data.
# 
# Data dispersion, particularly in Den and Vp, complicates the generation of accurate models and reflects the inherent heterogeneity of the subsurface. Variations across boreholes and core samples, along with some false anomalies from measurement issues, must be addressed to ensure data integrity and reliable modeling. Low correlation (or low covariance) in the dataset, resulting from fractures and methodological limitations, leads to models with low R² values.
# 
# ML algorithms and Python-based tools have proven valuable for integrating legacy data with new sources, thereby improving dataset quality. Techniques such as imputation and regression modeling help fill gaps and enhance the resolution of petrophysical properties. Overall, ML tools present significant opportunities to improve legacy data and support its integration with new datasets, leading to more accurate mining models.
# 
# Both this analysis and the previous notebook demonstrate the effective use of open-source Python tools and ML models to address incomplete and sparse petrophysical data in mining exploration. This adaptable workflow can be applied across various geological contexts, benefiting other mining projects as well.
#%% md
# ## Next Steps
# 
# Here are some possible next steps to build on the work from this and the previous notebooks:
# 
# * Consider exploring classification models, especially if there is interest in classifying different types of rocks or deposits based on petrophysical properties rather than predicting continuous variables.
# 
# * Explore advanced feature engineering or transformation techniques to uncover more complex relationships between variables. This could potentially improve the model’s performance.
# 
# * Further optimize the machine learning models by performing hyperparameter tuning. This can improve the accuracy of the predictions and refine the model selection process.
# 
# * Apply the propagation of missing properties using the models obtained here to other boreholes with limited data.
# 
# These steps would extend the current analysis, enhance model accuracy, and contribute to more practical applications in mining exploration.
#%%
