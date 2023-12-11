"""
Gravity data
============


Read gravity data from Stonepark project and convert to subsurface

"""
import numpy as np

import subsurface as ss
from dotenv import dotenv_values
import pandas as pd
import matplotlib.pyplot as plt


# %%
config = dotenv_values()
path = config.get("PATH_TO_STONEPARK_BOUGUER")

# %% 
df = pd.read_csv(path, sep=',', header=0)

interesting_columns = df[['X', 'Y', 'Bouguer_267_complete']]

omf_extent = np.array([559902.8297554839, 564955.6824703198 * 1.01, 644278.2600910577, 650608.7353560531, -1753.4, 160.3266042185512])

# %%
# Plot gravity data
# Create scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(df['X'], df['Y'], c=df['Bouguer_267_complete'])
plt.title('Gravity Data Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Bouguer_267_complete')

# Set extent of the plot
plt.xlim(omf_extent[0], omf_extent[1])
plt.ylim(omf_extent[2], omf_extent[3])

# Hide axis labels
plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.yaxis.set_ticklabels([])

# Display plot
plt.show()

