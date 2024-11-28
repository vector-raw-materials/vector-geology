"""
Multiphysics property prediction from hyperspectral drill core data
===================================================================
This notebook uses drill core data to train a model that predicts petrophysical properties from hyperspectral data.j
"""

#%%
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

import hklearn

#%%
# We have prepared a Stack object with the hyperspectral and petrophysical data integrated into it
# Load the Stack
dotenv.load_dotenv()
base_path = os.getenv("PATH_TO_HyTorch")

S = hklearn.Stack.load(f"{base_path}/Training_Stack")

#%%
# Get the spectra and properties (hklearn filters out the NaNs)
X = S.X() # Spectra
y = S.y() # Properties and their standard deviations

#%%
# Visualize a single spectrum
plt.figure(figsize=(4, 3))
plt.plot(S.get_wavelengths("SWIR")/1e3, S.X("SWIR")[550])
plt.plot(S.get_wavelengths("MWIR")/1e3, S.X("MWIR")[550])
plt.plot(S.get_wavelengths("LWIR")/1e3, S.X("LWIR")[550])
plt.xlabel(r"Wavelength $(\mu m)$")
plt.legend(["VNIR-SWIR", "MWIR", "LWIR"])
plt.tight_layout()
plt.show()

#%% md
# Step 1: Filtering
# -----------------
# We do two steps of filtering:
# 1. We use the standard deviations to eliminate points with lithological contacts.
# 2. We use HDBSCAN to generate clusters based on the PCA of the spectra, which eliminates 'noisy' spectra that aren't spectrally abundant

#%%
# High variance filtering
# Remove the high variance points (Using the rolling standard deviations)
keep_idx = np.logical_and(S.y()[:, 4] < 5, np.logical_and(S.y()[:, 5] < 5e-2, S.y()[:, -1] < 1000))
X = X[keep_idx]
y = y[keep_idx, :4]

#%%
# Clustering
# Fit a PCA
from hylite.filter import PCA
pca, loadings, _ = PCA(X, bands=30)
pca.data = pca.data/np.max(np.abs(pca.data), axis=0)[None, :]

# Init
clustering = HDBSCAN(10, 10)

# Fit + Predict
labels = clustering.fit_predict(np.c_[y[:, 0] * 1e-2, pca.data])
un_l, un_cts = np.unique(labels, return_counts=True)

#%% md
# Using `matplotlib.pyplot` to visualize the effect of the filtering and clustering

#%%
# Plot the properties
fig, axs = plt.subplot_mosaic([['A)', 'B)', 'C)']], layout='constrained', sharey=True, sharex=True, figsize=(8, 4))

# Original Data
label = list(axs.keys())
ax = list(axs.values())
m = ax[0].scatter(S.y()[:, 1], S.y()[:, 2], c=S.y()[:, 0]/1e3, s=3)
ax[0].set_title(r"Original Data" + "\n" + r"$(N = %d)$" % S.y().shape[0])
ax[0].set_ylabel(r"Density $(g.cm^{-3})$")
ax[0].set_title(label[0], loc='left', fontsize='medium')
cbaxes = inset_axes(ax[0], width="3%", height="37%", loc=3)
cbaxes.tick_params(labelsize=8)
plt.colorbar(cax=cbaxes, mappable=m)
cbaxes.set_ylabel(r"Depths $(km)$", fontsize=8)

# Cleaned data
m1 = ax[1].scatter(y[:, 1], y[:, 2], c=y[:, -1], s=3, cmap="cool")
ax[1].set_title(r"Cleaned Data" + "\n" + r"$(N = %d)$" % y.shape[0])
ax[1].set_xlabel(r"Slowness $(\mu s.m^{-1})$")
ax[1].set_title(label[1], loc='left', fontsize='medium')
cbaxes = inset_axes(ax[1], width="3%", height="37%", loc=3)
cbaxes.tick_params(labelsize=8)
plt.colorbar(cax=cbaxes, mappable=m1)
cbaxes.set_ylabel(r"$\gamma$ $(API)$", fontsize=8)

# Labeled data
m2 = ax[2].scatter(y[labels >= 0., 1], y[labels >= 0., 2], c=labels[labels >= 0.], s=3, cmap="turbo")
ax[2].set_title(r"Clustered Data" + "\n" + r"$(N = %d, N_c = %d)$" % (np.sum(labels >= 0.), un_l.shape[0] - 1))
ax[2].set_title(label[2], loc='left', fontsize='medium')
cbaxes = inset_axes(ax[2], width="3%", height="37%", loc=3)
cbaxes.tick_params(labelsize=8)
plt.colorbar(cax=cbaxes, mappable=m2)
cbaxes.set_ylabel(r"Class", fontsize=8)
plt.show()

#%% md
# Step 2: Extract the hyperspectral data
# --------------------------------------

#%%
# Save the labeled data (Drop the NaNs)
fin_idx = labels >= 0.
# Complete spectrum
fin_X = 1 - S.X()[keep_idx][fin_idx]
# SWIR
fin_swir = 1 - S.X(sensor="SWIR")[keep_idx][fin_idx]
# MWIR
fin_mwir = 1 - S.X(sensor="MWIR")[keep_idx][fin_idx]
# LWIR
fin_lwir = 1 - S.X(sensor="LWIR")[keep_idx][fin_idx]
# Scale the properties to keep the order of magnitude the same
fin_y = S.y()[keep_idx][fin_idx, 1:4] * np.array([1e-3, 1e-1, 1e-3])[None, :]
# Labels
fin_lbls = labels[fin_idx].astype(int)

#%% md
# Step 3: Define a shuffled Train + Validation split
# --------------------------------------------------

#%%
# Use stratified shuffle splitting
n_splits = 6
test_size = 0.25
sss = StratifiedShuffleSplit(n_splits=n_splits,
                             test_size=test_size,
                             random_state=404)

idxs = np.arange(fin_lbls.shape[0])
train_idxs = []
valid_idxs = []

for train_idx, valid_idx in sss.split(idxs, fin_lbls):
    train_idxs.append(train_idx)
    valid_idxs.append(valid_idx)
    
# Stack
train_idxs = np.vstack(train_idxs)
valid_idxs = np.vstack(valid_idxs)

#%% md
# Step 4: Define a `pytorch` model
# ------------------------------------

#%%
# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import R2Score, MeanSquaredError
import copy
from torch.utils.data import Dataset, DataLoader

# Classes
# Dataset
class MultimodalDataset(Dataset):
    def __init__(self, swir, mwir, lwir, labels, targets):
        self.swir = swir
        self.mwir = mwir
        self.lwir = lwir
        self.labels = labels
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.swir[idx], self.mwir[idx], self.lwir[idx], self.labels[idx], self.targets[idx]


class WeightedMSELoss(nn.Module):
    def __init__(self, non_neg_penalty_weight=1.0):
        super(WeightedMSELoss, self).__init__()
        self.non_neg_penalty_weight = non_neg_penalty_weight

    def forward(self, inputs, weights, targets):
        # Calculate the MSE loss for each example in the batch
        mse_loss = (inputs - targets) ** 2
        # Apply weights to the MSE loss
        weighted_mse_loss = mse_loss * weights[:, None]
        # Calculate the mean loss
        loss = weighted_mse_loss.mean()
        
        # Add non-negativity penalty
        non_neg_penalty = self.non_neg_penalty_weight * torch.sum(torch.clamp(-inputs, min=0) ** 2)
        total_loss = loss + non_neg_penalty
        
        return total_loss

class MultiHeadedMLP(nn.Module):
    def __init__(self, in_sizes, hidden_sizes, out_channels, output_size, conv_kernel_size=[3, 3, 3], conv_stride=1, conv_padding=1):
        super(MultiHeadedMLP, self).__init__()
        
        # Calculate output sizes after convolution
        self.swir_conv_output_size = self._calculate_conv_output_size(in_sizes[0], conv_kernel_size[0], conv_stride, conv_padding)
        self.mwir_conv_output_size = self._calculate_conv_output_size(in_sizes[1], conv_kernel_size[1], conv_stride, conv_padding)
        self.lwir_conv_output_size = self._calculate_conv_output_size(in_sizes[2], conv_kernel_size[2], conv_stride, conv_padding)
        
        # Define separate input heads for each band type with a conv layer
        self.swir_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=conv_kernel_size[0], stride=conv_stride, padding=conv_padding),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.swir_conv_output_size * out_channels, hidden_sizes[0]),
            nn.ReLU(),
        )
        self.mwir_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=conv_kernel_size[1], stride=conv_stride, padding=conv_padding),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.mwir_conv_output_size * out_channels, hidden_sizes[1]),
            nn.ReLU(),
        )
        self.lwir_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=conv_kernel_size[2], stride=conv_stride, padding=conv_padding),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.lwir_conv_output_size * out_channels, hidden_sizes[2]),
            nn.ReLU(),
        )
        
        # Define a shared hidden layer after combining the inputs
        combined_input_size = hidden_sizes[0] + hidden_sizes[1] + hidden_sizes[2]
        self.shared_layer = nn.Sequential(
            nn.Linear(combined_input_size, combined_input_size * 2),
            nn.ReLU(),
            nn.Linear(combined_input_size * 2, combined_input_size // 2),
            nn.ReLU(),
            nn.Linear(combined_input_size // 2, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )
    
    def _calculate_conv_output_size(self, input_size, kernel_size, stride, padding):
        return (input_size - kernel_size + 2 * padding) // stride + 1
    
    def forward(self, swir, mwir, lwir):
        # Add channel dimension for conv layer
        swir = swir.unsqueeze(1)
        mwir = mwir.unsqueeze(1)
        lwir = lwir.unsqueeze(1)
        
        swir_out = self.swir_head(swir)
        mwir_out = self.mwir_head(mwir)
        lwir_out = self.lwir_head(lwir)

        # Concatenate the outputs from each head
        combined = torch.cat((swir_out, mwir_out, lwir_out), dim=1)

        # Pass through the shared layer
        output = self.shared_layer(combined)

        return output
    
#%% md
# Initialize the model and prepare for training

#%%
# Make datasets
batch_size = 10

# Initialize a model
hidden_sizes = [32, 32, 32]
in_sizes = [S.X(sensor).shape[1] for sensor in S.get_sensors()]
output_size = 3
conv_kernel_size = [60, 40, 20]
conv_stride = 1
conv_padding = 1
out_channels = 4

model = MultiHeadedMLP(in_sizes, hidden_sizes,
                       out_channels, output_size,
                       conv_kernel_size, conv_stride,
                       conv_padding)

# Loss Function
wt_loss_fn = WeightedMSELoss(non_neg_penalty_weight=2)
loss_fn = nn.MSELoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Number of training epochs (Per fold)
n_epochs = 100
 
# Initialize parameters
best_mse = np.inf
best_weights = None
train_history = []
history = []

#%% md
# Training
#---------

#%%
# Begin Training
for j in range(n_splits):  
    # Fold training
    train_idx = train_idxs[j]
    # Fold Validation
    valid_idx = valid_idxs[j]
    
    # Get the separated datasets
    # Training
    train_X, train_swir, train_mwir, train_lwir, train_y = torch.Tensor(fin_X[train_idx]), torch.Tensor(fin_swir[train_idx]), torch.Tensor(fin_mwir[train_idx]), torch.Tensor(fin_lwir[train_idx]), torch.Tensor(fin_y[train_idx])
    # Validation
    valid_X, valid_swir, valid_mwir, valid_lwir, valid_y = torch.Tensor(fin_X[valid_idx]), torch.Tensor(fin_swir[valid_idx]), torch.Tensor(fin_mwir[valid_idx]), torch.Tensor(fin_lwir[valid_idx]), torch.Tensor(fin_y[valid_idx])

    # Compute the weights
    fold_idxs = [train_idx, valid_idx]
    weights = []

    for i in range(2):
        # Define the weights
        lbls, counts = np.unique(fin_lbls[fold_idxs[i]], return_counts=True)
        counts = 1/counts
        class_weights = counts/counts.sum()
        # Assign the weights
        loss_weights = np.array([class_weights[fin_lbls[i] == lbls] for i in range(fin_lbls[fold_idxs[i]].shape[0])])
        weights.append(torch.Tensor(loss_weights))

    train_dataset = MultimodalDataset(train_swir, train_mwir, train_lwir, weights[0], train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    with tqdm(range(n_epochs), unit=" epochs", mininterval=0, disable=False) as bar_:
        bar_.set_description(f"Training Fold {j + 1}")

        for epoch in bar_:
            model.train()

            with tqdm(train_loader, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for batch_swir, batch_mwir, batch_lwir, batch_weights, y_batch in bar:

                    # Forward pass
                    y_pred = model(batch_swir, batch_mwir, batch_lwir)

                    # Calculate Loss
                    loss = wt_loss_fn(y_pred, batch_weights, y_batch)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()

                    # Update weights
                    optimizer.step()

            # Log training loss for the epoch
            train_pred = model(train_swir, train_mwir, train_lwir)
            train_mse = loss_fn(train_pred, train_y)
            train_history.append(train_mse.item())

            # Validation Loss
            valid_pred = model(valid_swir, valid_mwir, valid_lwir)
            mse = loss_fn(valid_pred, valid_y)
            history.append(mse.item())

            if mse.item() < best_mse:
                best_mse = mse.item()
                best_weights = copy.deepcopy(model.state_dict())

            # Print progress
            bar_.set_postfix({"Training Loss" : train_mse.item(), "Validation Loss": mse.item(), "Best Loss": best_mse})
        
# Restore model with best weights
model.load_state_dict(best_weights)

#%% md
# Step 5: Compare the predictions
# -------------------------------
# To ensure reproducibility, we have included the best performing model from our study.

#%%
# Load the pre-trained model
model.load_state_dict(torch.load(f"{base_path}/KSL_MultiFold_SixFold.pth"))

#%%
# Check the predictions
input_swir = torch.Tensor(fin_swir)
input_mwir = torch.Tensor(fin_mwir)
input_lwir = torch.Tensor(fin_lwir)

# Ensure the model is in evaluation mode
model.eval()

# Run the forward pass - no need to track gradients here
with torch.no_grad():
    predictions = model(input_swir, input_mwir, input_lwir)
    
# Inverse scaler
meas_ = fin_y * np.array([1e3, 1e1, 1e3])[None, :]
pred_ = predictions * np.array([1e3, 1e1, 1e3])[None, :]

#%%
# Plot scatter
lowlims = [130, 1.75, -10]
highlims = [320, 3.4, 200]
titles = [r"Slowness", r"Density", r"Gamma-Ray"]
units = [r"$\ \mu \mathrm{s.m^{-1}}$", r"$\ \mathrm{g.cm^{-3}}$", "$\ \mathrm{API}$"]

fig, axs = plt.subplot_mosaic([['A)', 'B)', 'C)']], layout='constrained', figsize=(6, 2.2))
props = dict(boxstyle='round', facecolor='lightblue', edgecolor="lightblue", alpha=0.5)

# Original Data
label = list(axs.keys())
cax = list(axs.values())

for i in range(pred_.shape[1]):
    meas = meas_[:, i]
    pred = pred_[:, i]
    ax = cax[i]
    
    # Compute Metric
    metric = MeanSquaredError()
    metricr2 = R2Score()
    metric.update(torch.Tensor(meas), torch.Tensor(pred))
    metricr2.update(torch.Tensor(meas), torch.Tensor(pred))
    rmse = np.sqrt(metric.compute().item())
    r2 = metricr2.compute().item()
    
    ax.scatter(meas, pred, s=3)
    ax.set_title(label[i], loc='left', fontsize='medium')
    ax.set_title(titles[i])
    
    textstr = '\n'.join((
    r'$R^2=%.3f$' % (r2, ),
    r'$RMSE=%.3f$' % (rmse, ),
    ))
        
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', bbox=props)
    
    ax.axline((0, 0), slope=1, c="r")
    ax.set_xlim([lowlims[i], highlims[i]])
    ax.set_ylim([lowlims[i], highlims[i]])
    ax.set_xlabel("Measured")
    if i == 0:
        ax.set_ylabel("Predicted")
    ax.set_aspect("equal")

plt.show()
