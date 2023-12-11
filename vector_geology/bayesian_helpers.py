import numpy as np
from sklearn.linear_model import LinearRegression


def calculate_scale_shift(a: np.ndarray, b: np.ndarray) -> tuple:
    # Reshape arrays for sklearn
    a_reshaped = a.reshape(-1, 1)
    b_reshaped = b.reshape(-1, 1)

    # Linear regression
    model = LinearRegression().fit(a_reshaped, b_reshaped)

    # Get scale and shift
    s = model.coef_[0][0]
    c = model.intercept_[0]

    return s, c


def gaussian_kernel(locations, length_scale, variance):
    import torch
    # Compute the squared Euclidean distance between each pair of points
    locations = torch.tensor(locations.values)
    distance_squared = torch.cdist(locations, locations, p=2).pow(2)
    # Compute the covariance matrix using the Gaussian kernel
    covariance_matrix = variance * torch.exp(-0.5 * distance_squared / length_scale**2)
    return covariance_matrix