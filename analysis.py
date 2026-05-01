import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from pykrige.ok import OrdinaryKriging

np.random.seed(42)

# ----------------------------
# Data generation
# ----------------------------
def generate_data(n=50):
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    z = np.sin(x / 10) + np.cos(y / 10) + np.random.normal(0, 0.2, n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


# ----------------------------
# Variogram computation
# ----------------------------
def compute_variogram(coords, values, n_bins=15, max_range=None):
    distances = pdist(coords)
    diffs = pdist(values.reshape(-1, 1))
    semivariance = 0.5 * diffs**2

    if max_range is None:
        max_range = distances.max() * 0.5  # cutoff for stability

    mask = distances <= max_range
    distances = distances[mask]
    semivariance = semivariance[mask]

    bins = np.linspace(0, max_range, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    gamma, _, _ = binned_statistic(
        distances,
        semivariance,
        statistic="mean",
        bins=bins
    )

    return bin_centers, gamma


# ----------------------------
# Spherical model
# ----------------------------
def spherical_model(h, nugget, sill, vrange):
    h = np.asarray(h)
    hr = h / vrange

    return np.where(
        h <= vrange,
        nugget + (sill - nugget) * (1.5 * hr - 0.5 * hr**3),
        sill
    )


# ----------------------------
# Fit variogram model properly
# ----------------------------
def fit_variogram(bin_centers, gamma):
    mask = ~np.isnan(gamma)

    x = bin_centers[mask]
    y = gamma[mask]

    # initial guess
    p0 = [
        np.min(y),
        np.max(y),
        x.max() * 0.5
    ]

    params, _ = curve_fit(spherical_model, x, y, p0=p0, maxfev=5000)
    return params


# ----------------------------
# Main pipeline
# ----------------------------
data = generate_data()

coords = data[["x", "y"]].to_numpy()
values = data["z"].to_numpy()

# Variogram
bin_centers, gamma = compute_variogram(coords, values)

# Fit model
nugget, sill, vrange = fit_variogram(bin_centers, gamma)

print(f"Fitted parameters:\nNugget={nugget:.3f}, Sill={sill:.3f}, Range={vrange:.3f}")

# Smooth curve
h_vals = np.linspace(0, bin_centers.max(), 200)
model_vals = spherical_model(h_vals, nugget, sill, vrange)

# Plot variogram
plt.figure()
plt.scatter(bin_centers, gamma, label="Experimental")
plt.plot(h_vals, model_vals, "r", label="Fitted spherical")
plt.xlabel("Distance (h)")
plt.ylabel("Semivariance")
plt.title("Variogram Model Fit")
plt.grid(True)
plt.legend()
plt.show()


# ----------------------------
# Kriging (using fitted params)
# ----------------------------
x = data["x"].to_numpy()
y = data["y"].to_numpy()
z = data["z"].to_numpy()

grid_x = np.linspace(0, 100, 100)
grid_y = np.linspace(0, 100, 100)

OK = OrdinaryKriging(
    x, y, z,
    variogram_model='spherical',
    variogram_parameters={
        'nugget': nugget,
        'sill': sill,
        'range': vrange
    },
    verbose=False,
    enable_plotting=False
)

z_pred, z_var = OK.execute('grid', grid_x, grid_y)

# Plot result
plt.figure(figsize=(10, 8))
plt.imshow(z_pred, origin='lower', extent=(0, 100, 0, 100))
plt.scatter(x, y, c=z, edgecolor='k')
plt.colorbar(label="Predicted Concentration")
plt.title("Kriging Prediction Map")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()