Geostatistical Visualization of Mineral Distribution

Overview:

This project demonstrates how to model and visualize the spatial distribution of mineral concentrations using geostatistics. It implements a complete workflow from data generation to spatial prediction using Kriging interpolation.
The goal is to simulate and analyze how mineral values vary across space and to produce both prediction maps and uncertainty maps.

Features:

Synthetic geological dataset generation
Experimental variogram computation
Variogram modeling (spherical model)
Ordinary Kriging interpolation
Visualization of:
Sample data points
Interpolated mineral distribution
Prediction uncertainty

Concepts Covered:

Spatial dependence
Variogram (nugget, sill, range)
Geostatistical modeling
Kriging interpolation

Technologies Used:

Python
NumPy
Pandas
Matplotlib
SciPy
PyKrige

Workflow:

Generate or load spatial data (x, y, z)
Compute pairwise distances and semivariance
Build the experimental variogram
Fit a theoretical variogram model
Apply Ordinary Kriging
Visualize prediction and uncertainty

Installation:

pip install numpy pandas matplotlib scipy pykrige

Usage:

Run the main script:
python analysis.py
You will see:
Sample point distribution
Variogram plot
Kriging prediction map
Uncertainty map

Applications:

Mineral exploration
Ore body modeling
Environmental geochemistry
Spatial data science

Future Improvements:

Use real geological datasets
Add 3D Kriging
Implement anisotropy
Build interactive visualizations (Plotly)
Multi-variable analysisAuthor

Vitaly Andriyko
(Geology + Mathematics + Programming)

License
This project is open-source and available under the MIT License.
