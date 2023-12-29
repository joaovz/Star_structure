import numpy as np
from constants import *
from data_handling import csv_to_arrays
from eos_library import PolytropicEOS
from star_family_tides import DeformedStarFamily


# Set the path of the figures
figures_path = "figures/app_polytropic_eos"

# Open the .csv file with the expected Love number vs Compactness curve
(expected_C, expected_k2) = csv_to_arrays(
    fname='data/Polytropic_n_1_k2_vs_C.csv')

# Create the EOS object
eos = PolytropicEOS(k=1.0e8, n=1)

# Set the pressure at the center and surface of the star
rho_center = 5.691e15 * MASS_DENSITY_CGS_TO_GU      # Central density [m^-2]
p_center = eos.p(rho_center)                        # Central pressure [m^-2]
p_surface = 0.0                                     # Surface pressure [m^-2]

# Set the p_center space that characterizes the star family
p_center_space = p_center * np.logspace(-5.0, 0.0, 50)

# Define the object
star_family_object = DeformedStarFamily(eos, p_center_space, p_surface)

# Solve the TOV equation and the tidal equation
star_family_object.solve_tidal(max_step=100.0)

# Plot the calculated and expected Love number vs Compactness curves
star_family_object.plot_curve(
    x_axis="C", y_axis="k2", figure_path=figures_path + "/comparison", expected_x=expected_C, expected_y=expected_k2)

# Plot all curves
star_family_object.plot_all_curves(figures_path)
