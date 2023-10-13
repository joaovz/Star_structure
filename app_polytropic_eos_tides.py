import numpy as np
import matplotlib.pyplot as plt
from data_handling import *
from perfect_fluid_star_family_tides import DeformedStarFamily
from eos_library import PolytropicEOS


# Set the figures path
figures_path = "figures/app_polytropic_eos"

# Open the .dat file with the k2 vs C curve for the polytropic n=1 EOS
expected_C, expected_k2 = dat_to_array(
    fname='data/k2_vs_c_polytropic_n_1.dat',
    usecols=(0, 1))

# Create the EOS object
eos = PolytropicEOS(k=1.0e8, n=1)

# Set the pressure at the center and surface of the star
rho_center = 4.3e-9             # Center density [m^-2]
p_center = eos.p(rho_center)    # Center pressure [m^-2]
p_surface = 0.0                 # Surface pressure [m^-2]

# Print the values used for p_center and p_surface
print(f"p_center = {p_center} [m^-2]")
print(f"p_surface = {p_surface} [m^-2]")

# Set the p_center space that characterizes the star family
p_center_space = p_center * np.logspace(-5.0, 0.0, 50)

# Define the object
star_family_object = DeformedStarFamily(eos.rho, p_center_space, p_surface)

# Solve the TOV equation, and the tidal equation
star_family_object.solve_tidal(max_step=100.0)

# Plot the calculated and expected Love number vs Compactness curves
star_family_object.plot_curve(
    x_axis="C", y_axis="k2", figure_path=figures_path + "/comparison", expected_x=expected_C, expected_y=expected_k2)

# Plot all curves
star_family_object.plot_all_curves(figures_path)
