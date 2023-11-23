import numpy as np
from data_handling import *
from perfect_fluid_star_family_tides import DeformedStarFamily
from eos_library import BSk20EOS


# Set the path of the figures
figures_path = "figures/app_bsk20_eos"

# Open the .csv file with the expected Mass vs Radius curve (units in solar mass and km)
expected_radius, expected_mass = csv_to_arrays(
    fname='data/BSk20_M_vs_R.csv')

# Open the .csv file with the expected Lambda vs Compactness curve
expected_C, expected_Lambda = csv_to_arrays(
    fname='data/BSk20_Lambda_vs_C.csv')
expected_k2 = (3 / 2) * expected_C**5 * expected_Lambda

# Set the rho_space
max_rho = 1.619e-9      # Maximum density [m^-2]
rho_space = max_rho * np.logspace(-9.0, 0.0, 10000)

# Create the EOS object
eos = BSk20EOS(rho_space)

# Set the pressure at the center and surface of the star
rho_center = max_rho            # Center density [m^-2]
p_center = eos.p(rho_center)    # Center pressure [m^-2]
p_surface = 1e-22               # Surface pressure [m^-2]

# Print the values used for p_center and p_surface
print(f"p_center = {p_center} [m^-2]")
print(f"p_surface = {p_surface} [m^-2]")

# Set the p_center space that characterizes the star family
p_center_space = p_center * np.logspace(-2.2, 0.0, 50)

# Define the object
star_family_object = DeformedStarFamily(eos, p_center_space, p_surface)

# Solve the TOV equation and the tidal equation
star_family_object.solve_tidal(max_step=100.0)

# Plot the calculated and expected Mass vs Radius curves
star_family_object.plot_curve(
    x_axis="R", y_axis="M", figure_path=figures_path + "/comparison", expected_x=expected_radius, expected_y=expected_mass)

# Plot the calculated and expected Love number vs Compactness curves
star_family_object.plot_curve(
    x_axis="C", y_axis="k2", figure_path=figures_path + "/comparison", expected_x=expected_C, expected_y=expected_k2)

# Plot all curves
star_family_object.plot_all_curves(figures_path)
