import numpy as np
from data_handling import *
from star_structure import Star
from star_family_structure import StarFamily
from eos_library import BSk20EOS


# Set the path of the figures
figures_path = "figures/app_bsk20_eos"

# Open the .csv file with the expected Mass vs Radius curve (units in solar mass and km)
(expected_radius, expected_mass) = csv_to_arrays(
    fname='data/BSk20_M_vs_R.csv')

# Set the rho_space
max_rho = 2.181e15 * MASS_DENSITY_CGS_TO_GU     # Maximum density [m^-2]
rho_space = max_rho * np.logspace(-9.0, 0.0, 10000)

# Create the EOS object
eos = BSk20EOS(rho_space)

# Set the pressure at the center and surface of the star
rho_center = max_rho                            # Central density [m^-2]
p_center = eos.p(rho_center)                    # Central pressure [m^-2]
p_surface = 1e23 * PRESSURE_CGS_TO_GU           # Surface pressure [m^-2]

# Single star

# Define the object
star_object = Star(eos, p_center, p_surface)

# Solve the TOV equation
star_object.solve_tov(max_step=100.0)

# Plot the star structure curves
star_object.plot_star_structure_curves(figures_path)

# Star Family

# Set the p_center space that characterizes the star family
p_center_space = p_center * np.logspace(-2.2, 0.0, 50)

# Create the star family object
star_family_object = StarFamily(eos, p_center_space, p_surface)

# Solve the TOV equation
star_family_object.solve_tov(max_step=100.0)

# Plot the calculated and expected Mass vs Radius curves
star_family_object.plot_curve(
    x_axis="R", y_axis="M", figure_path=figures_path + "/comparison", expected_x=expected_radius, expected_y=expected_mass)

# Plot all curves
star_family_object.plot_all_curves(figures_path)
