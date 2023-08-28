import numpy as np
from data_handling import *
from star_structure import Star
from star_family import StarFamily
from eos_library import TableEOS


# Create the EOS object
eos = TableEOS(fname='data/SLy4.dat')

# Set the pressure at the center and surface of the star
rho_center = 2.2e-9             # Center density [m^-2]
p_center = eos.p(rho_center)    # Center pressure [m^-2]
p_surface = 1e-22               # Surface pressure [m^-2]

# Print the values used for p_center and p_surface
print(f"p_center = {p_center} [m^-2]")
print(f"p_surface = {p_surface} [m^-2]")

# Single star

# Define the object
star_object = Star(eos.rho, p_center, p_surface)

# Solve the TOV equation
star_object.solve_tov(max_step=100.0)

# Plot the star structure curves
star_object.plot_star_structure_curves()

# Star Family

# Set the p_center space that characterizes the star family
p_center_space = p_center * np.logspace(-3.0, 0.0, 50)

# Create the star family object
star_family_object = StarFamily(eos.rho, p_center_space, p_surface)

# Solve the TOV equation
star_family_object.solve_tov(max_step=100.0)

# Plot the calculated radius-mass curve
star_family_object.plot_radius_mass_curve()

# Show the derivative of the mass with respect to rho_center curve
star_family_object.plot_dm_drho_center_curve()
