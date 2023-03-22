import numpy as np
from scipy.interpolate import CubicSpline
from star_family import StarFamily
from data_handling import *


# Open the .dat file with the EOS
rho, p = dat_to_array(
    fname='data/EOSFull_GM1_BPS.dat',
    usecols=(0, 1),
    unit_convertion=(DENSITY_CGS_TO_GU, PRESSURE_CGS_TO_GU))

# Convert the EOS to a spline function
rho_spline_function = CubicSpline(p, rho, extrapolate=False)

# Set the pressure at the center and surface of the star
p_center = p[-1]        # Center pressure [m^-2]
p_surface = 1e-12       # Surface pressure [m^-2]

# Print the values used for p_center and p_surface
print(f"p_center = {p_center} [m^-2]")
print(f"p_surface = {p_surface} [m^-2]")

# Set the p_center space that characterizes the star family
p_center_space = p_center * np.linspace(0.1, 1.0, 100)

# Create the star family object
star_family_object = StarFamily(rho_spline_function, p_center_space, p_surface)

# Solve the TOV equation
star_family_object.solve_tov(max_step=100.0)

# Show the radius-mass curve
star_family_object.show_radius_mass_curve()
