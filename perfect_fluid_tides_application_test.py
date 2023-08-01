import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from perfect_fluid_star_family_tides import DeformedStarFamily
from data_handling import *


# Open the .dat file with the EOS
rho, p = dat_to_array(
    fname='data/EOSFull_GM1_BPS.dat',
    usecols=(0, 1),
    unit_conversion=(DENSITY_CGS_TO_GU, PRESSURE_CGS_TO_GU))

# Convert the EOS to a spline function
rho_spline_function = CubicSpline(p, rho, extrapolate=False)

# Open the .dat file with the expected radius-mass curve (units in solar mass and km)
expected_mass, expected_radius = dat_to_array(
    fname='data/MIR-GM1-HT-Local.dat',
    usecols=(0, 2))

# Set the pressure at the center and surface of the star
p_center = p[-1]        # Center pressure [m^-2]
p_surface = 1e-20       # Surface pressure [m^-2]

# Print the values used for p_center and p_surface
print(f"p_center = {p_center} [m^-2]")
print(f"p_surface = {p_surface} [m^-2]")

# Set the p_center space that characterizes the star family
p_center_space = p_center * np.logspace(-4.7, 0.0, 50)

# Create the star family object
star_family_object = DeformedStarFamily(rho_spline_function, p_center_space, p_surface)

# Solve the TOV equation, and the tidal equation
star_family_object.solve_tidal(max_step=2.0)

# Plot the calculated radius-mass curve
star_family_object.plot_radius_mass_curve(show_plot=False)

# Add the expected radius-mass curve to the plot, enable legend, and show the plot
plt.plot(expected_radius, expected_mass, linewidth=1, label="Expected curve")
plt.legend()
plt.show()

# Plot the calculated k2 curve
star_family_object.plot_k2_curve()
