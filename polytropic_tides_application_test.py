import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from perfect_fluid_star_family_tides import DeformedStarFamily
from data_handling import *


# Open the .dat file with the k2 vs C curve for the polytropic n=1 EOS
expected_C, expected_k2 = dat_to_array(
    fname='data/k2_vs_c_polytropic_n_1.dat',
    usecols=(0, 1))

# Set the EOS and pressure at the center and surface of the star
def rho(p):
    c = 1.0e8       # [m^2]
    return (np.abs(p / c))**(1 / 2)

def p(rho):
    c = 1.0e8       # [m^2]
    return c * rho**2

rho_center = 2.376364e-9        # Center density [m^-2]
p_center = p(rho_center)        # Center pressure [m^-2]
p_surface = 0.0                 # Surface pressure [m^-2]

# Print the values used for p_center and p_surface
print(f"p_center = {p_center} [m^-2]")
print(f"p_surface = {p_surface} [m^-2]")

# Set the p_center space that characterizes the star family
p_center_space = p_center * np.logspace(-4.0, 1.0, 50)

# Define the object
star_family_object = DeformedStarFamily(rho, p_center_space, p_surface)

# Solve the tidal equation
start_time = time.process_time()
star_family_object.solve_tidal(max_step=1.0)
end_time = time.process_time()
print(f"Tidal solver execution time: {end_time - start_time} s")

# Show the radius-mass curve
star_family_object.plot_radius_mass_curve()

# Show the k2 curve
star_family_object.plot_k2_curve(show_plot=False)

# Add the expected k2-C curve to the plot, enable legend, and show the plot
plt.plot(expected_C, expected_k2, linewidth=1, label="Expected curve", marker='.')
plt.legend()
plt.show()
