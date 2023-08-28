import numpy as np
import matplotlib.pyplot as plt
from data_handling import *
from perfect_fluid_star_family_tides import DeformedStarFamily
from eos_library import PolytropicEOS


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

# Show the radius-mass curve
star_family_object.plot_radius_mass_curve()

# Show the derivative of the mass with respect to rho_center curve
star_family_object.plot_dm_drho_center_curve()

# Show the k2 curve
star_family_object.plot_k2_curve(show_plot=False)

# Add the expected k2-C curve to the plot, enable legend, and show the plot
plt.plot(expected_C, expected_k2, linewidth=1, label="Expected curve", marker='.')
plt.legend()
plt.show()
