import numpy as np
from data_handling import *
from perfect_fluid_star_family_tides import DeformedStarFamily
from eos_library import TableEOS


# Set the path of the figures
figures_path = "figures/app_table_eos"

# Open the .dat file with the expected Mass vs Radius curve (units in solar mass and km)
expected_mass, expected_radius = dat_to_array(
    fname='data/MIR-GM1-HT-Local.dat',
    usecols=(0, 2))

# Create the EOS object
eos = TableEOS(fname='data/EOSFull_GM1_BPS.dat')

# Set the pressure at the center and surface of the star
rho_center = 1.5e-9             # Center density [m^-2]
p_center = eos.p(rho_center)    # Center pressure [m^-2]
p_surface = 1e-22               # Surface pressure [m^-2]

# Print the values used for p_center and p_surface
print(f"p_center = {p_center} [m^-2]")
print(f"p_surface = {p_surface} [m^-2]")

# Set the p_center space that characterizes the star family
p_center_space = p_center * np.logspace(-2.85, 0.0, 50)

# Create the star family object
star_family_object = DeformedStarFamily(eos, p_center_space, p_surface)

# Solve the TOV equation and the tidal equation
star_family_object.solve_tidal(max_step=100.0)

# Plot the calculated and expected Mass vs Radius curves
star_family_object.plot_curve(
    x_axis="R", y_axis="M", figure_path=figures_path + "/comparison", expected_x=expected_radius, expected_y=expected_mass)

# Plot all curves
star_family_object.plot_all_curves(figures_path)
