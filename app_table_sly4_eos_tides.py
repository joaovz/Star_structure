import numpy as np
from data_handling import *
from star_family_tides import DeformedStarFamily
from eos_library import TableEOS


# Set the path of the figures
figures_path = "figures/app_table_sly4_eos"

# Create the EOS object
eos = TableEOS(fname='data/SLy4.csv', eos_name='SLy4EOS')

# Set the pressure at the center and surface of the star
rho_center = 2.864e15 * MASS_DENSITY_CGS_TO_GU      # Central density [m^-2]
p_center = eos.p(rho_center)                        # Central pressure [m^-2]
p_surface = 1e23 * PRESSURE_CGS_TO_GU               # Surface pressure [m^-2]

# Set the p_center space that characterizes the star family
p_center_space = p_center * np.logspace(-3.0, 0.0, 50)

# Create the star family object
star_family_object = DeformedStarFamily(eos, p_center_space, p_surface)

# Solve the TOV equation and the tidal equation
star_family_object.solve_tidal(max_step=100.0)

# Plot all curves
star_family_object.plot_all_curves(figures_path)
