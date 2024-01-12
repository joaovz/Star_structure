import numpy as np
from constants import UnitConversion as uconv
from eos_library import PolytropicEOS
from star_family_structure import StarFamily
from star_structure import Star


# Set the path of the figures
figures_path = "figures/app_polytropic_eos"

# Create the EOS object
eos = PolytropicEOS(k=1.0e8, n=1)

# Set the pressure at the center of the star
rho_center = 5.691e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Central density [m^-2]
p_center = eos.p(rho_center)                                # Central pressure [m^-2]

# Single star

# Define the object
star_object = Star(eos, p_center)

# Solve the TOV equation
star_object.solve_tov(max_step=100.0)

# Plot the star structure curves
star_object.plot_star_structure_curves(figures_path)

# Star Family

# Set the p_center space that characterizes the star family
p_center_space = p_center * np.logspace(-5.0, 0.0, 50)

# Define the object
star_family_object = StarFamily(eos, p_center_space)

# Solve the TOV equation
star_family_object.solve_tov(max_step=100.0)

# Plot all curves
star_family_object.plot_all_curves(figures_path)
