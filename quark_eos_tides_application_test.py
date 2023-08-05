import numpy as np
from perfect_fluid_star_family_tides import DeformedStarFamily
from data_handling import *
from eos_library import QuarkEOS


# Create the EOS object (values chosen to build a strange star)
B = 130 * MeV_fm3_to_SI * PRESSURE_SI_TO_GU
a2 = (100 * MeV_fm3_to_SI * ENERGY_DENSITY_SI_TO_GU)**(1/2)
a4 = 0.6
eos = QuarkEOS(B, a2, a4)

# Print the values used for B, a2, and a4
print(f"B = {B} [m^-2]")
print(f"a2 = {a2} [m^-1]")
print(f"a4 = {a4} [dimensionless]")

# Set the pressure at the center and surface of the star
rho_center = 2.376364e-9        # Center density [m^-2]
p_center = eos.p(rho_center)    # Center pressure [m^-2]
p_surface = 0.0                 # Surface pressure [m^-2]

# Print the values used for p_center and p_surface
print(f"p_center = {p_center} [m^-2]")
print(f"p_surface = {p_surface} [m^-2]")

# Set the p_center space that characterizes the star family
p_center_space = p_center * np.logspace(-4.0, 1.0, 50)

# Define the object
star_family_object = DeformedStarFamily(eos.rho, p_center_space, p_surface)

# Solve the TOV equation, and the tidal equation
star_family_object.solve_tidal(max_step=1.0)

# Show the radius-mass curve
star_family_object.plot_radius_mass_curve()

# Show the k2 curve
star_family_object.plot_k2_curve()