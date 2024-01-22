import numpy as np
from constants import UnitConversion as uconv
from eos_library import SLy4EOS
from star_family_structure import StarFamily
from star_structure import Star


def main():
    """Main logic
    """

    # Set the path of the figures
    figures_path = "figures/app_sly4_eos"

    # Set the rho_space
    max_rho = 2.864e15 * uconv.MASS_DENSITY_CGS_TO_GU       # Maximum density [m^-2]
    rho_space = max_rho * np.logspace(-11.0, 0.0, 10000)

    # Create the EOS object
    eos = SLy4EOS(rho_space)

    # Set the central pressure of the star
    rho_center = max_rho                            # Central density [m^-2]
    p_center = eos.p(rho_center)                    # Central pressure [m^-2]

    # Single star

    # Define the object
    star_object = Star(eos, p_center)

    # Solve the TOV system and plot all curves
    star_object.solve_tov()
    star_object.plot_all_curves(figures_path)

    # Star Family

    # Set the p_center space that characterizes the star family
    p_center_space = p_center * np.logspace(-3.0, 0.0, 50)

    # Create the star family object
    star_family_object = StarFamily(eos, p_center_space)

    # Solve the TOV system and plot all curves
    star_family_object.solve_tov()
    star_family_object.plot_all_curves(figures_path)


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
