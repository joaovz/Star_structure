import numpy as np
from constants import UnitConversion as uconv
from eos_library import PolytropicEOS
from star_family_structure import StarFamily
from star_structure import Star


def main():
    """Main logic
    """

    # Constants
    FIGURES_PATH = "figures/app_polytropic_eos"                 # Path of the figures folder
    STARS_MAX_RHO = 5.80e15 * uconv.MASS_DENSITY_CGS_TO_GU      # Maximum density used to create the star family [m^-2]
    STARS_LOGSPACE = np.logspace(-5.0, 0.0, 50)                 # Logspace used to create the star family

    # EOS parameters
    k = 1.0e8       # Proportional constant [dimensionless]
    n = 1           # Polytropic index [dimensionless]

    # Create the EOS object
    eos = PolytropicEOS(k, n)

    # Set the central pressure of the star and p_center space of the star family
    rho_center = STARS_MAX_RHO          # Central density [m^-2]
    p_center = eos.p(rho_center)        # Central pressure [m^-2]
    p_center_space = p_center * STARS_LOGSPACE

    # Single star

    # Create the star object
    star_object = Star(eos, p_center)

    # Solve the TOV system and plot all curves
    star_object.solve_tov()
    star_object.plot_all_curves(FIGURES_PATH)

    # Star Family

    # Create the star family object
    star_family_object = StarFamily(eos, p_center_space)

    # Solve the TOV system and plot all curves
    star_family_object.solve_tov()
    star_family_object.plot_all_curves(FIGURES_PATH)


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
