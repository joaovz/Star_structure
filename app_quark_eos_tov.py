import numpy as np
from constants import UnitConversion as uconv
from eos_library import QuarkEOS
from star_family_structure import StarFamily
from star_structure import Star


def main():
    """Main logic
    """

    # Constants
    FIGURES_PATH = "figures/app_quark_eos"                      # Path of the figures folder
    STARS_MAX_RHO = 1.70e15 * uconv.MASS_DENSITY_CGS_TO_GU      # Maximum density used to create the star family [m^-2]
    STARS_LOGSPACE = np.logspace(-4.0, 0.0, 50)                 # Logspace used to create the star family

    # EOS parameters (values chosen to build a strange star)
    a2 = 14.61e3        # Model free parameter [MeV^2]
    a4 = 0.64           # Model free parameter [dimensionless]
    B = 130.76**4       # Model free parameter [MeV^4]

    # Create the EOS object
    eos = QuarkEOS(a2, a4, B)

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
