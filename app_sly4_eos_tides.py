import numpy as np
from constants import UnitConversion as uconv
from eos_library import SLy4EOS
from star_family_tides import DeformedStarFamily
from star_tides import DeformedStar


def main():
    """Main logic
    """

    # Constants
    FIGURES_PATH = "figures/app_sly4_eos"                   # Path of the figures folder
    MAX_RHO = 2.864e15 * uconv.MASS_DENSITY_CGS_TO_GU       # Maximum density [m^-2]
    EOS_LOGSPACE = np.logspace(-11.0, 0.0, 10000)           # Logspace used to create the EOS
    STARS_LOGSPACE = np.logspace(-2.4, 0.0, 50)             # Logspace used to create the star family

    # Create the EOS object
    eos_rho_space = MAX_RHO * EOS_LOGSPACE
    eos = SLy4EOS(eos_rho_space)

    # Set the central pressure of the star and p_center space of the star family
    rho_center = MAX_RHO                # Central density [m^-2]
    p_center = eos.p(rho_center)        # Central pressure [m^-2]
    p_center_space = p_center * STARS_LOGSPACE

    # Single star

    # Create the star object
    star_object = DeformedStar(eos, p_center)

    # Solve the combined TOV+tidal system and plot all curves
    star_object.solve_combined_tov_tidal()
    star_object.plot_all_curves(FIGURES_PATH)

    # Star Family

    # Create the star family object
    star_family_object = DeformedStarFamily(eos, p_center_space)

    # Solve the combined TOV+tidal system and plot all curves
    star_family_object.solve_combined_tov_tidal()
    star_family_object.plot_all_curves(FIGURES_PATH)


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
