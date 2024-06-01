import numpy as np
from constants import UnitConversion as uconv
from eos_library import QuarkEOS, SLy4EOS, HybridEOS
from star_family_tides import DeformedStarFamily
from star_tides import DeformedStar


def main():
    """Main logic
    """

    # Constants
    FIGURES_PATH = "figures/app_hybrid_eos"                     # Path of the figures folder
    EOS_MAX_RHO = 3.00e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Maximum density used to create the EOS [m^-2]
    STARS_MAX_RHO = 2.92e15 * uconv.MASS_DENSITY_CGS_TO_GU      # Maximum density used to create the star family [m^-2]
    EOS_LOGSPACE = np.logspace(-11.0, 0.0, 10000)               # Logspace used to create the EOS
    STARS_LOGSPACE = np.logspace(-2.0, 0.0, 50)                 # Logspace used to create the star family

    # Create the QuarkEOS object (values chosen to build a hybrid star)
    a2 = 100**2     # Model free parameter [MeV^2]
    a4 = 0.8        # Model free parameter [dimensionless]
    B = 160**4      # Model free parameter [MeV^4]
    quark_eos = QuarkEOS(a2, a4, B)

    # Create the EOS object
    sly4_eos_rho_space = EOS_MAX_RHO * EOS_LOGSPACE
    sly4_eos = SLy4EOS(sly4_eos_rho_space)

    # Create the HybridEOS object
    sly4_maximum_stable_rho_center = 2.865e15 * uconv.MASS_DENSITY_CGS_TO_GU
    hybrid_eos = HybridEOS(quark_eos, sly4_eos, "data/SLy4_EOS.csv", sly4_maximum_stable_rho_center)

    # Set the central pressure of the star and p_center space of the star family
    rho_center = STARS_MAX_RHO              # Central density [m^-2]
    p_center = hybrid_eos.p(rho_center)     # Central pressure [m^-2]
    p_center_space = p_center * STARS_LOGSPACE

    # Single star

    # Create the star object
    star_object = DeformedStar(hybrid_eos, p_center)

    # Solve the combined TOV+tidal system and plot all curves
    star_object.solve_combined_tov_tidal()
    star_object.plot_all_curves(FIGURES_PATH)

    # Star Family

    # Create the star family object
    star_family_object = DeformedStarFamily(hybrid_eos, p_center_space)

    # Solve the combined TOV+tidal system and plot all curves
    star_family_object.solve_combined_tov_tidal()
    star_family_object.plot_all_curves(FIGURES_PATH)


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
