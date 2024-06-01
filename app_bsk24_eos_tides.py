import numpy as np
from constants import UnitConversion as uconv
from data_handling import csv_to_arrays
from eos_library import BSk24EOS
from star_family_tides import DeformedStarFamily
from star_tides import DeformedStar


def main():
    """Main logic
    """

    # Constants
    FIGURES_PATH = "figures/app_bsk24_eos"                          # Path of the figures folder
    EXPECTED_M_VS_R_FILE = "data/BSk24_M_vs_R.csv"                  # File with the expected Mass vs Radius curve
    EXPECTED_LAMBDA_VS_C_FILE = "data/BSk24_Lambda_vs_C.csv"        # File with the expected Lambda vs Compactness curve
    EOS_MAX_RHO = 2.30e15 * uconv.MASS_DENSITY_CGS_TO_GU            # Maximum density used to create the EOS [m^-2]
    STARS_MAX_RHO = 2.29e15 * uconv.MASS_DENSITY_CGS_TO_GU          # Maximum density used to create the star family [m^-2]
    EOS_LOGSPACE = np.logspace(-11.0, 0.0, 10000)                   # Logspace used to create the EOS
    STARS_LOGSPACE = np.logspace(-2.3, 0.0, 50)                     # Logspace used to create the star family

    # Open the .csv file with the expected Mass vs Radius curve (units in solar mass and km)
    (expected_radius, expected_mass) = csv_to_arrays(EXPECTED_M_VS_R_FILE)

    # Open the .csv file with the expected Lambda vs Compactness curve
    (expected_C, expected_Lambda) = csv_to_arrays(EXPECTED_LAMBDA_VS_C_FILE)
    expected_k2 = (3 / 2) * expected_C**5 * expected_Lambda

    # Create the EOS object
    eos_rho_space = EOS_MAX_RHO * EOS_LOGSPACE
    eos = BSk24EOS(eos_rho_space)

    # Set the central pressure of the star and p_center space of the star family
    rho_center = STARS_MAX_RHO          # Central density [m^-2]
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

    # Solve the combined TOV+tidal system
    star_family_object.solve_combined_tov_tidal()

    # Plot the calculated and expected Mass vs Radius curves
    star_family_object.plot_curve(
        x_axis="R", y_axis="M", figure_path=FIGURES_PATH + "/comparison", expected_x=expected_radius, expected_y=expected_mass)

    # Plot the calculated and expected Love number vs Compactness curves
    star_family_object.plot_curve(
        x_axis="C", y_axis="k2", figure_path=FIGURES_PATH + "/comparison", expected_x=expected_C, expected_y=expected_k2)

    # Plot all curves
    star_family_object.plot_all_curves(FIGURES_PATH)


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
