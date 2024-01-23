import numpy as np
from constants import UnitConversion as uconv
from data_handling import csv_to_arrays
from eos_library import PolytropicEOS
from star_family_tides import DeformedStarFamily
from star_tides import DeformedStar


def main():
    """Main logic
    """

    # Constants
    FIGURES_PATH = "figures/app_polytropic_eos"                     # Path of the figures folder
    EXPECTED_K2_VS_C_FILE = "data/Polytropic_n_1_k2_vs_C.csv"       # File with the expected Love number vs Compactness curve
    MAX_RHO = 5.691e15 * uconv.MASS_DENSITY_CGS_TO_GU               # Maximum density [m^-2]
    STARS_LOGSPACE = np.logspace(-5.0, 0.0, 50)                     # Logspace used to create the star family

    # EOS parameters
    k = 1.0e8       # Proportional constant [dimensionless]
    n = 1           # Polytropic index [dimensionless]

    # Open the .csv file with the expected Love number vs Compactness curve
    (expected_C, expected_k2) = csv_to_arrays(EXPECTED_K2_VS_C_FILE)

    # Create the EOS object
    eos = PolytropicEOS(k, n)

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

    # Solve the combined TOV+tidal system
    star_family_object.solve_combined_tov_tidal()

    # Plot the calculated and expected Love number vs Compactness curves
    star_family_object.plot_curve(
        x_axis="C", y_axis="k2", figure_path=FIGURES_PATH + "/comparison", expected_x=expected_C, expected_y=expected_k2)

    # Plot all curves
    star_family_object.plot_all_curves(FIGURES_PATH)


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
