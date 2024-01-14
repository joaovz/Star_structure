import numpy as np
from constants import UnitConversion as uconv
from data_handling import csv_to_arrays
from eos_library import TableEOS
from star_family_structure import StarFamily
from star_structure import Star


def main():
    """Main logic
    """

    # Set the path of the figures
    figures_path = "figures/app_table_gm1_eos"

    # Open the .csv file with the expected Mass vs Radius curve (units in solar mass and km)
    (expected_radius, expected_mass) = csv_to_arrays(
        fname='data/GM1_M_vs_R.csv')

    # Create the EOS object
    eos = TableEOS(fname='data/GM1.csv', eos_name='GM1EOS')

    # Set the pressure at the center and surface of the star
    rho_center = 1.977e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Central density [m^-2]
    p_center = eos.p(rho_center)                                # Central pressure [m^-2]
    p_surface = 1e23 * uconv.PRESSURE_CGS_TO_GU                 # Surface pressure [m^-2]

    # Single star

    # Define the object
    star_object = Star(eos, p_center, p_surface)

    # Solve the TOV equation
    star_object.solve_tov()

    # Plot the star structure curves
    star_object.plot_star_structure_curves(figures_path)

    # Star Family

    # Set the p_center space that characterizes the star family
    p_center_space = p_center * np.logspace(-2.85, 0.0, 50)

    # Create the star family object
    star_family_object = StarFamily(eos, p_center_space, p_surface)

    # Solve the TOV equation
    star_family_object.solve_tov()

    # Plot the calculated and expected Mass vs Radius curves
    star_family_object.plot_curve(
        x_axis="R", y_axis="M", figure_path=figures_path + "/comparison", expected_x=expected_radius, expected_y=expected_mass)

    # Plot all curves
    star_family_object.plot_all_curves(figures_path)


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
