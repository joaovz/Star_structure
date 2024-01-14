import numpy as np
from constants import UnitConversion as uconv
from eos_library import TableEOS
from star_family_structure import StarFamily
from star_structure import Star


def main():
    """Main logic
    """

    # Set the path of the figures
    figures_path = "figures/app_table_sly4_eos"

    # Create the EOS object
    eos = TableEOS(fname='data/SLy4.csv', eos_name='SLy4EOS')

    # Set the pressure at the center of the star
    rho_center = 2.864e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Central density [m^-2]
    p_center = eos.p(rho_center)                                # Central pressure [m^-2]

    # Single star

    # Define the object
    star_object = Star(eos, p_center)

    # Solve the TOV equation
    star_object.solve_tov()

    # Plot the star structure curves
    star_object.plot_star_structure_curves(figures_path)

    # Star Family

    # Set the p_center space that characterizes the star family
    p_center_space = p_center * np.logspace(-3.0, 0.0, 50)

    # Create the star family object
    star_family_object = StarFamily(eos, p_center_space)

    # Solve the TOV equation
    star_family_object.solve_tov()

    # Plot all curves
    star_family_object.plot_all_curves(figures_path)


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
