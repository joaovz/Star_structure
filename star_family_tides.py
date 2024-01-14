from alive_progress import alive_bar
import numpy as np
from constants import DefaultValues as dval
from constants import UnitConversion as uconv
from eos_library import PolytropicEOS
from star_family_structure import StarFamily
from star_tides import DeformedStar


class DeformedStarFamily(StarFamily):
    """Class with all the properties and methods necessary to describe a family of deformed stars

    Args:
        StarFamily (class): Parent class with all the properties and methods necessary to describe a family of stars
        Each star in the family is characterized by a specific value of central pressure (p_center)
    """

    def __init__(self, eos, p_center_space, p_surface=dval.P_SURFACE):

        # Execute parent class' __init__ method
        super().__init__(eos, p_center_space, p_surface)

        # Create a star object with the first p_center value, using instead the DeformedStar class
        self.star_object = DeformedStar(eos, self.p_center_space[0], p_surface)

        # Create the k2 array to store this star family property
        self.k2_array = np.zeros(self.p_center_space.size)

    def _config_plot(self):

        # Execute parent class' _config_plot method
        super()._config_plot()

        # Add new functions to the plot dictionary
        self.plot_dict["k2"] = {
            "name": "Love number",
            "label": "$k2 ~ [dimensionless]$",
            "value": self.k2_array,
        }

        # Add new curves to be plotted on the list
        extra_curves_list = [
            ["p_c", "k2"],
            ["rho_c", "k2"],
            ["R", "k2"],
            ["M", "k2"],
            ["C", "k2"],
        ]
        self.curves_list += extra_curves_list

    def solve_tidal(self, r_init=dval.R_INIT, r_final=dval.R_FINAL, method=dval.IVP_METHOD, max_step=dval.MAX_STEP,
                    atol_tov=dval.ATOL_TOV, atol_tidal=dval.ATOL_TIDAL, rtol=dval.RTOL):
        """Method that solves the tidal system for each star in the family, finding the tidal Love number k2

        Args:
            r_init (float, optional): Initial radial coordinate r of the IVP solve. Defaults to R_INIT
            r_final (float, optional): Final radial coordinate r of the IVP solve. Defaults to R_FINAL
            method (str, optional): Method used by the IVP solver. Defaults to IVP_METHOD
            max_step (float, optional): Maximum allowed step size for the IVP solver. Defaults to MAX_STEP
            atol_tov (float or array of float, optional): Absolute tolerance of the IVP solver for the TOV equation. Defaults to ATOL_TOV
            atol_tidal (float or array of float, optional): Absolute tolerance of the IVP solver for the tidal equation. Defaults to ATOL_TIDAL
            rtol (float, optional): Relative tolerance of the IVP solver. Defaults to RTOL
        """

        # Solve the TOV equation and the tidal equation for each star in the family
        with alive_bar(self.p_center_space.size) as bar:
            for k in range(self.p_center_space.size):
                self.star_object.solve_tov(self.p_center_space[k], r_init, r_final, method, max_step, atol_tov, rtol)
                self.star_object.solve_tidal(r_init, method, max_step, atol_tidal, rtol)
                self.radius_array[k] = self.star_object.star_radius
                self.mass_array[k] = self.star_object.star_mass
                self.k2_array[k] = self.star_object.k2
                bar()

        # Execute the stability criterion check and configure the plot
        self._check_stability()
        self._config_plot()


def main():
    """Main logic
    """

    # Create the EOS object
    eos = PolytropicEOS(k=1.0e8, n=1)

    # Set the pressure at the center and surface of the star
    rho_center = 5.691e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Central density [m^-2]
    p_center = eos.p(rho_center)                                # Central pressure [m^-2]

    # Set the p_center space that characterizes the star family
    p_center_space = p_center * np.logspace(-5.0, 0.0, 50)

    # Define the object
    star_family_object = DeformedStarFamily(eos, p_center_space)

    # Solve the tidal equation
    star_family_object.solve_tidal()

    # Plot all curves
    star_family_object.plot_all_curves()


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
