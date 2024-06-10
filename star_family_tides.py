from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from constants import Constants as const
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

    def __init__(self, eos, p_center_space, p_surface=dval.P_SURFACE, r_init=dval.R_INIT, r_final=dval.R_FINAL, method=dval.IVP_METHOD,
                 max_step=dval.MAX_STEP, atol_tov=dval.ATOL_TOV, atol_tidal=dval.ATOL_TIDAL, rtol=dval.RTOL):
        """Initialization method

        Args:
            eos (object): Python object with methods rho, p, drho_dp, and dp_drho that describes the EOS of the stars
            p_center_space (array of float): Array with the central pressure of each star in the family [m^-2]
            p_surface (float, optional): Surface pressure of the stars [m^-2]. Defaults to P_SURFACE
            r_init (float, optional): Initial radial coordinate r of the IVP solve [m]. Defaults to R_INIT
            r_final (float, optional): Final radial coordinate r of the IVP solve [m]. Defaults to R_FINAL
            method (str, optional): Method used by the IVP solver. Defaults to IVP_METHOD
            max_step (float, optional): Maximum allowed step size for the IVP solver [m]. Defaults to MAX_STEP
            atol_tov (float or array of float, optional): Absolute tolerance of the IVP solver for the TOV system. Defaults to ATOL_TOV
            atol_tidal (float, optional): Absolute tolerance of the IVP solver for the tidal system. Defaults to ATOL_TIDAL
            rtol (float, optional): Relative tolerance of the IVP solver. Defaults to RTOL
        """

        # Execute parent class' __init__ method
        super().__init__(eos, p_center_space, p_surface, r_init, r_final, method, max_step, atol_tov, rtol)

        # Store the input parameters
        self.atol_tidal = atol_tidal

        # Create a star object with the first p_center value, using instead the DeformedStar class
        self.star_object = DeformedStar(eos, self.p_center_space[0], p_surface, r_init, r_final, method, max_step, atol_tov, atol_tidal, rtol)

        # Initialize deformed star family properties
        self.k2_array = np.zeros(self.p_center_space.size)              # Array with the tidal Love numbers of the stars [dimensionless]
        self.lambda_array = np.zeros(self.p_center_space.size)          # Array with the tidal deformabilities of the stars [dimensionless]
        self.canonical_lambda = np.inf                                  # Tidal deformability of the canonical star (M = 1.4 M_sun) [dimensionless]
        self.maximum_k2 = np.inf                                        # Maximum k2 of the star family [dimensionless]
        self.maximum_k2_star_rho_center = self.MAX_RHO                  # Central density of the star with the maximum k2 [m^-2]
        self.maximum_k2_star_p_center = self.eos.p(self.MAX_RHO)        # Central pressure of the star with the maximum k2 [m^-2]

    def _config_tidal_plot(self):
        """Method that configures the plotting of the tidal related curves
        """

        # Add new functions to the plot dictionary
        self.plot_dict["k2"] = {
            "name": "Love number",
            "label": "$k_2 ~ [dimensionless]$",
            "value": self.k2_array,
        }
        self.plot_dict["Lambda"] = {
            "name": "Tidal deformability",
            "label": "$\\log_{10} \\left( \\Lambda ~ [dimensionless] \\right)$",
            "value": np.log10(self.lambda_array),
        }

        # Add new curves to be plotted on the list
        self.curves_list += [
            ["rho_c", "k2"],
            ["R", "k2"],
            ["M", "k2"],
            ["C", "k2"],
            ["rho_c", "Lambda"],
            ["R", "Lambda"],
            ["M", "Lambda"],
            ["C", "Lambda"],
        ]

    def _calc_maximum_k2_star(self, solve_first=False):
        """Method that calculates the maximum k2 star properties

        Args:
            solve_first (bool, optional): Flag that enables the solve in the beginning of the logic. Defaults to False
        """

        # Solve first if requested
        if solve_first is True:
            self.solve_combined_tov_tidal(False)

        # Calculate the maximum k2 star p_center, rho_center, and k2 using the array directly
        k2_max_index = np.argmax(self.k2_array)
        self.maximum_k2_star_p_center = self.p_center_space[k2_max_index]
        self.maximum_k2_star_rho_center = self.rho_center_space[k2_max_index]
        self.maximum_k2 = self.k2_array[k2_max_index]

        # Create the k2 vs p_center interpolated function and calculate its derivative
        k2_p_center_spline = CubicSpline(self.p_center_space, self.k2_array, extrapolate=False)
        dk2_dp_center_spline = k2_p_center_spline.derivative()

        # Calculate the maximum k2 star p_center, rho_center, and k2
        dk2_dp_center_roots = dk2_dp_center_spline.roots()
        for dk2_dp_center_root in dk2_dp_center_roots:
            possible_maximum_k2 = k2_p_center_spline(dk2_dp_center_root)
            if possible_maximum_k2 > self.maximum_k2:
                self.maximum_k2_star_p_center = dk2_dp_center_root
                self.maximum_k2_star_rho_center = self.eos.rho(self.maximum_k2_star_p_center)
                self.maximum_k2 = k2_p_center_spline(self.maximum_k2_star_p_center)

        # Debug graph
        if const.DEBUG is True:
            self._config_plot()
            self._config_tidal_plot()
            self.plot_curve(x_axis="rho_c", y_axis="k2")
            plt.plot(self.maximum_k2_star_rho_center * uconv.MASS_DENSITY_GU_TO_CGS, self.maximum_k2, linewidth=1, marker=".", markersize=4**2, label="${k_2}_{max}$")
            plt.legend()
            plt.show()

        # Return the calculated rho_center
        return self.maximum_k2_star_rho_center

    def _calc_canonical_lambda(self):
        """Method that calculates the tidal deformability of the canonical star (M = 1.4 M_sun)
        """

        # Calculate canonical_lambda only if canonical_rho_center was found
        if self.canonical_rho_center < self.MAX_RHO:

            # Solve the combined TOV+tidal system for the canonical star and get the canonical tidal deformability
            self.star_object.solve_combined_tov_tidal(self.eos.p(self.canonical_rho_center), False)
            self.canonical_lambda = self.star_object.lambda_tidal

    def find_canonical_star(self):

        # Execute parent class' find_canonical_star method
        super().find_canonical_star()

        # Calculate canonical_lambda
        self._calc_canonical_lambda()

    def find_maximum_k2_star(self):
        """Method that finds the maximum k2 star

        Raises:
            ValueError: Exception in case the initial radial coordinate is too large
            RuntimeError: Exception in case the IVP fails to solve the equation
            RuntimeError: Exception in case the IVP fails to find the ODE termination event
        """

        self._find_star(self._calc_maximum_k2_star, self.maximum_stable_rho_center)

    def solve_combined_tov_tidal(self, show_results=True):
        """Method that solves the combined TOV+tidal system for each star in the family, finding p, m, nu, and y

        Args:
            show_results (bool, optional): Flag that enables the results printing after the solve. Defaults to True

        Raises:
            ValueError: Exception in case the initial radial coordinate is too large
            RuntimeError: Exception in case the IVP fails to solve the equation
            RuntimeError: Exception in case the IVP fails to find the ODE termination event
        """

        # Reinitialize the arrays with the right size
        self.radius_array = np.zeros(self.p_center_space.size)
        self.mass_array = np.zeros(self.p_center_space.size)
        self.phase_trans_radius_array = np.zeros(self.p_center_space.size)
        self.phase_trans_mass_array = np.zeros(self.p_center_space.size)
        self.k2_array = np.zeros(self.p_center_space.size)
        self.lambda_array = np.zeros(self.p_center_space.size)

        # Solve the combined TOV+tidal system for each star in the family
        start_time = perf_counter()
        for k, p_center in enumerate(self.p_center_space):
            self.star_object.solve_combined_tov_tidal(p_center, False)
            self.radius_array[k] = self.star_object.star_radius
            self.mass_array[k] = self.star_object.star_mass
            self.phase_trans_radius_array[k] = self.star_object.star_phase_trans_radius
            self.phase_trans_mass_array[k] = self.star_object.star_phase_trans_mass
            self.k2_array[k] = self.star_object.k2
            self.lambda_array[k] = self.star_object.lambda_tidal
        self.execution_time = perf_counter() - start_time

        # Configure the plot
        self._config_plot()
        self._config_tidal_plot()

        # Show results if requested
        if show_results is True:
            self.print_results()

    def print_results(self):
        """Method that prints the results found
        """

        # Execute parent class' print_results method
        super().print_results()

        # Calculate the star family properties
        self._calc_maximum_k2_star()
        self._calc_canonical_lambda()

        # Print the results
        print(f"Tidal deformability of the canonical star (Lambda_canonical) = {(self.canonical_lambda):e} [dimensionless]")
        print(f"Maximum k2 star central density (rho_center_k2_max) = {(self.maximum_k2_star_rho_center * uconv.MASS_DENSITY_GU_TO_CGS):e} [g cm^-3]")
        print(f"Maximum k2 (k2_max) = {(self.maximum_k2):e} [dimensionless]")


def main():
    """Main logic
    """

    # Constants
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

    # Create the star family object
    star_family_object = DeformedStarFamily(eos, p_center_space)

    # Solve the combined TOV+tidal system and plot all curves
    star_family_object.solve_combined_tov_tidal()
    star_family_object.plot_all_curves()


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
