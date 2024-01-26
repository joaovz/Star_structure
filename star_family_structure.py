import os
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from constants import DefaultValues as dval
from constants import UnitConversion as uconv
from eos_library import PolytropicEOS
from star_structure import Star


class StarFamily:
    """Class with all the properties and methods necessary to describe a family of stars. Each star
    in the family is characterized by a specific value of central pressure (p_center)
    """

    # Class constants
    FIGURES_PATH = "figures/star_family"                # Path of the figures folder
    MAX_RHO = 1.0e16 * uconv.MASS_DENSITY_CGS_TO_GU     # Maximum density [m^-2]
    WIDE_LOGSPACE = np.logspace(-3.0, 0.0, 10)          # Wide logspace used in values search
    NARROW_LOGSPACE = np.logspace(-0.1, 0.1, 10)        # Narrow logspace used in values search

    def __init__(self, eos, p_center_space, p_surface=dval.P_SURFACE, r_init=dval.R_INIT, r_final=dval.R_FINAL,
                 method=dval.IVP_METHOD, max_step=dval.MAX_STEP, atol_tov=dval.ATOL_TOV, rtol=dval.RTOL):
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
            rtol (float, optional): Relative tolerance of the IVP solver. Defaults to RTOL
        """

        # Store the input parameters
        self.eos = eos
        self.p_center_space = p_center_space
        self.p_surface = p_surface
        self.r_init = r_init
        self.r_final = r_final
        self.method = method
        self.max_step = max_step
        self.atol_tov = atol_tov
        self.rtol = rtol

        # Create a star object with the first p_center value
        self.star_object = Star(eos, self.p_center_space[0], p_surface, r_init, r_final, method, max_step, atol_tov, rtol)

        # Calculate the rho_center_space
        self.rho_center_space = self.eos.rho(self.p_center_space)

        # Initialize star family properties
        self.radius_array = np.zeros(self.p_center_space.size)      # Array with the radii of the stars [m]
        self.mass_array = np.zeros(self.p_center_space.size)        # Array with the masses of the stars [m]
        self.maximum_mass = np.inf                                  # Maximum mass of the star family [m]
        self.maximum_stable_rho_center = self.MAX_RHO               # Maximum stable central density of the star family [m^-2]
        self.canonical_rho_center = self.MAX_RHO                    # Central density of the canonical star (M = 1.4 M_sun) [m^-2]
        self.canonical_radius = np.inf                              # Radius of the canonical star (M = 1.4 M_sun) [m]

    def _config_plot(self):
        """Method that configures the plotting
        """

        # Create a dictionary with all the functions used in plotting, with each name and label description
        self.plot_dict = {
            "rho_c": {
                "name": "Central density",
                "label": "$\\rho_{c} ~ [g \\cdot cm^{-3}]$",
                "value": self.rho_center_space * uconv.MASS_DENSITY_GU_TO_CGS,
            },
            "R": {
                "name": "Radius",
                "label": "$R ~ [km]$",
                "value": self.radius_array / 10**3,
            },
            "M": {
                "name": "Mass",
                "label": "$M ~ [M_{\\odot}]$",
                "value": self.mass_array * uconv.MASS_GU_TO_SOLAR_MASS,
            },
            "C": {
                "name": "Compactness",
                "label": "$C ~ [dimensionless]$",
                "value": self.mass_array / self.radius_array,
            },
        }

        # Create a list with all the curves to be plotted
        self.curves_list = [
            ["rho_c", "R"],
            ["rho_c", "M"],
            ["rho_c", "C"],
            ["R", "M"],
            ["C", "R"],
            ["C", "M"],
        ]

    def _calc_maximum_mass_star(self):
        """Method that calculates the maximum mass star properties
        """

        # Create the mass vs rho_center interpolated function and calculate its derivative, used in the stability criterion
        mass_rho_center_spline = CubicSpline(self.rho_center_space, self.mass_array, extrapolate=False)
        dm_drho_center_spline = mass_rho_center_spline.derivative()

        # Calculate the maximum stable rho_center and maximum mass
        dm_drho_center_roots = dm_drho_center_spline.roots()
        if dm_drho_center_roots.size > 0:
            self.maximum_stable_rho_center = dm_drho_center_roots[0]
            self.maximum_mass = mass_rho_center_spline(self.maximum_stable_rho_center)

        # Return the calculated rho_center
        return self.maximum_stable_rho_center

    def _calc_canonical_star(self):
        """Method that calculates the canonical star properties
        """

        # Create the (mass - canonical_mass) vs rho_center interpolated function
        mass_minus_canonical_array = self.mass_array - 1.4 * uconv.MASS_SOLAR_MASS_TO_GU
        mass_minus_canonical_rho_center_spline = CubicSpline(self.rho_center_space, mass_minus_canonical_array, extrapolate=False)

        # Create the radius vs rho_center interpolated function
        radius_rho_center_spline = CubicSpline(self.rho_center_space, self.radius_array, extrapolate=False)

        # Calculate the canonical radius and rho_center
        mass_minus_canonical_rho_center_roots = mass_minus_canonical_rho_center_spline.roots()
        if mass_minus_canonical_rho_center_roots.size > 0:
            self.canonical_rho_center = mass_minus_canonical_rho_center_roots[0]
            self.canonical_radius = radius_rho_center_spline(self.canonical_rho_center)

        # Return the calculated rho_center
        return self.canonical_rho_center

    def _find_star(self, finder_method, solver_method, initial_rho_center=MAX_RHO):
        """Method that finds a specific star in the family using the finder method

        Args:
            finder_method (method): Method used to find the star
            solver_method (method): Method used to solve the properties of the star family
            initial_rho_center (float, optional): Initial central density used by the finder. Defaults to MAX_RHO

        Raises:
            ValueError: Exception in case the initial radial coordinate is too large
            RuntimeError: Exception in case the IVP fails to solve the equation
            RuntimeError: Exception in case the IVP fails to find the ODE termination event
        """

        # Set the p_center space and rho_center space used to find the star
        p_center = self.eos.p(initial_rho_center)           # Central pressure [m^-2]
        self.p_center_space = p_center * self.WIDE_LOGSPACE
        self.rho_center_space = self.eos.rho(self.p_center_space)

        # Solve the system and find the star through finder_method
        solver_method(False)
        calculated_rho_center = finder_method()

        # Redefine the p_center space to a stricter interval
        p_center = self.eos.p(calculated_rho_center)        # Central pressure [m^-2]
        self.p_center_space = p_center * self.NARROW_LOGSPACE
        self.rho_center_space = self.eos.rho(self.p_center_space)

        # Solve the system and find the star through finder_method
        solver_method(False)
        finder_method()

    def find_maximum_mass_star(self):
        """Method that finds the maximum mass star

        Raises:
            ValueError: Exception in case the initial radial coordinate is too large
            RuntimeError: Exception in case the IVP fails to solve the equation
            RuntimeError: Exception in case the IVP fails to find the ODE termination event
        """

        self._find_star(self._calc_maximum_mass_star, self.solve_tov, self.MAX_RHO)

    def find_canonical_star(self):
        """Method that finds the canonical star

        Raises:
            ValueError: Exception in case the initial radial coordinate is too large
            RuntimeError: Exception in case the IVP fails to solve the equation
            RuntimeError: Exception in case the IVP fails to find the ODE termination event
        """

        self._find_star(self._calc_canonical_star, self.solve_tov, self.maximum_stable_rho_center)

    def solve_tov(self, show_results=True):
        """Method that solves the TOV system, finding the radius and mass of each star in the family

        Args:
            show_results (bool, optional): Flag that enables the results printing after the solve. Defaults to True

        Raises:
            ValueError: Exception in case the initial radial coordinate is too large
            RuntimeError: Exception in case the IVP fails to solve the equation
            RuntimeError: Exception in case the IVP fails to find the ODE termination event
        """

        # Reinitialize the radius and mass arrays with the right size
        self.radius_array = np.zeros(self.p_center_space.size)
        self.mass_array = np.zeros(self.p_center_space.size)

        # Solve the TOV system for each star in the family
        start_time = perf_counter()
        for k, p_center in enumerate(self.p_center_space):
            self.star_object.solve_tov(p_center, False)
            self.radius_array[k] = self.star_object.star_radius
            self.mass_array[k] = self.star_object.star_mass
        self.execution_time = perf_counter() - start_time

        # Configure the plot
        self._config_plot()

        # Show results if requested
        if show_results is True:
            self.print_results()

    def print_results(self):
        """Method that prints the results found
        """

        # Calculate the star family properties
        self._calc_maximum_mass_star()
        self._calc_canonical_star()

        # Print the results
        print(f"\n##########    Star family solve results    ##########")
        print(f"Executed the solution in: {(self.execution_time):.3f} [s]")
        print(f"Maximum mass (M_max) = {(self.maximum_mass * uconv.MASS_GU_TO_SOLAR_MASS):e} [solar mass]")
        print(f"Maximum stable central density (rho_center_max) = {(self.maximum_stable_rho_center * uconv.MASS_DENSITY_GU_TO_CGS):e} [g ⋅ cm^-3]")
        print(f"Radius of the canonical star (R_canonical) = {(self.canonical_radius / 10**3):e} [km]")
        print(f"Central density of the canonical star (rho_center_canonical) = {(self.canonical_rho_center * uconv.MASS_DENSITY_GU_TO_CGS):e} [g ⋅ cm^-3]")

    def plot_curve(self, x_axis="R", y_axis="M", figure_path=FIGURES_PATH, expected_x=None, expected_y=None):
        """Method that plots some curve of the star family

        Args:
            x_axis (str, optional): Key of self.plot_dict to indicate the x_axis used. Defaults to "R"
            y_axis (str, optional): Key of self.plot_dict to indicate the y_axis used. Defaults to "M"
            figure_path (str, optional): Path used to save the figure. Defaults to FIGURES_PATH
            expected_x (array of float, optional): Array with the x values of the expected curve. Defaults to None
            expected_y (array of float, optional): Array with the y values of the expected curve. Defaults to None
        """

        # Create a simple plot
        plt.figure()
        plt.plot(self.plot_dict[x_axis]["value"], self.plot_dict[y_axis]["value"], linewidth=1, label="Calculated curve", marker=".")
        eos_name = self.eos.eos_name.replace("EOS", " EOS")
        plt.title(f"{self.plot_dict[y_axis]["name"]} vs {self.plot_dict[x_axis]["name"]} curve of the {eos_name} star family", y=1.05)
        plt.xlabel(self.plot_dict[x_axis]["label"])
        plt.ylabel(self.plot_dict[y_axis]["label"])

        # If expected curve is received, add a second comparison curve, and enable legend
        if (expected_x is not None) and (expected_y is not None):
            plt.plot(expected_x, expected_y, linewidth=1, label="Expected curve")
            plt.legend()

        # Create the folder if necessary and save the figure
        os.makedirs(figure_path, exist_ok=True)
        x_axis_name = self.plot_dict[x_axis]["name"].lower().replace(" ", "_")
        y_axis_name = self.plot_dict[y_axis]["name"].lower().replace(" ", "_")
        figure_name = f"{y_axis_name}_vs_{x_axis_name}_curve.svg"
        complete_path = os.path.join(figure_path, figure_name)
        plt.savefig(complete_path)

    def plot_all_curves(self, figures_path=FIGURES_PATH):
        """Method that plots all curves specified by the self.curves_list

        Args:
            figures_path (str, optional): Path used to save the figures. Defaults to FIGURES_PATH
        """

        for axis in self.curves_list:
            self.plot_curve(axis[0], axis[1], figures_path)
        plt.show()


def main():
    """Main logic
    """

    # Constants
    MAX_RHO = 5.691e15 * uconv.MASS_DENSITY_CGS_TO_GU       # Maximum density [m^-2]
    STARS_LOGSPACE = np.logspace(-5.0, 0.0, 50)             # Logspace used to create the star family

    # EOS parameters
    k = 1.0e8       # Proportional constant [dimensionless]
    n = 1           # Polytropic index [dimensionless]

    # Create the EOS object
    eos = PolytropicEOS(k, n)

    # Set the central pressure of the star and p_center space of the star family
    rho_center = MAX_RHO                # Central density [m^-2]
    p_center = eos.p(rho_center)        # Central pressure [m^-2]
    p_center_space = p_center * STARS_LOGSPACE

    # Create the star family object
    star_family_object = StarFamily(eos, p_center_space)

    # Solve the TOV system and plot all curves
    star_family_object.solve_tov()
    star_family_object.plot_all_curves()


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
