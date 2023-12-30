import os
from alive_progress import alive_bar
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
    FIGURES_PATH = "figures/star_family"

    def __init__(self, eos, p_center_space, p_surface):
        """Initialization method

        Args:
            eos (object): Python object with methods rho, p, drho_dp, and dp_drho that describes the EOS of the stars
            p_center_space (array of float): Array with the central pressure of each star in the family [m^-2]
            p_surface (float): Surface pressure of the stars [m^-2]
        """

        # Store the input parameters
        self.p_center_space = p_center_space

        # Create a star object with the first p_center value
        self.star_object = Star(eos, self.p_center_space[0], p_surface)

        # Calculate the rho_center_space
        self.rho_center_space = self.star_object.eos.rho(self.p_center_space)

        # Initialize star family properties
        self.radius_array = np.zeros(self.p_center_space.size)
        self.mass_array = np.zeros(self.p_center_space.size)
        self.maximum_mass = None
        self.maximum_stable_rho_center = None

    def _check_stability(self):
        """Method that checks the stability criterion for the star family
        """

        # Create the mass vs rho_center interpolated function and calculate its derivative, used in the stability criterion
        self.mass_rho_center_spline = CubicSpline(self.rho_center_space, self.mass_array, extrapolate=False)
        self.dm_drho_center_spline = self.mass_rho_center_spline.derivative()
        self.dm_drho_center = self.dm_drho_center_spline(self.rho_center_space)

        # Calculate the maximum stable rho_center and maximum mass
        dm_drho_center_roots = self.dm_drho_center_spline.roots()
        if dm_drho_center_roots.size > 0:
            self.maximum_stable_rho_center = dm_drho_center_roots[0]
            self.maximum_mass = self.mass_rho_center_spline(self.maximum_stable_rho_center)
            print(f"Maximum stable rho_center = {(self.maximum_stable_rho_center * uconv.MASS_DENSITY_GU_TO_CGS):e} [g ⋅ cm^-3]")
            print(f"Maximum mass = {(self.maximum_mass * uconv.MASS_GU_TO_SOLAR_MASS):e} [solar mass]")
        else:
            print("Maximum stable rho_center not reached")

    def _config_plot(self):
        """Method that configures the plotting
        """

        # Create a dictionary with all the functions used in plotting, with each name and label description
        self.plot_dict = {
            "p_c": {
                "name": "Central pressure",
                "label": "$p_{c} ~ [dyn \\cdot cm^{-2}]$",
                "value": self.p_center_space * uconv.PRESSURE_GU_TO_CGS,
            },
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
            "dM_drho_c": {
                "name": "Mass derivative",
                "label": "$\\dfrac{\\partial M}{\\partial \\rho_{c}} ~ [m^3]$",
                "value": self.dm_drho_center,
            },
        }

        # Create a list with all the curves to be plotted
        self.curves_list = [
            ['p_c', 'R'],
            ['p_c', 'M'],
            ['p_c', 'C'],
            ['rho_c', 'R'],
            ['rho_c', 'M'],
            ['rho_c', 'C'],
            ['rho_c', 'dM_drho_c'],
            ['R', 'M'],
            ['C', 'R'],
            ['C', 'M'],
        ]

    def find_maximum_mass(self, r_init=dval.R_INIT, r_final=dval.R_FINAL, method=dval.IVP_METHOD, max_step=dval.MAX_STEP, atol=dval.ATOL, rtol=dval.RTOL):
        """Method that finds the maximum mass star

         Args:
            r_init (float, optional): Initial radial coordinate r of the IVP solve. Defaults to R_INIT
            r_final (float, optional): Final radial coordinate r of the IVP solve. Defaults to R_FINAL
            method (str, optional): Method used by the IVP solver. Defaults to IVP_METHOD
            max_step (float, optional): Maximum allowed step size for the IVP solver. Defaults to MAX_STEP
            atol (float, optional): Absolute tolerance of the IVP solver. Defaults to ATOL
            rtol (float, optional): Relative tolerance of the IVP solver. Defaults to RTOL
        """

        # Set the p_center space and rho_center space used to find the maximum mass star
        rho_center = 1.0e16 * uconv.MASS_DENSITY_CGS_TO_GU      # Central density [m^-2]
        p_center = self.star_object.eos.p(rho_center)           # Central pressure [m^-2]
        self.p_center_space = p_center * np.logspace(-3.0, 0.0, 10)
        self.rho_center_space = self.star_object.eos.rho(self.p_center_space)

        # Solve the TOV system and find the maximum mass star through _check_stability method
        self.solve_tov(r_init, r_final, method, max_step, atol, rtol)

        # Redefine the p_center space to a stricter interval
        rho_center = self.maximum_stable_rho_center         # Central density [m^-2]
        p_center = self.star_object.eos.p(rho_center)       # Central pressure [m^-2]
        self.p_center_space = p_center * np.logspace(-0.1, 0.1, 10)
        self.rho_center_space = self.star_object.eos.rho(self.p_center_space)

        # Solve the TOV system and find the maximum mass star through _check_stability method
        self.maximum_stable_rho_center = None
        self.solve_tov(r_init, r_final, method, max_step, atol, rtol)

        # If the maximum stable rho_center is not found in the second TOV solve, raise an exception
        if self.maximum_stable_rho_center is None:
            raise Exception(f"Maximum mass star not found!")

    def solve_tov(self, r_init=dval.R_INIT, r_final=dval.R_FINAL, method=dval.IVP_METHOD, max_step=dval.MAX_STEP, atol=dval.ATOL, rtol=dval.RTOL):
        """Method that solves the TOV system, finding the radius and mass of each star in the family

        Args:
            r_init (float, optional): Initial radial coordinate r of the IVP solve. Defaults to R_INIT
            r_final (float, optional): Final radial coordinate r of the IVP solve. Defaults to R_FINAL
            method (str, optional): Method used by the IVP solver. Defaults to IVP_METHOD
            max_step (float, optional): Maximum allowed step size for the IVP solver. Defaults to MAX_STEP
            atol (float, optional): Absolute tolerance of the IVP solver. Defaults to ATOL
            rtol (float, optional): Relative tolerance of the IVP solver. Defaults to RTOL
        """

        # Initialize the radius and mass arrays with the right size
        self.radius_array = np.zeros(self.p_center_space.size)
        self.mass_array = np.zeros(self.p_center_space.size)

        # Solve the TOV equation for each star in the family
        with alive_bar(self.p_center_space.size) as bar:
            for k in range(self.p_center_space.size):
                self.star_object.solve_tov(self.p_center_space[k], r_init, r_final, method, max_step, atol, rtol)
                self.radius_array[k] = self.star_object.star_radius
                self.mass_array[k] = self.star_object.star_mass
                bar()

        # Execute the stability criterion check and configure the plot
        self._check_stability()
        self._config_plot()

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
        plt.plot(self.plot_dict[x_axis]['value'], self.plot_dict[y_axis]['value'], linewidth=1, label="Calculated curve", marker='.')
        eos_name = self.star_object.eos.eos_name.replace('EOS', ' EOS')
        plt.title(f"{self.plot_dict[y_axis]['name']} vs {self.plot_dict[x_axis]['name']} curve of the {eos_name} star family", y=1.05)
        plt.xlabel(self.plot_dict[x_axis]['label'])
        plt.ylabel(self.plot_dict[y_axis]['label'])

        # If expected curve is received, add a second comparison curve, and enable legend
        if (expected_x is not None) and (expected_y is not None):
            plt.plot(expected_x, expected_y, linewidth=1, label="Expected curve")
            plt.legend()

        # Create the folder if necessary and save the figure
        os.makedirs(figure_path, exist_ok=True)
        x_axis_name = self.plot_dict[x_axis]['name'].lower().replace(' ', '_')
        y_axis_name = self.plot_dict[y_axis]['name'].lower().replace(' ', '_')
        figure_name = f"{y_axis_name}_vs_{x_axis_name}_curve.png"
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


# This logic is a simple example, only executed when this file is run directly in the command prompt
if __name__ == "__main__":

    # Create the EOS object
    eos = PolytropicEOS(k=1.0e8, n=1)

    # Set the pressure at the center and surface of the star
    rho_center = 5.691e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Central density [m^-2]
    p_center = eos.p(rho_center)                                # Central pressure [m^-2]
    p_surface = 0.0                                             # Surface pressure [m^-2]

    # Set the p_center space that characterizes the star family
    p_center_space = p_center * np.logspace(-5.0, 0.0, 50)

    # Define the object
    star_family_object = StarFamily(eos, p_center_space, p_surface)

    # Solve the TOV equation
    star_family_object.solve_tov(max_step=100.0)

    # Plot all curves
    star_family_object.plot_all_curves()
