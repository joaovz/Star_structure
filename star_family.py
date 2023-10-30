import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from alive_progress import alive_bar
from star_structure import Star
from eos_library import PolytropicEOS


class StarFamily:
    """Class with all the properties and methods necessary to describe a family of stars. Each star
    in the family is characterized by a specific value of central pressure (p_center)
    """

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

        # Create the radius and mass arrays to store these star family properties
        self.radius_array = np.zeros(self.p_center_space.size)
        self.mass_array = np.zeros(self.p_center_space.size)

    def _check_stability(self):
        """Method that checks the stability criterion for the star family
        """

        # Create the mass vs rho_center interpolated function and calculate its derivative, used in the stability criterion
        self.mass_rho_center_spline = CubicSpline(self.rho_center_space, self.mass_array, extrapolate=False)
        self.dm_drho_center_spline = self.mass_rho_center_spline.derivative()
        self.dm_drho_center = self.dm_drho_center_spline(self.rho_center_space)

        # Calculate the maximum stable rho_center
        dm_drho_center_roots = self.dm_drho_center_spline.roots()
        if dm_drho_center_roots.size > 0:
            print(f"Maximum stable rho_center = {dm_drho_center_roots} [m^-2]")
        else:
            print("Maximum stable rho_center not reached")

    def _config_plot(self):
        """Method that configures the plotting
        """

        # Create a dictionary with all the functions used in plotting, with each name and label description
        self.plot_dict = {
            "p_c": {
                "name": "Central pressure",
                "label": "$p_{c} ~ [m^{-2}]$",
                "value": self.p_center_space,
            },
            "rho_c": {
                "name": "Central density",
                "label": "$\\rho_{c} ~ [m^{-2}]$",
                "value": self.rho_center_space,
            },
            "R": {
                "name": "Radius",
                "label": "$R ~ [km]$",
                "value": self.radius_array / 10**3,
            },
            "M": {
                "name": "Mass",
                "label": "$M ~ [M_{\\odot}]$",
                "value": self.mass_array / self.star_object.SOLAR_MASS,
            },
            "C": {
                "name": "Compactness",
                "label": "$C = M/R ~ [dimensionless]$",
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

    def solve_tov(self, r_begin=np.finfo(float).eps, r_end=np.inf, method='RK45', max_step=np.inf, atol=1e-9, rtol=1e-6):
        """Method that solves the TOV system, finding the radius and mass of each star in the family

        Args:
            r_begin (float, optional): Radial coordinate r at the beginning of the IVP solve. Defaults to np.finfo(float).eps
            r_end (float, optional): Radial coordinate r at the end of the IVP solve. Defaults to np.inf
            method (str, optional): Method used by the IVP solver. Defaults to 'RK45'
            max_step (float, optional): Maximum allowed step size for the IVP solver. Defaults to np.inf
            atol (float, optional): Absolute tolerance of the IVP solver. Defaults to 1e-9
            rtol (float, optional): Relative tolerance of the IVP solver. Defaults to 1e-6
        """

        # Solve the TOV equation for each star in the family
        with alive_bar(self.p_center_space.size) as bar:
            for k in range(self.p_center_space.size):
                self.star_object.solve_tov(self.p_center_space[k], r_begin, r_end, method, max_step, atol, rtol)
                self.radius_array[k] = self.star_object.star_radius
                self.mass_array[k] = self.star_object.star_mass
                bar()

        # Execute the stability criterion check and configure the plot
        self._check_stability()
        self._config_plot()

    def plot_curve(self, x_axis="R", y_axis="M", figure_path="figures/star_family", expected_x=None, expected_y=None):
        """Method that plots some curve of the star family

        Args:
            x_axis (str, optional): Key of self.plot_dict to indicate the x_axis used. Defaults to "R"
            y_axis (str, optional): Key of self.plot_dict to indicate the y_axis used. Defaults to "M"
            figure_path (str, optional): Path used to save the figure. Defaults to "figures/star_family"
            expected_x (array of float, optional): Array with the x values of the expected curve. Defaults to None
            expected_y (array of float, optional): Array with the y values of the expected curve. Defaults to None
        """

        # Create a simple plot
        plt.figure()
        plt.plot(self.plot_dict[x_axis]['value'], self.plot_dict[y_axis]['value'], linewidth=1, label="Calculated curve", marker='.')
        plt.title(f"{self.plot_dict[y_axis]['name']} vs {self.plot_dict[x_axis]['name']} curve of the star family")
        plt.xlabel(self.plot_dict[x_axis]['label'])
        plt.ylabel(self.plot_dict[y_axis]['label'])

        # If expected curve is received, add a second comparison curve, and enable legend
        if (expected_x is not None) and (expected_y is not None):
            plt.plot(expected_x, expected_y, linewidth=1, label="Expected curve")
            plt.legend()

        # Create the folder if necessary and save the figure
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        x_axis_name = self.plot_dict[x_axis]['name'].lower().replace(' ', '_')
        y_axis_name = self.plot_dict[y_axis]['name'].lower().replace(' ', '_')
        plt.savefig(f"{figure_path}/{y_axis_name}_vs_{x_axis_name}_curve.png")

    def plot_all_curves(self, figures_path="figures/star_family"):
        """Method that plots all curves specified by the self.curves_list

        Args:
            figures_path (str, optional): Path used to save the figures. Defaults to "figures/star_family"
        """

        for axis in self.curves_list:
            self.plot_curve(axis[0], axis[1], figures_path)
        plt.show()


# This logic is a simple example, only executed when this file is run directly in the command prompt
if __name__ == "__main__":

    # Create the EOS object
    eos = PolytropicEOS(k=1.0e8, n=1)

    # Set the pressure at the center and surface of the star
    rho_center = 4.3e-9             # Center density [m^-2]
    p_center = eos.p(rho_center)    # Center pressure [m^-2]
    p_surface = 0.0                 # Surface pressure [m^-2]

    # Print the values used for p_center and p_surface
    print(f"p_center = {p_center} [m^-2]")
    print(f"p_surface = {p_surface} [m^-2]")

    # Set the p_center space that characterizes the star family
    p_center_space = p_center * np.logspace(-5.0, 0.0, 50)

    # Define the object
    star_family_object = StarFamily(eos, p_center_space, p_surface)

    # Solve the TOV equation
    star_family_object.solve_tov(max_step=100.0)

    # Plot all curves
    star_family_object.plot_all_curves()
