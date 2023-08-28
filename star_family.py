import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from alive_progress import alive_bar
from star_structure import Star
from eos_library import PolytropicEOS


class StarFamily:
    """Class with all the properties and methods necessary to describe a family of stars. Each star
    in the family is characterized by a specific value of center pressure (p_center)
    """

    def __init__(self, rho_eos, p_center_space, p_surface):
        """Initialization method

        Args:
            rho_eos (function): Python function in the format rho(p) that describes the EOS of the stars
            p_center_space (array of float): Array with the center pressure of each star in the family [m^-2]
            p_surface (float): Surface pressure of the stars [m^-2]
        """

        # Store the input parameters
        self.p_center_space = p_center_space

        # Create a star object with the first p_center value
        self.star_object = Star(rho_eos, self.p_center_space[0], p_surface)

        # Calculate the rho_center_space
        self.rho_center_space = self.star_object.rho_eos(self.p_center_space)

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

        # Create the radius and mass arrays to store these star family properties
        self.radius_array = np.zeros(self.p_center_space.size)
        self.mass_array = np.zeros(self.p_center_space.size)

        # Solve the TOV equation for each star in the family
        with alive_bar(self.p_center_space.size) as bar:
            for k in range(self.p_center_space.size):
                self.star_object.solve_tov(self.p_center_space[k], r_begin, r_end, method, max_step, atol, rtol)
                self.radius_array[k] = self.star_object.star_radius
                self.mass_array[k] = self.star_object.star_mass
                bar()

    def check_stability(self):
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

    def plot_radius_mass_curve(self, show_plot=True):
        """Method that plots the radius-mass curve of the star family

        Args:
            show_plot (bool, optional): Flag to enable the command to show the plot at the end. Defaults to True
        """

        # Create a simple plot of the radius-mass curve
        plt.figure()
        plt.plot(self.radius_array / 10**3, self.mass_array / self.star_object.SOLAR_MASS, linewidth=1, label="Calculated curve", marker='.')
        plt.title("Radius-Mass curve for the star family")
        plt.xlabel("$R ~ [km]$")
        plt.ylabel("$M ~ [M_{\\odot}]$")

        # Show plot if requested
        if show_plot is True:
            plt.show()

    def plot_dm_drho_center_curve(self):
        """Method that plots the derivative of the mass with respect to rho_center curve for the star family
        """

        # Execute the stability criterion check
        self.check_stability()

        # Create a simple plot of the derivative of the mass with respect to rho_center
        plt.figure()
        plt.plot(self.rho_center_space, self.dm_drho_center, linewidth=1, label="Calculated curve", marker='.')
        plt.title("$\\dfrac{\\partial M}{\partial \\rho_{c}}$ curve for the star family")
        plt.xlabel("$\\rho ~ [m^{-2}]$")
        plt.ylabel("$\\dfrac{\\partial M}{\partial \\rho_{c}} ~ [m^3]$")
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
    star_family_object = StarFamily(eos.rho, p_center_space, p_surface)

    # Solve the TOV equation
    star_family_object.solve_tov(max_step=100.0)

    # Show the radius-mass curve
    star_family_object.plot_radius_mass_curve()

    # Show the derivative of the mass with respect to rho_center curve
    star_family_object.plot_dm_drho_center_curve()
