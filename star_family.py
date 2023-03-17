import numpy as np
import matplotlib.pyplot as plt
from star_structure import Star


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
        self.star_object = Star(rho_eos, p_center_space[0], p_surface)

    def solve_tov(self, r_begin=np.finfo(float).eps, r_end=np.inf, r_nsamples=10**6, method='RK45'):
        """Method that solves the TOV system, finding the radius and mass of each star in the family

        Args:
            r_begin (float, optional): Radial coordinate r at the beginning of the IVP solve. Defaults to np.finfo(float).eps
            r_end (float, optional): Radial coordinate r at the end of the IVP solve. Defaults to np.inf
            r_nsamples (int, optional): Number of samples used to create the r_space array. Defaults to 10**6
            method (str, optional): Method used by the IVP solver. Defaults to 'RK45'
        """

        # Create the radius and mass arrays to store these star family properties
        self.radius_array = np.zeros(self.p_center_space.size)
        self.mass_array = np.zeros(self.p_center_space.size)

        # Solve the TOV equation for each star in the family
        for k in range(self.p_center_space.size):
            self.star_object.solve_tov(p_center_space[k], r_begin, r_end, r_nsamples, method)
            self.radius_array[k] = self.star_object.star_radius
            self.mass_array[k] = self.star_object.star_mass

    def show_radius_mass_curve(self):
        """Method that plots the radius-mass curve of the star family
        """

        # Show a simple plot of the radius-mass curve
        plt.figure()
        plt.plot(self.radius_array / 10**3, self.mass_array / self.star_object.SOLAR_MASS, linewidth=1)
        plt.title("Radius-Mass curve for the star family")
        plt.xlabel("R [km]")
        plt.ylabel("$M / M_{\odot}$")
        plt.show()


# This logic is a simple example, only executed when this file is run directly in the command prompt
if __name__ == "__main__":

    # Set the EOS and pressure at the center and surface of the star
    def rho(p):
        c = 1.0e8       # [m^2]
        return (np.abs(p / c))**(1 / 2)

    def p(rho):
        c = 1.0e8       # [m^2]
        return c * rho**2

    rho_center = 2.376364e-9        # Center density [m^-2]
    p_center = rho(rho_center)      # Center pressure [m^-2]
    p_surface = 0.0                 # Surface pressure [m^-2]

    # Set the p_center space that characterizes the star family
    p_center_space = p_center * np.linspace(1.0, 2.0, 100)

    # Define the object
    star_family_object = StarFamily(rho, p_center_space, p_surface)

    # Solve the TOV equation
    star_family_object.solve_tov()

    # Show the radius-mass curve
    star_family_object.show_radius_mass_curve()
