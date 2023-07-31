import numpy as np
import matplotlib.pyplot as plt
from star_family import StarFamily
from perfect_fluid_star_tides import DeformedStar
from eos_library import PolytropicEOS


class DeformedStarFamily(StarFamily):
    """Class with all the properties and methods necessary to describe a family of perfect fluid deformed stars

    Args:
        StarFamily (class): Parent class with all the properties and methods necessary to describe a family of stars.
        Each star in the family is characterized by a specific value of center pressure (p_center)
    """

    def __init__(self, rho_eos, p_center_space, p_surface):

        # Execute parent class' init function
        super().__init__(rho_eos, p_center_space, p_surface)

        # Create a star object with the first p_center value, using instead the DeformedStar class
        self.star_object = DeformedStar(rho_eos, self.p_center_space[0], p_surface)

    def solve_tidal(self, r_begin=np.finfo(float).eps, r_end=np.inf, r_nsamples=10**6, method='RK45', max_step=np.inf):
        """Method that solves the tidal system for each star in the family, finding the tidal Love number k2

        Args:
            r_begin (float, optional): Radial coordinate r at the beginning of the IVP solve. Defaults to np.finfo(float).eps
            r_end (float, optional): Radial coordinate r at the end of the IVP solve. Defaults to np.inf
            r_nsamples (int, optional): Number of samples used to create the r_space array. Defaults to 10**6
            method (str, optional): Method used by the IVP solver. Defaults to 'RK45'
            max_step (float, optional): Maximum allowed step size for the IVP solver. Defaults to np.inf
        """

        # Create the radius, mass, and k2 arrays to store these star family properties
        self.radius_array = np.zeros(self.p_center_space.size)
        self.mass_array = np.zeros(self.p_center_space.size)
        self.k2_array = np.zeros(self.p_center_space.size)

        # Solve the TOV equation and the tidal equation for each star in the family
        for k in range(self.p_center_space.size):
            self.star_object.solve_tov(self.p_center_space[k], r_begin, r_end, r_nsamples, method, max_step)
            self.star_object.solve_tidal(r_begin, method, max_step)
            self.radius_array[k] = self.star_object.star_radius
            self.mass_array[k] = self.star_object.star_mass
            self.k2_array[k] = self.star_object.k2

    def plot_k2_curve(self, show_plot=True):
        """Method that plots the k2 curve of the star family

        Args:
            show_plot (bool, optional): Flag to enable the command to show the plot at the end. Defaults to True
        """

        # Create a simple plot of the k2 curve
        plt.figure()
        plt.plot(self.mass_array / self.radius_array, self.k2_array, linewidth=1, label="Calculated curve", marker='.')
        plt.title("Tidal Love number curve for the star family")
        plt.xlabel("C = M/R [dimensionless]")
        plt.ylabel("k2 [dimensionless]")

        # Show plot if requested
        if show_plot is True:
            plt.show()


# This logic is a simple example, only executed when this file is run directly in the command prompt
if __name__ == "__main__":

    # Create the EOS object
    eos = PolytropicEOS(k=1.0e8, n=1)

    # Set the pressure at the center and surface of the star
    rho_center = 2.376364e-9        # Center density [m^-2]
    p_center = eos.p(rho_center)    # Center pressure [m^-2]
    p_surface = 0.0                 # Surface pressure [m^-2]

    # Print the values used for p_center and p_surface
    print(f"p_center = {p_center} [m^-2]")
    print(f"p_surface = {p_surface} [m^-2]")

    # Set the p_center space that characterizes the star family
    p_center_space = p_center * np.logspace(-4.0, 1.0, 50)

    # Define the object
    star_family_object = DeformedStarFamily(eos.rho, p_center_space, p_surface)

    # Solve the tidal equation
    star_family_object.solve_tidal(max_step=1.0)

    # Show the radius-mass curve
    star_family_object.plot_radius_mass_curve()

    # Show the k2 curve
    star_family_object.plot_k2_curve()
