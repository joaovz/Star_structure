import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar
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

        # Execute parent class' __init__ method
        super().__init__(rho_eos, p_center_space, p_surface)

        # Create a star object with the first p_center value, using instead the DeformedStar class
        self.star_object = DeformedStar(rho_eos, self.p_center_space[0], p_surface)

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

    def solve_tidal(self, r_begin=np.finfo(float).eps, r_end=np.inf, method='RK45', max_step=np.inf, atol=1e-9, rtol=1e-6):
        """Method that solves the tidal system for each star in the family, finding the tidal Love number k2

        Args:
            r_begin (float, optional): Radial coordinate r at the beginning of the IVP solve. Defaults to np.finfo(float).eps
            r_end (float, optional): Radial coordinate r at the end of the IVP solve. Defaults to np.inf
            method (str, optional): Method used by the IVP solver. Defaults to 'RK45'
            max_step (float, optional): Maximum allowed step size for the IVP solver. Defaults to np.inf
            atol (float, optional): Absolute tolerance of the IVP solver. Defaults to 1e-9
            rtol (float, optional): Relative tolerance of the IVP solver. Defaults to 1e-6
        """

        # Solve the TOV equation and the tidal equation for each star in the family
        with alive_bar(self.p_center_space.size) as bar:
            for k in range(self.p_center_space.size):
                self.star_object.solve_tov(self.p_center_space[k], r_begin, r_end, method, max_step, atol, rtol)
                self.star_object.solve_tidal(r_begin, method, max_step, atol, rtol)
                self.radius_array[k] = self.star_object.star_radius
                self.mass_array[k] = self.star_object.star_mass
                self.k2_array[k] = self.star_object.k2
                bar()

        # Execute the stability criterion check and configure the plot
        self._check_stability()
        self._config_plot()


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
    star_family_object = DeformedStarFamily(eos.rho, p_center_space, p_surface)

    # Solve the tidal equation
    star_family_object.solve_tidal(max_step=100.0)

    # Plot all curves
    star_family_object.plot_all_curves()
