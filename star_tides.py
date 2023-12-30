import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from constants import DefaultValues as dval
from constants import UnitConversion as uconv
from eos_library import PolytropicEOS
from star_structure import Star


class DeformedStar(Star):
    """Class with all the properties and methods necessary to describe a single deformed star

    Args:
        Star (class): Parent class with all the properties and methods necessary to describe a single star
    """

    # Class constants
    FIGURES_PATH = "figures/star_tides"

    def __init__(self, eos, p_center, p_surface):

        # Execute parent class' __init__ method
        super().__init__(eos, p_center, p_surface)

        # Set the initial values: g and h at r = r_init
        self.g_init = 0.0
        self.h_init = 0.0

        # Initialize deformed star properties: tidal Love number
        self.k2 = 0.0       # Tidal Love number [dimensionless]

    def _tidal_ode_system(self, r, y):
        """Method that implements the tidal ODE system in the form ``dy/dr = f(r, y)``, used by the IVP solver

        Args:
            r (float): Independent variable of the ODE system (radial coordinate r)
            y (array of float): Array with the dependent variables of the ODE system (g and h)

        Returns:
            array of float: Right hand side of the equation ``dy/dr = f(r, y)`` (dg_dr, dh_dr)
        """

        # Variables of the system
        (g, h) = y

        # Functions evaluated at current r
        p = self.p_spline_function(r)
        m = self.m_spline_function(r)
        rho = self.eos.rho(p)
        exp_lambda = (1 - 2 * m / r)**(-1)

        # Derivatives of the functions evaluated at current r
        dnu_dr = (2 * (m + 4 * np.pi * r**3 * p)) / (r * (r - 2 * m))
        drho_dp = self.eos.drho_dp(p)

        # Coefficients of the ODE
        l = 2
        c0 = (
            exp_lambda * (
                - (l * (l + 1) / r**2)
                + 4 * np.pi * (rho + p) * drho_dp
                + 4 * np.pi * (5 * rho + 9 * p)
            )
            - (dnu_dr)**2
        )
        c1 = (2 / r) + exp_lambda * ((2 * m / r**2) + 4 * np.pi * r * (p - rho))

        # ODE System that describes the tidal deformation of the star
        dg_dr = -(c1 * g + c0 * h)
        dh_dr = g
        return (dg_dr, dh_dr)

    def solve_tidal(self, r_init=dval.R_INIT, method=dval.IVP_METHOD, max_step=dval.MAX_STEP, atol=dval.ATOL, rtol=dval.RTOL):
        """Method that solves the tidal system for the star, finding the tidal Love number k2

        Args:
            r_init (float, optional): Initial radial coordinate r of the IVP solve. Defaults to R_INIT
            method (str, optional): Method used by the IVP solver. Defaults to IVP_METHOD
            max_step (float, optional): Maximum allowed step size for the IVP solver. Defaults to MAX_STEP
            atol (float, optional): Absolute tolerance of the IVP solver. Defaults to ATOL
            rtol (float, optional): Relative tolerance of the IVP solver. Defaults to RTOL

        Raises:
            RuntimeError: Exception in case the IVP fails to solve the equation
        """

        # Calculate the initial values, given by the series solution near r = 0
        l = 2
        self.g_init = l * r_init**(l - 1)
        self.h_init = r_init**l

        # Solve the ODE system
        ode_solution = solve_ivp(
            self._tidal_ode_system,
            (r_init, self.star_radius),
            (self.g_init, self.h_init),
            method,
            max_step=max_step,
            atol=atol,
            rtol=rtol)
        self.r_tidal_ode_solution = ode_solution.t
        (self.g_tidal_ode_solution, self.h_tidal_ode_solution) = ode_solution.y

        # Check the ODE solution status and treat the exception case
        if ode_solution.status == -1:
            raise RuntimeError(ode_solution.message)

        # Calculate the tidal Love number k2, that represents the star tidal deformation
        c = self.star_mass / self.star_radius
        y = self.star_radius * self.g_tidal_ode_solution[-1] / self.h_tidal_ode_solution[-1]
        self.k2 = (
            (8 / 5) * c**5 * ((1 - 2 * c)**2) * (2 + 2 * c * (y - 1) - y) / (
                2 * c * (6 - 3 * y + 3 * c * (5 * y - 8))
                + 4 * c**3 * (13 - 11 * y + c * (3 * y - 2) + 2 * c**2 * (1 + y))
                + 3 * ((1 - 2 * c)**2) * (2 - y + 2 * c * (y - 1)) * np.log(1 - 2 * c)
            )
        )

    def plot_perturbation_curves(self, figure_path=FIGURES_PATH):
        """Method that prints the tidal Love number (k2) and the compactness of the star, and plots the perturbation solution found

        Args:
            figure_path (str, optional): Path used to save the figure. Defaults to FIGURES_PATH
        """

        # Print the tidal Love number (k2) and the compactness of the star
        print(f"Tidal Love number (k2) = {(self.k2):e} [dimensionless]")
        print(f"Compactness (C = M/R) = {(self.star_mass / self.star_radius):e} [dimensionless]")

        # Show a simple plot of the solution
        plt.figure()
        r_tidal_ode_solution_km = self.r_tidal_ode_solution / 10**3
        plt.plot(r_tidal_ode_solution_km, self.g_tidal_ode_solution, linewidth=1, label="$g ~ [m^{-1}]$")
        plt.plot(r_tidal_ode_solution_km, self.h_tidal_ode_solution, linewidth=1, label="$h ~ [dimensionless]$")
        plt.title(f"Perturbation solution for the {self.eos.eos_name.replace('EOS', ' EOS')} star", y=1.05)
        plt.xlabel("$r ~ [km]$")
        plt.legend()

        # Create the folder if necessary and save the figure
        os.makedirs(figure_path, exist_ok=True)
        figure_name = "star_perturbation_graph.png"
        complete_path = os.path.join(figure_path, figure_name)
        plt.savefig(complete_path)

        # Show graph
        plt.show()


# This logic is a simple example, only executed when this file is run directly in the command prompt
if __name__ == "__main__":

    # Create the EOS object
    eos = PolytropicEOS(k=1.0e8, n=1)

    # Set the pressure at the center and surface of the star
    rho_center = 5.691e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Central density [m^-2]
    p_center = eos.p(rho_center)                                # Central pressure [m^-2]
    p_surface = 0.0                                             # Surface pressure [m^-2]

    # Define the object
    star_object = DeformedStar(eos, p_center, p_surface)

    # Solve the TOV equation
    star_object.solve_tov(max_step=100.0)

    # Plot the star structure curves
    star_object.plot_star_structure_curves()

    # Solve the tidal deformation
    star_object.solve_tidal(max_step=100.0)

    # Plot the perturbation curves
    star_object.plot_perturbation_curves()
