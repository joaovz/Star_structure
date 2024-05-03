import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from constants import Constants as const
from constants import DefaultValues as dval
from constants import UnitConversion as uconv
from eos_library import PolytropicEOS
from star_structure import Star


class DeformedStar(Star):
    """Class with all the properties and methods necessary to describe a single deformed star

    Args:
        Star (class): Parent class with all the properties and methods necessary to describe a single star
    """

    def __init__(self, eos, p_center, p_surface=dval.P_SURFACE, r_init=dval.R_INIT, r_final=dval.R_FINAL, method=dval.IVP_METHOD,
                 max_step=dval.MAX_STEP, atol_tov=dval.ATOL_TOV, atol_tidal=dval.ATOL_TIDAL, rtol=dval.RTOL):
        """Initialization method

        Args:
            eos (object): Python object with methods rho, p, drho_dp, and dp_drho that describes the EOS of the star
            p_center (float): Central pressure of the star [m^-2]
            p_surface (float, optional): Surface pressure of the star [m^-2]. Defaults to P_SURFACE
            r_init (float, optional): Initial radial coordinate r of the IVP solve [m]. Defaults to R_INIT
            r_final (float, optional): Final radial coordinate r of the IVP solve [m]. Defaults to R_FINAL
            method (str, optional): Method used by the IVP solver. Defaults to IVP_METHOD
            max_step (float, optional): Maximum allowed step size for the IVP solver [m]. Defaults to MAX_STEP
            atol_tov (float or array of float, optional): Absolute tolerance of the IVP solver for the TOV system. Defaults to ATOL_TOV
            atol_tidal (float, optional): Absolute tolerance of the IVP solver for the tidal system. Defaults to ATOL_TIDAL
            rtol (float, optional): Relative tolerance of the IVP solver. Defaults to RTOL
        """

        # Execute parent class' __init__ method
        super().__init__(eos, p_center, p_surface, r_init, r_final, method, max_step, atol_tov, rtol)

        # Store the input parameters
        self.atol_tidal = atol_tidal

        # Set the initial values: y at r = r_init
        self.y_init = 2.0       # Calculated with the power series solution of the analytical equation valid for r -> 0

        # Initialize deformed star properties
        self.k2 = 0.0               # Tidal Love number [dimensionless]
        self.lambda_tidal = 0.0     # Tidal deformability [dimensionless]

    def _combined_tov_tidal_ode_system(self, r, s):
        """Method that implements the combined TOV+tidal ODE system in the form ``ds/dr = f(r, s)``, used by the IVP solver

        Args:
            r (float): Independent variable of the ODE system (radial coordinate r)
            s (array of float): Array with the dependent variables of the ODE system (p, m, nu, y)

        Returns:
            array of float: Right hand side of the equation ``ds/dr = f(r, s)`` (dp_dr, dm_dr, dnu_dr, dy_dr)
        """

        # Variables of the system
        (p, m, nu, y) = s

        # Call the TOV ODE system
        (dp_dr, dm_dr, dnu_dr) = self._tov_ode_system(r, (p, m, nu))

        # Set the derivative to zero to saturate the function, as this condition indicates the end of integration
        if p <= self.p_surface:
            dy_dr = 0.0
        else:
            # Functions and derivatives evaluated at current r
            rho = self.eos.rho(p)
            drho_dp = self.eos.drho_dp(p)
            exp_lambda = (1 - 2 * m / r)**(-1)

            # Coefficients of the tidal ODE
            c0 = (
                exp_lambda * (
                    - (6 / r**2)
                    + 4 * np.pi * ((rho + p) * drho_dp + 5 * rho + 9 * p)
                )
                - (dnu_dr)**2
            )
            c1 = (2 / r) + exp_lambda * ((2 * m / r**2) + 4 * np.pi * r * (p - rho))

            # ODE system that describes the tidal deformation of the star
            dy_dr = ((1 / r) - c1) * y - (y**2 / r) - c0 * r

        # Return f(r, s) of the combined system
        return (dp_dr, dm_dr, dnu_dr, dy_dr)

    def _process_tov_tidal_ode_solution(self, ode_solution):
        """Method that processes the combined TOV+tidal ODE solution, identifying errors, saving variables,
        and calculating the tidal Love number k2 and the tidal deformability Lambda

        Args:
            ode_solution (Object returned by solve_ivp): Object that contains all information about the ODE solution

        Raises:
            RuntimeError: Exception in case the IVP fails to solve the equation
            RuntimeError: Exception in case the IVP fails to find the ODE termination event
        """

        # Process the TOV ODE solution
        self._process_tov_ode_solution(ode_solution)

        # Unpack the tidal variables
        self.y_ode_solution = ode_solution.y[3]

        # Calculate the compactness C and perturbation at the surface y_s
        c = self.star_mass / self.star_radius
        delta_y_s = - 4 * np.pi * self.star_radius**3 * self.rho_ode_solution[-1] / self.star_mass
        y_s = self.y_ode_solution[-1] + delta_y_s

        # Calculate the tidal Love number k2 and the tidal deformability Lambda
        if c <= const.C_SMALL:
            # First order Taylor expansion of k2 for C -> 0, used for small C values (more stable)
            self.k2 = (2 - y_s) / (2 * (y_s + 3)) + ((5 * y_s**2 + 10 * y_s - 30) * c) / (2 * (y_s + 3)**2)
        else:
            # Complete k2 expression
            self.k2 = (
                (8 / 5) * c**5 * ((1 - 2 * c)**2) * (2 + 2 * c * (y_s - 1) - y_s) / (
                    2 * c * (6 - 3 * y_s + 3 * c * (5 * y_s - 8))
                    + 4 * c**3 * (13 - 11 * y_s + c * (3 * y_s - 2) + 2 * c**2 * (1 + y_s))
                    + 3 * ((1 - 2 * c)**2) * (2 - y_s + 2 * c * (y_s - 1)) * np.log(1 - 2 * c)
                )
            )
        self.lambda_tidal = (2 / 3) * self.k2 * c**(-5)

    def solve_combined_tov_tidal(self, p_center=None, show_results=True):
        """Method that solves the combined TOV+tidal system for the star, finding p, m, nu, and y

        Args:
            p_center (float, optional): Central pressure of the star [m^-2]. Defaults to None
            show_results (bool, optional): Flag that enables the results printing after the solve. Defaults to True

        Raises:
            ValueError: Exception in case the initial radial coordinate is too large
            RuntimeError: Exception in case the IVP fails to solve the equation
            RuntimeError: Exception in case the IVP fails to find the ODE termination event
        """

        # Calculate the TOV ODE system initial values
        self._calc_tov_init_values(p_center)

        # Solve the combined TOV+tidal ODE system
        atol = list(self.atol_tov) + [self.atol_tidal]
        ode_solution = solve_ivp(
            self._combined_tov_tidal_ode_system,
            (self.r_init, self.r_final),
            (self.p_init, self.m_init, self.nu_init, self.y_init),
            self.method,
            events=self._tov_ode_termination_event,
            max_step=self.max_step,
            atol=atol,
            rtol=self.rtol)

        # Process the TOV+tidal ODE solution
        self._process_tov_tidal_ode_solution(ode_solution)

        # Show results if requested
        if show_results is True:
            self.print_results()

    def print_results(self):
        """Method that prints the results found
        """

        # Execute parent class' print_results method
        super().print_results()

        # Print the results
        print(f"Tidal Love number (k2) = {(self.k2):e} [dimensionless]")
        print(f"Tidal deformability (Lambda) = {(self.lambda_tidal):e} [dimensionless]")
        print(f"Perturbation (y(R)) = {(self.y_ode_solution[-1]):e} [dimensionless]")

    def plot_all_curves(self, figure_path=Star.FIGURES_PATH):
        """Method that plots the solution found

        Args:
            figure_path (str, optional): Path used to save the figure. Defaults to FIGURES_PATH
        """

        # Execute parent class' plot_all_curves method
        super().plot_all_curves(figure_path)

        # Show a simple plot of the solution
        plt.figure(figsize=(6.0, 4.5))
        plt.plot(self.r_ode_solution / 10**3, self.y_ode_solution, linewidth=1)
        plt.xlabel("$r ~ [km]$", fontsize=10)
        plt.ylabel("$y ~ [dimensionless]$", fontsize=10)

        # Create the folder if necessary and save the figure
        os.makedirs(figure_path, exist_ok=True)
        figure_name = "star_perturbation_graph.pdf"
        complete_path = os.path.join(figure_path, figure_name)
        plt.savefig(complete_path, bbox_inches="tight")

        # Show graph
        plt.show()


def main():
    """Main logic
    """

    # Constants
    MAX_RHO = 5.691e15 * uconv.MASS_DENSITY_CGS_TO_GU       # Maximum density [m^-2]

    # EOS parameters
    k = 1.0e8       # Proportional constant [dimensionless]
    n = 1           # Polytropic index [dimensionless]

    # Create the EOS object
    eos = PolytropicEOS(k, n)

    # Set the central pressure of the star
    rho_center = MAX_RHO                # Central density [m^-2]
    p_center = eos.p(rho_center)        # Central pressure [m^-2]

    # Create the star object
    star_object = DeformedStar(eos, p_center)

    # Solve the combined TOV+tidal system and plot all curves
    star_object.solve_combined_tov_tidal()
    star_object.plot_all_curves()


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
