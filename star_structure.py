import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from constants import DefaultValues as dval
from constants import UnitConversion as uconv
from eos_library import PolytropicEOS


class Star:
    """Class with all the properties and methods necessary to describe a single star
    """

    # Class constants
    FIGURES_PATH = "figures/star"       # Path of the figures folder

    def __init__(self, eos, p_center, p_surface=dval.P_SURFACE, r_init=dval.R_INIT, r_final=dval.R_FINAL,
                 method=dval.IVP_METHOD, max_step=dval.MAX_STEP, atol_tov=dval.ATOL_TOV, rtol=dval.RTOL):
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
            rtol (float, optional): Relative tolerance of the IVP solver. Defaults to RTOL
        """

        # Store the input parameters
        self.eos = eos
        self.p_center = p_center
        self.p_surface = p_surface
        self.r_init = r_init
        self.r_final = r_final
        self.method = method
        self.max_step = max_step
        self.atol_tov = atol_tov
        self.rtol = rtol

        # Set the initial values: pressure, mass, and metric function at r = r_init
        self.p_init = p_center          # Initial pressure [m^-2]
        self.m_init = 0.0               # Initial mass [m]
        self.nu_init = 1.0              # Initial metric function (g_tt = -e^nu) [dimensionless]

        # Initialize star properties
        self.star_radius = 0.0          # Star radius (R) [m]
        self.star_mass = 0.0            # Star mass (M) [m]

    def _tov_ode_system(self, r, s):
        """Method that implements the TOV ODE system in the form ``ds/dr = f(r, s)``, used by the IVP solver

        Args:
            r (float): Independent variable of the ODE system (radial coordinate r)
            s (array of float): Array with the dependent variables of the ODE system (p, m, nu)

        Returns:
            array of float: Right hand side of the equation ``ds/dr = f(r, s)`` (dp_dr, dm_dr, dnu_dr)
        """

        # Variables of the system
        (p, m, nu) = s

        # Set derivatives to zero to saturate functions, as this condition indicates the end of integration
        if p <= self.p_surface:
            dp_dr = 0.0
            dm_dr = 0.0
            dnu_dr = 0.0
        else:
            # ODE System that describes the interior structure of the star
            rho = self.eos.rho(p)
            dnu_dr = (2 * (m + 4 * np.pi * r**3 * p)) / (r * (r - 2 * m))       # Rate of change of the metric function
            dp_dr = -((rho + p) / 2) * dnu_dr                                   # Rate of change of the pressure
            dm_dr = 4 * np.pi * r**2 * rho                                      # Rate of change of the mass

        return (dp_dr, dm_dr, dnu_dr)

    def _tov_ode_termination_event(self, r, s):
        """Event method used by the IVP solver. The solver will find an accurate value of r at which
        ``event(r, s(r)) = 0`` using a root-finding algorithm

        Args:
            r (float): Independent variable of the ODE system (radial coordinate r)
            s (array of float): Array with the dependent variables of the ODE system (p, m, nu)

        Returns:
            float: ``p - p_surface``
        """
        return s[0] - self.p_surface

    _tov_ode_termination_event.terminal = True      # Set the event as a terminal event, terminating the integration of the ODE

    def _calc_tov_init_values(self, p_center=None):
        """Method that calculates the initial values used by the TOV solver

        Args:
            p_center (float, optional): Central pressure of the star [m^-2]. Defaults to None

        Raises:
            ValueError: Exception in case the initial radial coordinate is too large
        """

        # Transfer the p_center parameter if it was passed as an argument
        if p_center is not None:
            self.p_center = p_center

        # Calculate the coefficients of the series solution, used to calculate the initial conditions
        p_c = self.p_center
        rho_c = self.eos.rho(p_c)
        drho_dp_c = self.eos.drho_dp(p_c)
        p_2 = - (2 / 3) * np.pi * (rho_c + p_c) * (rho_c + 3 * p_c)
        rho_2 = p_2 * drho_dp_c
        m_3 = (4 / 3) * np.pi * rho_c
        m_5 = (4 / 5) * np.pi * rho_2

        # Calculate the r_init_max, based on the allowed relative error tolerance
        r_max_p = (np.abs(p_c / p_2) * self.rtol)**(1 / 2)
        r_max_rho = (np.abs(rho_c / rho_2) * self.rtol)**(1 / 2)
        r_init_max = min(r_max_p, r_max_rho)
        if self.r_init > r_init_max:
            raise ValueError(f"The initial radial coordinate is too large: (r_init = {self.r_init} [m]) > (r_init_max = {r_init_max} [m])")

        # Calculate the initial values, given by the series solution near r = 0
        r = self.r_init
        self.p_init = p_c + p_2 * r**2
        self.m_init = m_3 * r**3 + m_5 * r**5

    def _process_tov_ode_solution(self, ode_solution):
        """Method that processes the TOV ODE solution, identifying errors and saving variables

        Args:
            ode_solution (Object returned by solve_ivp): Object that contains all information about the ODE solution

        Raises:
            RuntimeError: Exception in case the IVP fails to solve the equation
            RuntimeError: Exception in case the IVP fails to find the ODE termination event
        """

        # Check the ODE solution status and treat each case
        if ode_solution.status == -1:
            raise RuntimeError(ode_solution.message)
        elif ode_solution.status != 1:
            raise RuntimeError("The solver did not find the ODE termination event")

        # Unpack the variables. Not using tuple unpack to allow reuse of this code
        self.r_ode_solution = ode_solution.t
        self.p_ode_solution = ode_solution.y[0]
        self.m_ode_solution = ode_solution.y[1]
        self.nu_ode_solution = ode_solution.y[2]
        self.rho_ode_solution = self.eos.rho(self.p_ode_solution)

        # Get the star radius, star mass, and surface nu from the ODE termination event
        self.star_radius = ode_solution.t_events[0][0]
        self.star_mass = ode_solution.y_events[0][0][1]
        nu_surface = ode_solution.y_events[0][0][2]

        # Adjust metric function with the correct boundary condition (nu(R) = ln(1 - 2M/R))
        self.nu_ode_solution += - nu_surface + np.log(1 - 2 * self.star_mass / self.star_radius)

    def solve_tov(self, p_center=None, show_results=True):
        """Method that solves the TOV system for the star, finding the functions p(r), m(r), nu(r), and rho(r)

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

        # Solve the TOV ODE system
        ode_solution = solve_ivp(
            self._tov_ode_system,
            (self.r_init, self.r_final),
            (self.p_init, self.m_init, self.nu_init),
            self.method,
            events=self._tov_ode_termination_event,
            max_step=self.max_step,
            atol=self.atol_tov,
            rtol=self.rtol)

        # Process the TOV ODE solution
        self._process_tov_ode_solution(ode_solution)

        # Show results if requested
        if show_results is True:
            self.print_results()

    def print_results(self):
        """Method that prints the results found
        """

        # Print the results
        print(f"\n##########    Single star solve results    ##########")
        print(f"Star radius (R) = {(self.star_radius / 10**3):e} [km]")
        print(f"Star mass (M) = {(self.star_mass * uconv.MASS_GU_TO_SOLAR_MASS):e} [solar mass]")
        print(f"Compactness (C = M/R) = {(self.star_mass / self.star_radius):e} [dimensionless]")

    def plot_all_curves(self, figure_path=FIGURES_PATH):
        """Method that plots the solution found

        Args:
            figure_path (str, optional): Path used to save the figure. Defaults to FIGURES_PATH
        """

        # Show a simple plot of the solution
        plt.figure(figsize=(6.0, 4.5))
        r_ode_solution_km = self.r_ode_solution / 10**3
        plt.plot(r_ode_solution_km, self.p_ode_solution * 1e-36 * uconv.PRESSURE_GU_TO_CGS, linewidth=1, label="$p ~ [10^{36} ~ dyn \\cdot cm^{-2}]$")
        plt.plot(r_ode_solution_km, self.m_ode_solution * uconv.MASS_GU_TO_SOLAR_MASS, linewidth=1, label="$m ~ [M_{\\odot}]$")
        plt.plot(r_ode_solution_km, self.nu_ode_solution, linewidth=1, label="$\\nu ~ [dimensionless]$")
        plt.plot(r_ode_solution_km, self.rho_ode_solution * 1e-15 * uconv.MASS_DENSITY_GU_TO_CGS, linewidth=1, label="$\\rho ~ [10^{15} ~ g \\cdot cm^{-3}]$")
        plt.title(f"TOV solution for the {self.eos.eos_name.replace("EOS", " EOS")} star", y=1.05, fontsize=11)
        plt.xlabel("$r ~ [km]$", fontsize=10)
        plt.legend()

        # Create the folder if necessary and save the figure
        os.makedirs(figure_path, exist_ok=True)
        figure_name = "star_structure_graph.pdf"
        complete_path = os.path.join(figure_path, figure_name)
        plt.savefig(complete_path)

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
    star_object = Star(eos, p_center)

    # Solve the TOV system and plot all curves
    star_object.solve_tov()
    star_object.plot_all_curves()


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
