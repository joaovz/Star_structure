import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from constants import DefaultValues as dval
from constants import UnitConversion as uconv
from eos_library import PolytropicEOS


class Star:
    """Class with all the properties and methods necessary to describe a single star
    """

    def __init__(self, eos, p_center, p_surface):
        """Initialization method

        Args:
            eos (object): Python object with methods rho, p, drho_dp, and dp_drho that describes the EOS of the star
            p_center (float): Central pressure of the star [m^-2]
            p_surface (float): Surface pressure of the star [m^-2]
        """

        # Store the input parameters
        self.p_center = p_center        # Central pressure (p(r = 0)) [m^-2]
        self.p_surface = p_surface      # Surface pressure (p(r = R)) [m^-2]. Boundary value for the termination of the ODE integration

        # Set the EOS object
        self.eos = eos

        # Set the initial values: pressure, mass, and metric function at r = r_init
        self.p_init = p_center          # Initial pressure [m^-2]
        self.m_init = 0.0               # Initial mass [m]
        self.nu_init = 0.0              # Initial metric function (g_tt = -e^nu) [dimensionless]

        # Initialize star properties
        self.star_radius = 0.0          # Star radius (R) [m]
        self.star_mass = 0.0            # Star mass (M) [m]

    def _ode_system(self, r, y):
        """Method that implements the TOV ODE system in the form ``dy/dr = f(r, y)``, used by the IVP solver

        Args:
            r (float): Independent variable of the ODE system (radial coordinate r)
            y (array of float): Array with the dependent variables of the ODE system (p, m, and nu)

        Returns:
            array of float: Right hand side of the equation ``dy/dr = f(r, y)`` (dp_dr, dm_dr, dnu_dr)

        Raises:
            Exception: Exception in case the pressure is outside the acceptable range (p > p_center)
            Exception: Exception in case the EOS function didn't return a number
        """

        # Variables of the system
        (p, m, nu) = y

        # Check if p is outside the acceptable range and raise an exception in this case
        if p > self.p_center:
            raise Exception(f"The pressure is outside the acceptable range (p = {p}): p > p_center")

        # Set derivatives to zero to saturate functions, as this condition indicates the end of integration
        if p <= self.p_surface:
            dp_dr = 0.0
            dm_dr = 0.0
            dnu_dr = 0.0
        else:
            # Calculate rho, check if it is some invalid value, and raise an exception in this case
            rho = self.eos.rho(p)
            if np.isnan(rho):
                raise Exception(f"The EOS function didn't return a number: p = {p}, rho = {rho}")

            # ODE System that describes the interior structure of the star
            dnu_dr = (2 * (m + 4 * np.pi * r**3 * p)) / (r * (r - 2 * m))       # Rate of change of the metric function
            dp_dr = -((rho + p) / 2) * dnu_dr                                   # TOV equation
            dm_dr = 4 * np.pi * r**2 * rho                                      # Rate of change of the mass

        return (dp_dr, dm_dr, dnu_dr)

    def _ode_termination_event(self, r, y):
        """Event method used by the IVP solver. The solver will find an accurate value of r at which
        ``event(r, y(r)) = 0`` using a root-finding algorithm

        Args:
            r (float): Independent variable of the ODE system (radial coordinate r)
            y (array of float): Array with the dependent variables of the ODE system (p, m, and nu)

        Returns:
            float: ``p - p_surface``
        """

        return y[0] - self.p_surface            # Condition of the event: trigger when condition == 0 (p == p_surface)
    _ode_termination_event.terminal = True      # Set the event as a terminal event, terminating the integration of the ODE

    def solve_tov(self, p_center=None, r_init=dval.R_INIT, r_final=dval.R_FINAL, method=dval.IVP_METHOD, max_step=dval.MAX_STEP, atol=dval.ATOL, rtol=dval.RTOL):
        """Method that solves the TOV system for the star, finding the functions p(r), m(r), nu(r), and rho(r)

        Args:
            p_center (float, optional): Central pressure of the star [m^-2]
            r_init (float, optional): Initial radial coordinate r of the IVP solve. Defaults to R_INIT
            r_final (float, optional): Final radial coordinate r of the IVP solve. Defaults to R_FINAL
            method (str, optional): Method used by the IVP solver. Defaults to IVP_METHOD
            max_step (float, optional): Maximum allowed step size for the IVP solver. Defaults to MAX_STEP
            atol (float, optional): Absolute tolerance of the IVP solver. Defaults to ATOL
            rtol (float, optional): Relative tolerance of the IVP solver. Defaults to RTOL

        Raises:
            Exception: Exception in case the IVP fails to solve the equation
            Exception: Exception in case the IVP fails to find the ODE termination event
            Exception: Exception in case the initial radial coordinate is too large
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
        r_max_p = (np.abs(p_c / p_2) * rtol)**(1 / 2)
        r_max_rho = (np.abs(rho_c / rho_2) * rtol)**(1 / 2)
        r_init_max = min(r_max_p, r_max_rho)
        if r_init > r_init_max:
            raise Exception(f"The initial radial coordinate is too large: (r_init = {r_init}) > (r_init_max = {r_init_max})")

        # Calculate the initial values, given by the series solution near r = 0
        r = r_init
        self.p_init = p_c + p_2 * r**2
        self.m_init = m_3 * r**3 + m_5 * r**5

        # Solve the ODE system
        ode_solution = solve_ivp(
            self._ode_system,
            (r_init, r_final),
            (self.p_init, self.m_init, self.nu_init),
            method,
            events=(self._ode_termination_event),
            max_step=max_step,
            atol=atol,
            rtol=rtol)
        self.r_ode_solution = ode_solution.t
        (self.p_ode_solution, self.m_ode_solution, self.nu_ode_solution) = ode_solution.y
        self.rho_ode_solution = self.eos.rho(self.p_ode_solution)

        # Check the ODE solution status and treat each case
        if ode_solution.status == -1:
            raise Exception(ode_solution.message)
        elif ode_solution.status != 1:
            raise Exception("The solver did not find the ODE termination event")
        # Get the star radius, star mass, and surface nu from the ODE termination event
        self.star_radius = ode_solution.t_events[0][0]
        self.star_mass = ode_solution.y_events[0][0][1]
        nu_surface = ode_solution.y_events[0][0][2]

        # Adjust metric function with the correct boundary condition (nu(R) = ln(1 - 2M/R))
        self.nu_ode_solution += - nu_surface + np.log(1 - 2 * self.star_mass / self.star_radius)

        # Create interpolated functions for the solution using CubicSpline
        self.p_spline_function = CubicSpline(self.r_ode_solution, self.p_ode_solution, extrapolate=False)
        self.m_spline_function = CubicSpline(self.r_ode_solution, self.m_ode_solution, extrapolate=False)
        self.nu_spline_function = CubicSpline(self.r_ode_solution, self.nu_ode_solution, extrapolate=False)
        self.rho_spline_function = CubicSpline(self.r_ode_solution, self.rho_ode_solution, extrapolate=False)

    def plot_star_structure_curves(self, figure_path="figures/star_structure"):
        """Method that prints the star radius and mass and plots the solution found

        Args:
            figure_path (str, optional): Path used to save the figure. Defaults to "figures/star_structure"
        """

        # Print the star radius and mass
        print(f"Star radius = {(self.star_radius / 10**3):e} [km]")
        print(f"Star mass = {(self.star_mass * uconv.MASS_GU_TO_SOLAR_MASS):e} [solar mass]")

        # Show a simple plot of the solution
        plt.figure()
        r_ode_solution_km = self.r_ode_solution / 10**3
        plt.plot(r_ode_solution_km, self.p_ode_solution * 1e-36 * uconv.PRESSURE_GU_TO_CGS, linewidth=1, label="$p ~ [10^{36} ~ dyn \\cdot cm^{-2}]$")
        plt.plot(r_ode_solution_km, self.m_ode_solution * uconv.MASS_GU_TO_SOLAR_MASS, linewidth=1, label="$m ~ [M_{\\odot}]$")
        plt.plot(r_ode_solution_km, self.nu_ode_solution, linewidth=1, label="$\\nu ~ [dimensionless]$")
        plt.plot(r_ode_solution_km, self.rho_ode_solution * 1e-15 * uconv.MASS_DENSITY_GU_TO_CGS, linewidth=1, label="$\\rho ~ [10^{15} ~ g \\cdot cm^{-3}]$")
        plt.title(f"TOV solution for the {self.eos.eos_name.replace('EOS', ' EOS')} star", y=1.05)
        plt.xlabel("$r ~ [km]$")
        plt.legend()

        # Create the folder if necessary and save the figure
        os.makedirs(figure_path, exist_ok=True)
        figure_name = "star_structure_graph.png"
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
    star_object = Star(eos, p_center, p_surface)

    # Solve the TOV equation
    star_object.solve_tov(max_step=100.0)

    # Plot the star structure curves
    star_object.plot_star_structure_curves()
