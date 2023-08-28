import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from eos_library import PolytropicEOS


class Star:
    """Class with all the properties and methods necessary to describe a single star
    """

    # Defining some constants
    SOLAR_MASS = 1.48e3         # Solar mass [m]
    SOLAR_RADIUS = 6.957e8      # Solar radius [m]

    def __init__(self, rho_eos, p_center, p_surface):
        """Initialization method

        Args:
            rho_eos (function): Python function in the format rho(p) that describes the EOS of the star
            p_center (float): Center pressure of the star [m^-2]
            p_surface (float): Surface pressure of the star [m^-2]
        """

        # Set the density function as the given EOS (rho(p))
        self.rho_eos = rho_eos

        # Set the integration constants: pressure, mass, metric function, and density at r=0, at the center
        self.p_0 = p_center                     # Center pressure [m^-2]
        self.m_0 = 0.0                          # Center mass [m]
        self.nu_0 = 0.0                         # Center metric function (g_tt = -e^nu) [dimensionless]
        self.rho_0 = self.rho_eos(self.p_0)     # Center energy density [m^-2]

        # Set the boundary value for the termination of the ODE integration: pressure at r=R, on the surface
        self.p_surface = p_surface              # Surface pressure [m^-2]

        # Initialize star properties: radius and total mass
        self.star_radius = 0.0                  # Star radius [m]
        self.star_mass = 0.0                    # Star mass [m]

    def _ode_system(self, r, y):
        """Method that implements the TOV ODE system in the form ``dy/dr = f(r, y)``, used by the IVP solver

        Args:
            r (float): Independent variable of the ODE system (radial coordinate r)
            y (array of float): Array with the dependent variables of the ODE system (p, m, and nu)

        Returns:
            array of float: Right hand side of the equation ``dy/dr = f(r, y)`` ([dp_dr, dm_dr, dnu_dr])

        Raises:
            Exception: Exception in case the pressure is outside the acceptable range (p > p_0)
            Exception: Exception in case the EOS function didn't return a number
        """

        # Variables of the system
        p = y[0]
        m = y[1]

        # Check if p is outside the acceptable range, and raise an exception in that case
        if p > self.p_0:
            raise Exception(f"The pressure is outside the acceptable range (p = {p}): p > p_0")

        # Set derivatives to zero to saturate functions, as this condition indicates the end of integration
        if p <= 0.0:
            dp_dr = 0.0
            dm_dr = 0.0
            dnu_dr = 0.0
        else:
            # Calculate rho, check if it is some invalid value, and raise an exception in that case
            rho = self.rho_eos(p)
            if np.isnan(rho):
                raise Exception(f"The EOS function didn't return a number: p = {p}, rho = {rho}")

            # ODE System that describes the interior structure of the star
            dp_dr = -((rho + p) * (m + 4 * np.pi * r**3 * p)) / (r * (r - 2 * m))       # TOV equation
            dm_dr = 4 * np.pi * r**2 * rho                                              # Rate of change of the mass
            dnu_dr = -(2 / (rho + p)) * dp_dr                                           # Rate of change of the metric function

        return [dp_dr, dm_dr, dnu_dr]

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

    def solve_tov(self, p_center=None, r_begin=np.finfo(float).eps, r_end=np.inf, method='RK45', max_step=np.inf, atol=1e-9, rtol=1e-6):
        """Method that solves the TOV system for the star, finding the functions p(r), m(r), nu(r), and rho(r)

        Args:
            p_center (float, optional): Center pressure of the star [m^-2]
            r_begin (float, optional): Radial coordinate r at the beginning of the IVP solve. Defaults to np.finfo(float).eps
            r_end (float, optional): Radial coordinate r at the end of the IVP solve. Defaults to np.inf
            method (str, optional): Method used by the IVP solver. Defaults to 'RK45'
            max_step (float, optional): Maximum allowed step size for the IVP solver. Defaults to np.inf
            atol (float, optional): Absolute tolerance of the IVP solver. Defaults to 1e-9
            rtol (float, optional): Relative tolerance of the IVP solver. Defaults to 1e-6

        Raises:
            Exception: Exception in case the IVP fails to solve the equation
            Exception: Exception in case the IVP fails to find the ODE termination event
        """

        # Transfer the p_center parameter if it was passed as an argument
        if p_center is not None:
            self.p_0 = p_center

        # Solve the ODE system
        ode_solution = solve_ivp(
            self._ode_system,
            [r_begin, r_end],
            [self.p_0, self.m_0, self.nu_0],
            method,
            events=[self._ode_termination_event],
            max_step=max_step,
            atol=atol,
            rtol=rtol)
        self.r_ode_solution = ode_solution.t
        self.p_ode_solution = ode_solution.y[0]
        self.m_ode_solution = ode_solution.y[1]
        self.rho_ode_solution = self.rho_eos(ode_solution.y[0])

        # Check the ODE solution status, and treat each case
        if ode_solution.status == -1:
            raise Exception(ode_solution.message)
        elif ode_solution.status != 1:
            raise Exception("The solver did not find the ODE termination event")
        # Get the star radius and mass from the ODE termination event
        self.star_radius = ode_solution.t_events[0][0]
        self.star_mass = ode_solution.y_events[0][0][1]

        # Adjust metric function with the correct boundary condition (nu(R) = ln(1 - 2M/R))
        self.nu_ode_solution = ode_solution.y[2] - ode_solution.y_events[0][0][2] + np.log(1 - 2 * self.star_mass / self.star_radius)

        # Create interpolated functions for the solution using CubicSpline
        self.p_spline_function = CubicSpline(self.r_ode_solution, self.p_ode_solution, extrapolate=False)
        self.m_spline_function = CubicSpline(self.r_ode_solution, self.m_ode_solution, extrapolate=False)
        self.nu_spline_function = CubicSpline(self.r_ode_solution, self.nu_ode_solution, extrapolate=False)
        self.rho_spline_function = CubicSpline(self.r_ode_solution, self.rho_ode_solution, extrapolate=False)

    def plot_star_structure_curves(self):
        """Method that prints the star radius and mass and plots the solution found
        """

        # Print the star radius and mass
        print(f"Star radius = {self.star_radius / 10**3} [km]")
        print(f"Star mass = {self.star_mass / self.SOLAR_MASS} [solar mass]")

        # Show a simple plot of the solution
        plt.figure()
        plt.plot(self.r_ode_solution / 10**3, self.p_ode_solution * 10**8, linewidth=1, label="$p ~ [10^{-8} m^{-2}]$")
        plt.plot(self.r_ode_solution / 10**3, self.m_ode_solution / self.SOLAR_MASS, linewidth=1, label="$m ~ [M_{\\odot}]$")
        plt.plot(self.r_ode_solution / 10**3, self.nu_ode_solution, linewidth=1, label="$\\nu ~ [dimensionless]$")
        plt.plot(self.r_ode_solution / 10**3, self.rho_ode_solution * 10**8, linewidth=1, label="$\\rho ~ [10^{-8} m^{-2}]$")
        plt.title("TOV solution for the star")
        plt.xlabel("$r ~ [km]$")
        plt.legend()
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

    # Define the object
    star_object = Star(eos.rho, p_center, p_surface)

    # Solve the TOV equation
    star_object.solve_tov(max_step=100.0)

    # Plot the star structure curves
    star_object.plot_star_structure_curves()
