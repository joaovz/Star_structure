import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


class Star:
    """Class with all the properties and methods necessary to describe a single star
    """

    # Defining some constants
    SOLAR_MASS = 1.48e3             # Solar mass [m]
    SOLAR_RADIUS = 6.957e8          # Solar radius [m]

    def __init__(self, rho_eos, p_center, p_surface):
        """Initialization method

        Args:
            rho_eos (function): Python function in the format rho(p) that describes the EOS of the star
            p_center (float): Center pressure of the star [m^-2]
            p_surface (float): Surface pressure of the star [m^-2]
        """

        # Set the density function as the EOS given (rho(p))
        self.rho = rho_eos

        # Set the integration constants: pressure, mass, and density at r=0, at the center
        self.p_0 = p_center                     # Center pressure [m^-2]
        self.m_0 = 0.0                          # Center mass [m]
        self.rho_0 = self.rho(self.p_0)         # Center energy density [m^-2]

        # Set the boundary value for the termination of the ODE integration: pressure at r=R, on the surface
        self.p_surface = p_surface              # Surface pressure [m^-2]

        # Initialize star properties: radius and total mass
        self.star_radius = 0.0                  # Star radius [m]
        self.star_mass = 0.0                    # Star mass [m]

    def _ode_system(self, r, y):
        """Method that implements the TOV ODE system in the form ``dy/dr = f(r, y)``, used by the IVP solver

        Args:
            r (float): Independent variable of the ODE system (radial coordinate r)
            y (array of float): Array with the dependent variables of the ODE system (p and m)

        Returns:
            array of float: Right hand side of the equation ``dy/dr = f(r, y)`` ([dp_dr, dm_dr])
        """

        # ODE System that describes the interior structure of the star
        p = y[0]
        m = y[1]
        rho = self.rho(p)
        dp_dr = -((rho + p) * (m + 4 * np.pi * r**3 * p)) / (r * (r - 2 * m))           # TOV equation
        dm_dr = 4 * np.pi * r**2 * rho                                                  # Rate of change of the mass function
        return [dp_dr, dm_dr]

    def _ode_termination_event(self, r, y):
        """Event method used by the IVP solver. The solver will find an accurate value of r at which
        ``event(r, y(r)) = 0`` using a root-finding algorithm

        Args:
            r (float): Independent variable of the ODE system (radial coordinate r)
            y (array of float): Array with the dependent variables of the ODE system (p and m)

        Returns:
            float: ``p - p_surface``
        """

        return y[0] - self.p_surface                # Condition of the event (event happens when condition == 0 ==> when p == p_surface)
    _ode_termination_event.terminal = True          # Set the event as a terminal event, terminating the integration of the ODE

    def solve_tov(self, r_begin=np.finfo(float).eps, r_end=np.inf, r_nsamples=10**6, method='RK45'):
        """Method that solves the TOV system for the star, finding the functions p(r), m(r), and rho(r)

        Args:
            r_begin (float, optional): Radial coordinate r at the beginning of the IVP solve. Defaults to np.finfo(float).eps
            r_end (float, optional): Radial coordinate r at the end of the IVP solve. Defaults to np.inf
            r_nsamples (int, optional): Number of samples used to create the r_space array. Defaults to 10**6
            method (str, optional): Method used by the IVP solver. Defaults to 'RK45'

        Raises:
            Exception: Exception in case the IVP fails to solve the equation
            Exception: Exception in case the IVP fails to find the ODE termination event
        """

        # Solve the ODE system
        ode_solution = solve_ivp(self._ode_system, [r_begin, r_end], [self.p_0, self.m_0], method=method, events=[self._ode_termination_event])
        r_ode_solution = ode_solution.t
        p_ode_solution = ode_solution.y[0]
        m_ode_solution = ode_solution.y[1]
        rho_ode_solution = self.rho(ode_solution.y[0])

        # Check the ODE solution status, and treat each case
        if ode_solution.status == -1:
            raise Exception(ode_solution.message)
        elif ode_solution.status == 0:
            raise Exception("The solver did not find the ODE termination event")
        elif ode_solution.status == 1:
            # Get the star radius and mass from the ODE termination event
            self.star_radius = ode_solution.t_events[0][0]
            self.star_mass = ode_solution.y_events[0][0][1]

        # Create interpolated functions for the solution using CubicSpline
        self.p_spline_function = CubicSpline(r_ode_solution, p_ode_solution)
        self.m_spline_function = CubicSpline(r_ode_solution, m_ode_solution)
        self.rho_spline_function = CubicSpline(r_ode_solution, rho_ode_solution)

        # Calculate the arrays for the solution according to the desired linspace
        self.r_space = np.linspace(r_begin, self.star_radius, r_nsamples)
        self.p_num_solution = self.p_spline_function(self.r_space)
        self.m_num_solution = self.m_spline_function(self.r_space)
        self.rho_num_solution = self.rho_spline_function(self.r_space)

    def show_result(self):
        """Method that print the star radius and mass and plot the solution found
        """

        # Print the star radius and mass
        print(f"Star radius = {self.star_radius/self.SOLAR_RADIUS} [solar radius]")
        print(f"Star mass = {self.star_mass/self.SOLAR_MASS} [solar mass]")

        # Show a simple plot of the solution
        plt.figure()
        plt.plot(self.r_space, self.p_num_solution * 10**8, linewidth=1, label="pressure [10^8 m^-2]")
        plt.plot(self.r_space, self.m_num_solution / self.SOLAR_MASS, linewidth=1, label="mass function [solar mass]")
        plt.plot(self.r_space, self.rho_num_solution * 10**8, linewidth=1, label="density [10^8 m^-2]")
        plt.title("TOV solution for the star")
        plt.xlabel("r [m]")
        plt.legend()
        plt.show()


# This logic is a simple example, only executed when this file is run directly in the command prompt
if __name__ == "__main__":

    # Set the EOS and pressure at the center and surface of the star
    def rho(p):
        c = 1.0e8           # [m^2]
        return (np.abs(p / c))**(1 / 2)

    def p(rho):
        c = 1.0e8           # [m^2]
        return c * rho**2

    rho_center = 2.376364e-9            # Center density in GU [m^-2]
    p_center = rho(rho_center)          # Center pressure in GU [m^-2]
    p_surface = 0.0

    # Define the object
    star_object = Star(rho, p_center, p_surface)

    # Solve the TOV equation
    star_object.solve_tov()

    # Show the result
    star_object.show_result()
