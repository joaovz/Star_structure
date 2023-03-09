import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

class Star:

    def __init__(self, rho_eos, p_c):

        # Set the density function as the EOS given (rho(p))
        self.rho = rho_eos

        # Set the integration constants: pressure, mass, and density at r=0
        self.p_0 = p_c
        self.m_0 = 0
        self.rho_0 = self.rho(self.p_0)

        # Initialize star properties: radius and total mass
        self.star_radius = 0
        self.star_mass = 0

    def _ode_system(self, r, y):

        # ODE System that describes the interior structure of the star
        p = y[0]
        m = y[1]
        dp_dr = -((self.rho(p) + p)*(m + 4*np.pi*r**3*p))/(r*(r - 2*m))         # TOV equation
        dm_dr = 4*np.pi*r**2*self.rho(p)                                        # Rate of change of the mass function
        return [dp_dr, dm_dr]

    def solve_tov(self, r_begin, r_end, r_nsamples, method='RK45'):

        # Solve the ODE system and calculate the density for the ODE solution
        ode_solution = solve_ivp(self._ode_system, [r_begin, r_end], [self.p_0, self.m_0], method=method)
        rho_ode_solution = self.rho(ode_solution.y[0])

        # Create interpolated functions for the solution using CubicSpline
        self.p_spline_function = CubicSpline(ode_solution.t, ode_solution.y[0])
        self.m_spline_function = CubicSpline(ode_solution.t, ode_solution.y[1])
        self.rho_spline_function = CubicSpline(ode_solution.t, rho_ode_solution)

        # Calculate the arrays for the solution according to the desired linspace
        self.r_space = np.linspace(r_begin, r_end, r_nsamples)
        self.p_num_solution = self.p_spline_function(self.r_space)
        self.m_num_solution = self.m_spline_function(self.r_space)
        self.rho_num_solution = self.rho_spline_function(self.r_space)

    def plot_result(self):

        # Show simple plot of the solution
        plt.figure()
        plt.plot(self.r_space, self.p_num_solution, linewidth=1, label='pressure')
        plt.plot(self.r_space, self.m_num_solution, linewidth=1, label='mass function')
        plt.plot(self.r_space, self.rho_num_solution, linewidth=1, label='density')
        plt.title('TOV solution for the star')
        plt.xlabel('r')
        plt.legend()
        plt.show()


if __name__ == "__main__":

    # Set the EOS and central pressure
    # def rho(p):
    #     c = 1.0
    #     return 12*(c * p)**(1/2)
    # p_c = 1*10**6

    def rho(p):
        c = 10**12
        return (p/c)**(1/2)
    p_c = 1*10**6

    # Define the object
    star_object = Star(rho, p_c)

    # Solve the TOV equation
    r_begin, r_end, r_nsamples = 1*10**(-12), 5*10**(-3), 1*10**6
    star_object.solve_tov(r_begin, r_end, r_nsamples)

    # Plot the result
    star_object.plot_result()
