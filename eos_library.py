import numpy as np
from scipy.interpolate import CubicSpline
from data_handling import *


class PolytropicEOS:
    """Class with the functions of the Polytropic EOS given by p = k * rho ** (1 + 1 / n)
    """

    def __init__(self, k, n):
        """Initialization method

        Args:
            k (float): Constant of proportionality of the EOS [m^2]
            n (float): Polytropic index [dimensionless]
        """
        self.k = float(k)
        self.m = 1.0 + 1.0 / float(n)

    def rho(self, p):
        """Function of the density in terms of the pressure

        Args:
            p (float): Pressure [m^-2]

        Returns:
            float: Density [m^-2]
        """
        return np.abs(p / self.k)**(1 / self.m)

    def p(self, rho):
        """Function of the pressure in terms of the density

        Args:
            rho (float): Density [m^-2]

        Returns:
            float: Pressure [m^-2]
        """
        return self.k * rho**self.m


class TableEOS:
    """Class with the functions of the EOS given by a table in a file, with density and pressure columns in CGS
    """

    def __init__(self, fname):
        """Initialization method

        Args:
            fname (string): File name of the table with the EOS (density and pressure columns)
        """

        # Open the .dat file with the EOS
        rho, p = dat_to_array(
            fname=fname,
            usecols=(0, 1),
            unit_conversion=(DENSITY_CGS_TO_GU, PRESSURE_CGS_TO_GU))

        # Convert the EOS to spline functions
        self.rho_spline_function = CubicSpline(p, rho, extrapolate=False)
        self.p_spline_function = CubicSpline(rho, p, extrapolate=False)

        # Save the center density and pressure
        self.rho_center = rho[-1]
        self.p_center = p[-1]

    def rho(self, p):
        """Function of the density in terms of the pressure

        Args:
            p (float): Pressure [m^-2]

        Returns:
            float: Density [m^-2]
        """
        return self.rho_spline_function(p)

    def p(self, rho):
        """Function of the pressure in terms of the density

        Args:
            rho (float): Density [m^-2]

        Returns:
            float: Pressure [m^-2]
        """
        return self.p_spline_function(rho)


# This logic is a simple example, only executed when this file is run directly in the command prompt
if __name__ == "__main__":

    ## Polytropic EOS test

    # Create the EOS object
    polytropic_eos = PolytropicEOS(k=1.32e5, n=2.25)

    # Calculate the center density and pressure
    rho_center = 2.376364e-9        # Center density [m^-2]

    p_center_calc = polytropic_eos.p(rho_center)
    rho_center_calc = polytropic_eos.rho(p_center_calc)

    # Print the result to check equations
    print(f"rho_center = {rho_center}")
    print(f"rho_center_calc = {rho_center_calc}")


    ## Table EOS test

    # Create the EOS object
    table_eos = TableEOS(fname='data/EOSFull_GM1_BPS.dat')

    # Calculate the center density and pressure
    rho_center = table_eos.rho_center       # Center density [m^-2]

    p_center_calc = table_eos.p(rho_center)
    rho_center_calc = table_eos.rho(p_center_calc)

    # Print the result to check equations
    print(f"rho_center = {rho_center}")
    print(f"rho_center_calc = {rho_center_calc}")
