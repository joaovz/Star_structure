import numpy as np


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
