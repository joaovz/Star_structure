import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from data_handling import *


class EOS:
    """Base class for the EOS classes
    """

    def __init__(self):
        """Initialization method
        """
        self.eos_name = self.__class__.__name__

    def _config_plot(self):
        """Method that configures the plotting
        """

        # Create a dictionary with all the functions used in plotting, with each name and label description
        self.plot_dict = {
            "rho": {
                "name": "Density",
                "label": "$\\rho ~ [m^{-2}]$",
                "value": self.rho_space,
            },
            "p": {
                "name": "Pressure",
                "label": "$p ~ [m^{-2}]$",
                "value": self.p_space,
            },
            "drho_dp": {
                "name": "Density derivative",
                "label": "$\\partial_{p}{\\rho} ~ [dimensionless]$",
                "value": self.drho_dp(self.p_space),
            },
            "dp_drho": {
                "name": "Pressure derivative",
                "label": "$\\partial_{\\rho}{p} ~ [dimensionless]$",
                "value": self.dp_drho(self.rho_space),
            },
            "gamma": {
                "name": "Adiabatic index",
                "label": "$\\Gamma ~ [dimensionless]$",
                "value": self.gamma(self.p_space),
            },
            "c_s": {
                "name": "Speed of sound",
                "label": "$c_s ~ [dimensionless]$",
                "value": self.c_s(self.rho_space),
            },
        }

        # Create a list with all the curves to be plotted
        self.curves_list = [
            ['p', 'rho'],
            ['p', 'drho_dp'],
            ['p', 'dp_drho'],
            ['p', 'gamma'],
            ['p', 'c_s'],
            ['rho', 'p'],
            ['rho', 'drho_dp'],
            ['rho', 'dp_drho'],
            ['rho', 'gamma'],
            ['rho', 'c_s'],
        ]

    def rho(self, p):
        """Function of the density in terms of the pressure

        Args:
            p (float): Pressure [m^-2]

        Returns:
            float: Density [m^-2]
        """
        return p

    def p(self, rho):
        """Function of the pressure in terms of the density

        Args:
            rho (float): Density [m^-2]

        Returns:
            float: Pressure [m^-2]
        """
        return rho

    def drho_dp(self, p):
        """Derivative of the density with respect to the pressure

        Args:
            p (float): Pressure [m^-2]

        Returns:
            float: drho_dp(p) [dimensionless]
        """
        return 1.0

    def dp_drho(self, rho):
        """Derivative of the pressure with respect to the density

        Args:
            rho (float): Density [m^-2]

        Returns:
            float: dp_drho(rho) [dimensionless]
        """
        return 1.0

    def gamma(self, p):
        """Adiabatic index Gamma in terms of the pressure

        Args:
            p (float): Pressure [m^-2]

        Returns:
            float: Gamma(p) [dimensionless]
        """

        rho = self.rho(p)
        gamma = ((rho + p) / p) * self.dp_drho(rho)

        return gamma

    def c_s(self, rho):
        """Speed of sound in terms of the density

        Args:
            rho (float): Density [m^-2]

        Returns:
            float: c_s(rho) [dimensionless]
        """
        return np.sqrt(self.dp_drho(rho))

    def check_eos(self, p_space, rtol=1e-6):
        """Check if EOS implementation is correct

        Args:
            p_space (array of float): Array that defines the pressure interval [m^-2]
            rtol (float, optional): Relative tolerance for the error. Defaults to 1e-6
        """

        # Calculate rho_space and p_space using the EOS functions
        rho_space = self.rho(p_space)
        p_space_calc = self.p(rho_space)
        eos_rel_error = np.abs((p_space_calc - p_space) / p_space)

        # Show warning if error in p_space_calc is greater than maximum acceptable error
        if max(eos_rel_error) > rtol:
            print(f"Warning: Error in {self.eos_name} calculation is larger than the acceptable error: {max(eos_rel_error)} > {rtol}")

        # Calculate drho_dp, dp_drho, and the error
        drho_dp = self.drho_dp(p_space)
        dp_drho = self.dp_drho(rho_space)
        derivative_rel_error = np.abs((drho_dp**(-1) - dp_drho) / dp_drho)

        # Show warning if derivative error is greater than maximum acceptable error
        if max(derivative_rel_error) > rtol:
            print(f"Warning: Error in {self.eos_name} derivatives calculation is larger than the acceptable error: {max(derivative_rel_error)} > {rtol}")

    def plot_curve(self, x_axis="p", y_axis="rho", figure_path="figures/eos_library"):
        """Method that plots some curve of the EOS

        Args:
            x_axis (str, optional): Key of self.plot_dict to indicate the x_axis used. Defaults to "p"
            y_axis (str, optional): Key of self.plot_dict to indicate the y_axis used. Defaults to "rho"
            figure_path (str, optional): Path used to save the figure. Defaults to "figures/eos_library"
        """

        # Create a simple plot
        plt.figure()
        plt.plot(self.plot_dict[x_axis]['value'], self.plot_dict[y_axis]['value'], linewidth=1)
        plt.title(f"{self.plot_dict[y_axis]['name']} vs {self.plot_dict[x_axis]['name']} curve of the {self.eos_name.replace('EOS', ' EOS')}")
        plt.xlabel(self.plot_dict[x_axis]['label'])
        plt.ylabel(self.plot_dict[y_axis]['label'])

        # Create the folder if necessary and save the figure
        complete_figure_path = f"{figure_path}/{self.eos_name.lower().replace('eos', '_eos')}"
        if not os.path.exists(complete_figure_path):
            os.makedirs(complete_figure_path)
        x_axis_name = self.plot_dict[x_axis]['name'].lower().replace(' ', '_')
        y_axis_name = self.plot_dict[y_axis]['name'].lower().replace(' ', '_')
        plt.savefig(f"{complete_figure_path}/{y_axis_name}_vs_{x_axis_name}_curve.png")

    def plot_all_curves(self, p_space, figures_path="figures/eos_library"):
        """Method that plots all curves specified by the self.curves_list

        Args:
            p_space (array of float): Array that defines the pressure interval [m^-2]
            figures_path (str, optional): Path used to save the figures. Defaults to "figures/eos_library"
        """

        # Set p_space and rho_space, and configure plot
        self.p_space = p_space
        self.rho_space = self.rho(p_space)
        self._config_plot()

        # Plot all curves
        for axis in self.curves_list:
            self.plot_curve(axis[0], axis[1], figures_path)
        plt.show()


class PolytropicEOS(EOS):
    """Class with the functions of the Polytropic EOS given by p = k * rho ** (1 + 1 / n)
    """

    def __init__(self, k, n):
        """Initialization method

        Args:
            k (float): Constant of proportionality of the EOS [m^2]
            n (float): Polytropic index [dimensionless]
        """

        # Execute parent class' __init__ method
        super().__init__()

        # Set the parameters
        self.k = float(k)
        self.m = 1.0 + 1.0 / float(n)

    def rho(self, p):
        return np.abs(p / self.k)**(1 / self.m)

    def p(self, rho):
        return self.k * rho**self.m

    def drho_dp(self, p):
        return (1 / (self.k * self.m)) * np.abs(p / self.k)**((1 / self.m) - 1)

    def dp_drho(self, rho):
        return self.k * self.m * rho**(self.m - 1)


class TableEOS(EOS):
    """Class with the functions of the EOS given by a table in a file, with density and pressure columns in CGS
    """

    def __init__(self, fname, eos_name="TableEOS"):
        """Initialization method

        Args:
            fname (string): File name of the table with the EOS (density and pressure columns)
            eos_name (string, optional): Name of the EOS. Defaults to "TableEOS"
        """

        # Execute parent class' __init__ method
        super().__init__()

        # Set the EOS name
        self.eos_name = eos_name

        # Open the .csv file with the EOS
        rho, p = csv_to_arrays(
            fname=fname,
            unit_conversion=(MASS_DENSITY_CGS_TO_GU, PRESSURE_CGS_TO_GU))

        # Convert the EOS to spline functions
        self.rho_spline_function = CubicSpline(p, rho, extrapolate=False)
        self.p_spline_function = CubicSpline(rho, p, extrapolate=False)

        # Calculate the derivatives
        self.drho_dp_spline_function = self.rho_spline_function.derivative()
        self.dp_drho_spline_function = self.p_spline_function.derivative()

        # Save the center and surface density and pressure
        self.rho_center = rho[-1]
        self.p_center = p[-1]
        self.rho_surface = rho[0]
        self.p_surface = p[0]

    def rho(self, p):
        return self.rho_spline_function(p)

    def p(self, rho):
        return self.p_spline_function(rho)

    def drho_dp(self, p):
        return self.drho_dp_spline_function(p)

    def dp_drho(self, rho):
        return self.dp_drho_spline_function(rho)


class QuarkEOS(EOS):
    """Class with the functions of the Quark EOS, defined by the grand thermodynamic potential given by:
    Omega = (3 / (4 * pi**2)) * (-a4 * mu**4 + a2 * mu**2) + B
    """

    def __init__(self, B, a2, a4):
        """Initialization method

        Args:
            B (float): Model free parameter [m^-2]
            a2 (float): Model free parameter [m^-1]
            a4 (float): Model free parameter [dimensionless]
        """

        # Execute parent class' __init__ method
        super().__init__()

        # Set the parameters
        self.B = B
        self.a2 = a2
        self.a4 = a4

    def rho(self, p):

        B = self.B
        a2 = self.a2
        a4 = self.a4

        rho = (
            3 * p + 4 * B + ((3 * a2**2) / (4 * np.pi**2 * a4)) * (
                1 + (1 + ((16 * np.pi**2 * a4) / (3 * a2**2)) * (p + B))**(1 / 2)
            )
        )

        return rho

    def p(self, rho):

        B = self.B
        a2 = self.a2
        a4 = self.a4

        p = (
            (1 / 3) * (rho - 4 * B) - (a2**2 / (12 * np.pi**2 * a4)) * (
                1 + (1 + ((16 * np.pi**2 * a4) / a2**2) * (rho - B))**(1 / 2)
            )
        )

        return p

    def drho_dp(self, p):

        B = self.B
        a2 = self.a2
        a4 = self.a4

        drho_dp = (
            3 + 2 * (1 + ((16 * np.pi**2 * a4) / (3 * a2**2)) * (p + B))**(-1 / 2)
        )

        return drho_dp

    def dp_drho(self, rho):

        B = self.B
        a2 = self.a2
        a4 = self.a4

        dp_drho = (
            (1 / 3) - (2 / 3) * (1 + ((16 * np.pi**2 * a4) / a2**2) * (rho - B))**(-1 / 2)
        )

        return dp_drho


class BSk20EOS(EOS):
    """Class with the functions of the BSk20 EOS
    """

    def __init__(self, rho_space):
        """Initialization method

        Args:
            rho_space (array of float): Array that defines the density interval [m^-2]
        """

        # Execute parent class' __init__ method
        super().__init__()

        # Calculate the p_space from the analytic expression
        p_space = self._p_analytic(rho_space)

        # Convert the EOS to spline functions
        self.rho_spline_function = CubicSpline(p_space, rho_space, extrapolate=False)
        self.p_spline_function = CubicSpline(rho_space, p_space, extrapolate=False)

        # Calculate the derivatives
        self.drho_dp_spline_function = self.rho_spline_function.derivative()
        self.dp_drho_spline_function = self.p_spline_function.derivative()

        # Save the center and surface density and pressure
        self.rho_center = rho_space[-1]
        self.p_center = p_space[-1]
        self.rho_surface = rho_space[0]
        self.p_surface = p_space[0]

    def _p_analytic(self, rho):
        """Analytic expression of the pressure

        Args:
            rho (float): Density [m^-2]

        Returns:
            float: Pressure [m^-2]
        """

        # Set the a_i parameters
        a = (
            0.0, 4.078, 7.587, 0.00839, 0.21695, 3.614, 11.942, 13.751, 1.3373, 3.606, -22.996, 1.6229,
            4.88, 14.274, 23.560, -1.5564, 2.095, 15.294, 0.084, 6.36, 11.67, -0.042, 14.8, 14.18
        )

        # Calculating xi
        xi = np.log10(rho / MASS_DENSITY_CGS_TO_GU)

        # Calculating zeta
        zeta = (
            ((a[1] + a[2] * xi + a[3] * xi**3) / (1 + a[4] * xi)) * (np.exp(a[5] * (xi - a[6])) + 1)**(-1)
            + (a[7] + a[8] * xi) * (np.exp(a[9] * (a[6] - xi)) + 1)**(-1)
            + (a[10] + a[11] * xi) * (np.exp(a[12] * (a[13] - xi)) + 1)**(-1)
            + (a[14] + a[15] * xi) * (np.exp(a[16] * (a[17] - xi)) + 1)**(-1)
            + a[18] / (1 + (a[19] * (xi - a[20]))**2)
            + a[21] / (1 + (a[22] * (xi - a[23]))**2)
        )

        # Calculating p
        p = 10**(zeta) * PRESSURE_CGS_TO_GU

        return p

    def rho(self, p):
        return self.rho_spline_function(p)

    def p(self, rho):
        return self.p_spline_function(rho)

    def drho_dp(self, p):
        return self.drho_dp_spline_function(p)

    def dp_drho(self, rho):
        return self.dp_drho_spline_function(rho)


# This logic is a simple example, only executed when this file is run directly in the command prompt
if __name__ == "__main__":

    # BSk20 EOS test

    # Set the rho_space
    max_rho = 4.3e-9        # Maximum density [m^-2]
    rho_space = max_rho * np.logspace(-5.0, 0.0, 1000)

    # Create the EOS object
    bsk20_eos = BSk20EOS(rho_space)

    # Set the p_space
    p_space = bsk20_eos.p(rho_space)

    # Check the EOS
    bsk20_eos.check_eos(p_space)

    # Create the EOS graphs
    bsk20_eos.plot_all_curves(p_space)

    # Polytropic EOS test

    # Create the EOS object
    polytropic_eos = PolytropicEOS(k=1.0e8, n=1)

    # Set the p_space
    max_rho = 4.3e-9                        # Maximum density [m^-2]
    max_p = polytropic_eos.p(max_rho)       # Maximum pressure [m^-2]
    p_space = max_p * np.logspace(-5.0, 0.0, 1000)

    # Check the EOS
    polytropic_eos.check_eos(p_space)

    # Create the EOS graphs
    polytropic_eos.plot_all_curves(p_space)

    # Quark EOS test

    # Create the EOS object (values chosen to build a strange star)
    B = 130**4 * ENERGY_DENSITY_NU_TO_GU
    a2 = (100**4 * ENERGY_DENSITY_NU_TO_GU)**(1 / 2)
    a4 = 0.6
    quark_eos = QuarkEOS(B, a2, a4)

    # Set the p_space
    max_rho = 1.2e-9                    # Maximum density [m^-2]
    max_p = quark_eos.p(max_rho)        # Maximum pressure [m^-2]
    p_space = max_p * np.logspace(-5.0, 0.0, 1000)

    # Check the EOS
    quark_eos.check_eos(p_space)

    # Create the EOS graphs
    quark_eos.plot_all_curves(p_space)

    # Table GM1 EOS test

    # Create the EOS object
    table_gm1_eos = TableEOS(fname='data/GM1.csv', eos_name='GM1EOS')

    # Set the p_space
    max_rho = 1.5e-9                        # Maximum density [m^-2]
    max_p = table_gm1_eos.p(max_rho)        # Maximum pressure [m^-2]
    p_space = max_p * np.logspace(-5.0, 0.0, 1000)

    # Check the EOS
    table_gm1_eos.check_eos(p_space)

    # Create the EOS graphs
    table_gm1_eos.plot_all_curves(p_space)

    # Table SLy4 EOS test

    # Create the EOS object
    table_sly4_eos = TableEOS(fname='data/SLy4.csv', eos_name='SLy4EOS')

    # Set the p_space
    max_rho = 2.2e-9                        # Maximum density [m^-2]
    max_p = table_sly4_eos.p(max_rho)       # Maximum pressure [m^-2]
    p_space = max_p * np.logspace(-5.0, 0.0, 1000)

    # Check the EOS
    table_sly4_eos.check_eos(p_space)

    # Create the EOS graphs
    table_sly4_eos.plot_all_curves(p_space)
