import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from constants import DefaultValues as dval
from constants import UnitConversion as uconv
from data_handling import csv_to_arrays


class EOS:
    """Base class for the EOS classes
    """

    # Class constants
    FIGURES_PATH = "figures/eos_library"

    def __init__(self):
        """Initialization method
        """

        # Initialize EOS properties
        self.eos_name = self.__class__.__name__     # EOS name, given by the class name
        self.maximum_stable_rho = None              # Maximum stable density [m^-2]
        self.p_trans = None                         # Phase transition pressure [m^-2]. It is default None to characterize no transition

    def _check_stability(self, p_space):
        """Method that checks the stability criterion for the EOS (c_s < 1)

        Args:
            p_space (array of float): Array that defines the pressure interval [m^-2]
        """

        # Calculate the speed of sound minus 1 to get the root, where the EOS becomes superluminal
        p_space_end = p_space[(p_space.size // 2):]
        rho_space = self.rho(p_space_end)
        cs_minus_1 = self.c_s(rho_space) - 1
        cs_minus_1_spline = CubicSpline(rho_space, cs_minus_1, extrapolate=False)

        # Find the roots of (c_s - 1)
        cs_minus_1_roots = cs_minus_1_spline.roots()
        if cs_minus_1_roots.size > 0:
            self.maximum_stable_rho = cs_minus_1_roots[0]
            print(f"{self.eos_name} maximum stable rho = {(self.maximum_stable_rho * uconv.MASS_DENSITY_GU_TO_CGS):e} [g cm^-3]")

    def _config_plot(self):
        """Method that configures the plotting
        """

        # Create a dictionary with all the functions used in plotting, with each name and label description
        with np.errstate(divide='ignore'):      # Ignore warnings due to log scales
            self.plot_dict = {
                "rho": {
                    "name": "Density",
                    "label": "$\\rho ~ [g ~ cm^{-3}]$",
                    "value": self.rho_space * uconv.MASS_DENSITY_GU_TO_CGS,
                },
                "p": {
                    "name": "Pressure",
                    "label": "$\\log_{10} \\left( p ~ [dyn ~ cm^{-2}] \\right)$",
                    "value": np.log10(self.p_space * uconv.PRESSURE_GU_TO_CGS),
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
                    "label": "$\\log_{10} \\left( \\Gamma ~ [dimensionless] \\right)$",
                    "value": np.log10(self.gamma(self.p_space)),
                },
                "c_s": {
                    "name": "Speed of sound",
                    "label": "$c_s ~ [dimensionless]$",
                    "value": self.c_s(self.rho_space),
                },
            }

        # Create a list with all the curves to be plotted
        self.curves_list = [
            ["rho", "p"],
            ["rho", "dp_drho"],
            ["rho", "gamma"],
            ["rho", "c_s"],
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

    def check_eos(self, p_space, rtol=dval.RTOL):
        """Check if EOS implementation is correct

        Args:
            p_space (array of float): Array that defines the pressure interval [m^-2]
            rtol (float, optional): Relative tolerance for the error. Defaults to RTOL
        """

        # Check the EOS stability
        self._check_stability(p_space)

        # Calculate rho_space and p_space using the EOS functions
        rho_space = self.rho(p_space)
        p_space_calc = self.p(rho_space)
        rho_space_calc = self.rho(p_space_calc)
        eos_rel_error = np.abs((rho_space_calc - rho_space) / rho_space)

        # Show warning if error in p_space_calc is greater than maximum acceptable error
        if max(eos_rel_error) > rtol:
            print(f"Warning: Error in {self.eos_name} calculation is larger than the acceptable error: {(max(eos_rel_error)):e} > {rtol:e}")

        # Calculate drho_dp, dp_drho, and the error
        drho_dp = self.drho_dp(p_space)
        dp_drho = self.dp_drho(rho_space)
        derivative_rel_error = np.abs((drho_dp**(-1) - dp_drho) / dp_drho)

        # Show warning if derivative error is greater than maximum acceptable error
        if max(derivative_rel_error) > rtol:
            print(f"Warning: Error in {self.eos_name} derivatives calculation is larger than the acceptable error: {(max(derivative_rel_error)):e} > {rtol:e}")

    def plot_curve(self, x_axis="p", y_axis="rho", figure_path=FIGURES_PATH):
        """Method that plots some curve of the EOS

        Args:
            x_axis (str, optional): Key of self.plot_dict to indicate the x_axis used. Defaults to "p"
            y_axis (str, optional): Key of self.plot_dict to indicate the y_axis used. Defaults to "rho"
            figure_path (str, optional): Path used to save the figure. Defaults to FIGURES_PATH
        """

        # Create a simple plot
        plt.figure(figsize=(6.0, 4.5))
        plt.plot(self.plot_dict[x_axis]["value"], self.plot_dict[y_axis]["value"], linewidth=1)
        plt.xlabel(self.plot_dict[x_axis]["label"], fontsize=10)
        plt.ylabel(self.plot_dict[y_axis]["label"], fontsize=10)

        # Create the folder if necessary and save the figure
        complete_figure_path = os.path.join(figure_path, self.eos_name.lower().replace("eos", "_eos"))
        os.makedirs(complete_figure_path, exist_ok=True)
        x_axis_name = self.plot_dict[x_axis]["name"].lower().replace(" ", "_")
        y_axis_name = self.plot_dict[y_axis]["name"].lower().replace(" ", "_")
        figure_name = f"{y_axis_name}_vs_{x_axis_name}_curve.pdf"
        complete_path = os.path.join(complete_figure_path, figure_name)
        plt.savefig(complete_path, bbox_inches="tight")

    def plot_all_curves(self, p_space=None, rho_space=None, figures_path=FIGURES_PATH):
        """Method that plots all curves specified by the self.curves_list

        Args:
            p_space (array of float, optional): Array that defines the pressure interval [m^-2]. Defaults to None
            rho_space (array of float, optional): Array that defines the density interval [m^-2]. Defaults to None
            figures_path (str, optional): Path used to save the figures. Defaults to FIGURES_PATH

        Raises:
            RuntimeError: Exception in case pressure and density spaces are not informed
        """

        # Set p_space and rho_space, and configure plot
        if p_space is not None:
            self.p_space = p_space
            self.rho_space = self.rho(p_space)
        elif rho_space is not None:
            self.p_space = self.p(rho_space)
            self.rho_space = rho_space
        else:
            raise RuntimeError("Pressure or density space must be informed to create the curves, but none were passed.")
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

    def __init__(self, file_name, eos_name="TableEOS"):
        """Initialization method

        Args:
            file_name (str): File name of the table with the EOS (density and pressure columns)
            eos_name (str, optional): Name of the EOS. Defaults to "TableEOS"
        """

        # Execute parent class' __init__ method
        super().__init__()

        # Set the EOS name
        self.eos_name = eos_name

        # Open the .csv file with the EOS
        (rho, p) = csv_to_arrays(
            file_name=file_name,
            unit_conversion=(uconv.MASS_DENSITY_CGS_TO_GU, uconv.PRESSURE_CGS_TO_GU))

        # Convert the EOS to spline functions
        self.rho_spline_function = CubicSpline(p, rho, extrapolate=False)
        self.p_spline_function = CubicSpline(rho, p, extrapolate=False)

        # Calculate the derivatives
        self.drho_dp_spline_function = self.rho_spline_function.derivative()
        self.dp_drho_spline_function = self.p_spline_function.derivative()

        # Save the maximum and minimum density and pressure
        self.rho_max = rho[-1]
        self.p_max = p[-1]
        self.rho_min = rho[0]
        self.p_min = p[0]

    def rho(self, p):
        return self.rho_spline_function(p)

    def p(self, rho):
        return self.p_spline_function(rho)

    def drho_dp(self, p):
        return self.drho_dp_spline_function(p)

    def dp_drho(self, rho):
        return self.dp_drho_spline_function(rho)


class InterpolatedEOS(EOS):
    """Class with the pressure of the EOS given by an interpolation function
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

        # Save the maximum and minimum density and pressure
        self.rho_max = rho_space[-1]
        self.p_max = p_space[-1]
        self.rho_min = rho_space[0]
        self.p_min = p_space[0]

    def _p_analytic(self, rho):
        """Analytic expression of the pressure

        Args:
            rho (float): Density [m^-2]

        Returns:
            float: Pressure [m^-2]
        """
        return rho

    def rho(self, p):
        return self.rho_spline_function(p)

    def p(self, rho):
        return self.p_spline_function(rho)

    def drho_dp(self, p):
        return self.drho_dp_spline_function(p)

    def dp_drho(self, rho):
        return self.dp_drho_spline_function(rho)


class QuarkEOS(EOS):
    """Class with the functions of the Quark EOS, defined by the grand thermodynamic potential density given by:
    omega = (3 / (4 * pi**2)) * (-a4 * mu**4 + a2 * mu**2) + B
    Every calculation is done internally using Natural Units (NU) with energy in MeV.
    Commom EOS functions have inputs and outputs in Geometrical Units (GU).
    """

    def __init__(self, a2, a4, B):
        """Initialization method

        Args:
            a2 (float): Model free parameter [MeV^2]
            a4 (float): Model free parameter [dimensionless]
            B (float): Model free parameter [MeV^4]
        """

        # Execute parent class' __init__ method
        super().__init__()

        # Set the parameters
        self.a2 = a2
        self.a4 = a4
        self.B = B

    def p_of_mu(self, mu):
        """Function of the pressure in terms of the chemical potential

        Args:
            mu (float): Chemical potential [MeV]

        Returns:
            float: Pressure [MeV^4]
        """

        a2 = self.a2
        a4 = self.a4
        B = self.B

        alpha = 3 / (4 * np.pi**2)
        p = alpha * a4 * mu**4 - alpha * a2 * mu**2 - B

        return p

    def mu_of_p(self, p):
        """Function of the chemical potential in terms of the pressure

        Args:
            p (float): Pressure [MeV^4]

        Returns:
            float: Chemical potential [MeV]
        """

        a2 = self.a2
        a4 = self.a4
        B = self.B

        mu = (a2 / (2 * a4))**(1 / 2) * (1 + (1 + ((16 * np.pi**2 * a4) / (3 * a2**2)) * (p + B))**(1 / 2))**(1 / 2)

        return mu

    def rho(self, p):

        p_nu = p * uconv.PRESSURE_GU_TO_NU                  # Convert to NU

        a2 = self.a2
        a4 = self.a4
        B = self.B

        rho_nu = (
            3 * p_nu + 4 * B + ((3 * a2**2) / (4 * np.pi**2 * a4)) * (
                1 + (1 + ((16 * np.pi**2 * a4) / (3 * a2**2)) * (p_nu + B))**(1 / 2)
            )
        )

        return rho_nu * uconv.ENERGY_DENSITY_NU_TO_GU       # Return result converted to GU

    def p(self, rho):

        rho_nu = rho * uconv.ENERGY_DENSITY_GU_TO_NU        # Convert to NU

        a2 = self.a2
        a4 = self.a4
        B = self.B

        p_nu = (
            (1 / 3) * (rho_nu - 4 * B) - (a2**2 / (12 * np.pi**2 * a4)) * (
                1 + (1 + ((16 * np.pi**2 * a4) / a2**2) * (rho_nu - B))**(1 / 2)
            )
        )

        return p_nu * uconv.PRESSURE_NU_TO_GU               # Return result converted to GU

    def drho_dp(self, p):

        p_nu = p * uconv.PRESSURE_GU_TO_NU                  # Convert to NU

        a2 = self.a2
        a4 = self.a4
        B = self.B

        drho_dp = (
            3 + 2 * (1 + ((16 * np.pi**2 * a4) / (3 * a2**2)) * (p_nu + B))**(-1 / 2)
        )

        return drho_dp      # Return result (dimensionless, so NU and GU are the same)

    def dp_drho(self, rho):

        rho_nu = rho * uconv.ENERGY_DENSITY_GU_TO_NU        # Convert to NU

        a2 = self.a2
        a4 = self.a4
        B = self.B

        dp_drho = (
            (1 / 3) - (2 / 3) * (1 + ((16 * np.pi**2 * a4) / a2**2) * (rho_nu - B))**(-1 / 2)
        )

        return dp_drho      # Return result (dimensionless, so NU and GU are the same)


class HybridEOS(EOS):
    """Class with the functions of the Hybrid EOS, defined by the a Quark EOS and some Hadron EOS
    """

    def __init__(self, quark_eos, hadron_eos, hadron_eos_table_file_name, hadron_eos_maximum_stable_rho_center):
        """Initialization method

        Args:
            quark_eos (EOS object): Object of the class QuarkEOS
            hadron_eos (EOS object): Object of the class EOS, or a class inherited from EOS
            hadron_eos_table_file_name (str): File name of the Hadron EOS table with (rho, p, nb) columns
            hadron_eos_maximum_stable_rho_center (float): Maximum stable central density of the Hadron EOS star family [m^-2]

        Raises:
            RuntimeError: Exception in case the solver fails to find the transition pressure
        """

        # Execute parent class' __init__ method
        super().__init__()

        # Store the input parameters
        self.quark_eos = quark_eos
        self.hadron_eos = hadron_eos
        self.hadron_eos_table_file_name = hadron_eos_table_file_name
        self.hadron_eos_maximum_stable_rho_center = hadron_eos_maximum_stable_rho_center

        # Initialize variables
        self.p_trans = None
        self.is_quark_eos = False
        self.is_hadron_eos = False
        self.is_hybrid_eos = False

        # Calculate the transition pressure
        self._calc_p_trans()

        # Save the maximum and minimum density and pressure
        if self.is_hadron_eos:
            self.rho_max = self.hadron_eos.rho_max
            self.p_max = self.hadron_eos.p_max
            self.rho_min = self.hadron_eos.rho_min
            self.p_min = self.hadron_eos.p_min
        elif self.is_quark_eos:
            self.rho_min = self.quark_eos.rho(0.0)
            self.p_min = 0.0
        else:
            self.rho_min = self.hadron_eos.rho_min
            self.p_min = self.hadron_eos.p_min

    def _calc_p_trans(self):
        """Method that calculates the pressure at the quark-hadron phase transition

        Raises:
            RuntimeError: Exception in case the solver fails to find the transition pressure
        """

        # Open the HadronEOS table file, using Natural Units (NU)
        nb_fm_3_to_si = (10**-15)**(-3)     # Conversion factor from fm^-3 to m^-3 (SI)
        nb_fm_3_to_nu = nb_fm_3_to_si * uconv.NUMBER_DENSITY_SI_TO_NU
        (rho_hadron_nu, self.p_hadron_nu, nb_hadron_nu) = csv_to_arrays(
            file_name=self.hadron_eos_table_file_name,
            usecols=(0, 1, 2),
            unit_conversion=(uconv.MASS_DENSITY_CGS_TO_GU * uconv.ENERGY_DENSITY_GU_TO_NU,
                             uconv.PRESSURE_CGS_TO_GU * uconv.PRESSURE_GU_TO_NU,
                             nb_fm_3_to_nu))

        # Calculate the HadronEOS Gibbs free energy per particle g = (rho + p) / nb
        self.g_hadron_nu = (rho_hadron_nu + self.p_hadron_nu) / nb_hadron_nu

        # Calculate the QuarkEOS Gibbs free energy per particle g = 3 mu
        self.g_quark_nu = 3 * self.quark_eos.mu_of_p(self.p_hadron_nu)

        # Create the (g_hadron_nu - g_quark_nu) vs p_hadron_nu interpolated function
        g_hadron_minus_g_quark_spline = CubicSpline(self.p_hadron_nu, self.g_hadron_nu - self.g_quark_nu, extrapolate=False)

        # Calculate the transition pressure
        g_hadron_minus_g_quark_roots = g_hadron_minus_g_quark_spline.roots()
        if g_hadron_minus_g_quark_roots.size > 0:
            self.p_trans_nu = np.max(g_hadron_minus_g_quark_roots)      # Use only the high pressure transition
            self.p_trans = self.p_trans_nu * uconv.PRESSURE_NU_TO_GU
            self.g_trans_nu = 3 * self.quark_eos.mu_of_p(self.p_trans_nu)
            self.is_hybrid_eos = True       # Indicate that this is indeed a hybrid EOS
        else:
            # Check if the EOS is a quark EOS or a hadron EOS
            if self.g_hadron_nu[0] < self.g_quark_nu[0]:
                self.is_hadron_eos = True
            else:
                self.is_quark_eos = True

        # Calculate the minimum and maximum transition densities
        if self.is_hybrid_eos is True:
            self.rho_trans_min = self.hadron_eos.rho(self.p_trans)
            self.rho_trans_max = self.quark_eos.rho(self.p_trans)

        # Set the EOS as a hadron EOS if the phase transition occurs only at a density greater than the maximum stable density
        if (self.is_hybrid_eos is True) and (self.rho_trans_min >= self.hadron_eos_maximum_stable_rho_center):
            self.is_hybrid_eos = False
            self.is_hadron_eos = True

    def plot_transition_graph(self, figure_path=EOS.FIGURES_PATH):
        """Method that plots the curves g vs p for the Quark and Hadron EOSs

        Args:
            figure_path (str, optional): Path used to save the figure. Defaults to FIGURES_PATH
        """

        # Create the graph
        plt.figure(figsize=(6.0, 4.5))
        plt.plot(self.p_hadron_nu, self.g_hadron_nu, linewidth=1, label="Hadron EOS")
        plt.plot(self.p_hadron_nu, self.g_quark_nu, linewidth=1, label="Quark EOS")
        if self.p_trans is not None:
            plt.plot(self.p_trans_nu, self.g_trans_nu, linewidth=1, marker=".", markersize=4**2, label="Transition")
        plt.xlabel("$p ~ [MeV^4]$", fontsize=10)
        plt.ylabel("$g ~ [MeV]$", fontsize=10)
        plt.legend()

        # Create the folder if necessary and save the figure
        complete_figure_path = os.path.join(figure_path, self.eos_name.lower().replace("eos", "_eos"))
        os.makedirs(complete_figure_path, exist_ok=True)
        x_axis_name = "pressure"
        y_axis_name = "gibbs_free_energy"
        figure_name = f"{y_axis_name}_vs_{x_axis_name}_curve.pdf"
        complete_path = os.path.join(complete_figure_path, figure_name)
        plt.savefig(complete_path, bbox_inches="tight")

    def rho(self, p):

        if np.ndim(p) == 0:     # Execute this logic if p is a scalar

            if self.is_hadron_eos or (self.is_hybrid_eos and (p < self.p_trans)):
                return self.hadron_eos.rho(p)           # Use the Hadron EOS if p < p_trans
            return self.quark_eos.rho(p)                # Use the Quark EOS if p >= p_trans

        else:                   # Execute this logic if p is an array

            # Check if the EOS is a Hybrid EOS
            if self.is_hybrid_eos is True:

                # Use the Hadron EOS if p < p_trans
                rho_hadron = self.hadron_eos.rho(p[p < self.p_trans])

                # Use the Quark EOS if p >= p_trans
                rho_quark = self.quark_eos.rho(p[p >= self.p_trans])

                # Combine the results
                rho_combined = np.zeros(p.size)
                rho_combined[p < self.p_trans] = rho_hadron
                rho_combined[p >= self.p_trans] = rho_quark

                # Return the array created
                return rho_combined

            # Return the value given by the Hadron EOS if the EOS is a Hadron EOS
            if self.is_hadron_eos is True:
                return self.hadron_eos.rho(p)

            # Return the value given by the Quark EOS if the EOS is a Quark EOS
            return self.quark_eos.rho(p)

    def p(self, rho):

        if np.ndim(rho) == 0:       # Execute this logic if rho is a scalar

            if self.is_hadron_eos or (self.is_hybrid_eos and (rho < self.rho_trans_min)):
                return self.hadron_eos.p(rho)           # Use the Hadron EOS if rho < rho_trans_min
            if self.is_quark_eos or (self.is_hybrid_eos and (rho > self.rho_trans_max)):
                return self.quark_eos.p(rho)            # Use the Quark EOS if rho > rho_trans_max
            return self.p_trans                         # Return p_trans if rho_trans_min < rho < rho_trans_max

        else:                       # Execute this logic if p is an array

            # Check if the EOS is a Hybrid EOS
            if self.is_hybrid_eos is True:

                # Use the Hadron EOS if rho < rho_trans_min
                p_hadron = self.hadron_eos.p(rho[rho < self.rho_trans_min])

                # Use the Quark EOS if rho > rho_trans_max
                p_quark = self.quark_eos.p(rho[rho > self.rho_trans_max])

                # Combine the results
                p_combined = np.zeros(rho.size)
                p_combined[rho < self.rho_trans_min] = p_hadron
                p_combined[(rho >= self.rho_trans_min) & (rho <= self.rho_trans_max)] = self.p_trans
                p_combined[rho > self.rho_trans_max] = p_quark

                # Return the array created
                return p_combined

            # Return the value given by the Hadron EOS if the EOS is a Hadron EOS
            elif self.is_hadron_eos is True:
                return self.hadron_eos.p(rho)

            # Return the value given by the Quark EOS if the EOS is a Quark EOS
            return self.quark_eos.p(rho)

    def drho_dp(self, p):

        if np.ndim(p) == 0:     # Execute this logic if p is a scalar

            if self.is_hadron_eos or (self.is_hybrid_eos and (p < self.p_trans)):
                return self.hadron_eos.drho_dp(p)       # Use the Hadron EOS if p < p_trans
            return self.quark_eos.drho_dp(p)            # Use the Quark EOS if p >= p_trans

        else:                   # Execute this logic if p is an array

            # Check if the EOS is a Hybrid EOS
            if self.is_hybrid_eos is True:

                # Use the Hadron EOS if p < p_trans
                drho_dp_hadron = self.hadron_eos.drho_dp(p[p < self.p_trans])

                # Use the Quark EOS if p >= p_trans
                drho_dp_quark = self.quark_eos.drho_dp(p[p >= self.p_trans])

                # Combine the results
                drho_dp_combined = np.zeros(p.size)
                drho_dp_combined[p < self.p_trans] = drho_dp_hadron
                drho_dp_combined[p >= self.p_trans] = drho_dp_quark

                # Return the array created
                return drho_dp_combined

            # Return the value given by the Hadron EOS if the EOS is a Hadron EOS
            if self.is_hadron_eos is True:
                return self.hadron_eos.drho_dp(p)

            # Return the value given by the Quark EOS if the EOS is a Quark EOS
            return self.quark_eos.drho_dp(p)

    def dp_drho(self, rho):

        if np.ndim(rho) == 0:       # Execute this logic if rho is a scalar

            if self.is_hadron_eos or (self.is_hybrid_eos and (rho < self.rho_trans_min)):
                return self.hadron_eos.dp_drho(rho)     # Use the Hadron EOS if rho < rho_trans_min
            if self.is_quark_eos or (self.is_hybrid_eos and (rho > self.rho_trans_max)):
                return self.quark_eos.dp_drho(rho)      # Use the Quark EOS if rho > rho_trans_max
            return 0.0                                  # Return zero if rho_trans_min < rho < rho_trans_max

        else:                       # Execute this logic if p is an array

            # Check if the EOS is a Hybrid EOS
            if self.is_hybrid_eos is True:

                # Use the Hadron EOS if rho < rho_trans_min
                dp_drho_hadron = self.hadron_eos.dp_drho(rho[rho < self.rho_trans_min])

                # Use the Quark EOS if rho > rho_trans_max
                dp_drho_quark = self.quark_eos.dp_drho(rho[rho > self.rho_trans_max])

                # Combine the results
                dp_drho_combined = np.zeros(rho.size)
                dp_drho_combined[rho < self.rho_trans_min] = dp_drho_hadron
                dp_drho_combined[(rho >= self.rho_trans_min) & (rho <= self.rho_trans_max)] = 0.0
                dp_drho_combined[rho > self.rho_trans_max] = dp_drho_quark

                # Return the array created
                return dp_drho_combined

            # Return the value given by the Hadron EOS if the EOS is a Hadron EOS
            elif self.is_hadron_eos is True:
                return self.hadron_eos.dp_drho(rho)

            # Return the value given by the Quark EOS if the EOS is a Quark EOS
            return self.quark_eos.dp_drho(rho)


class BSk20EOS(InterpolatedEOS):
    """Class with the functions of the BSk20 EOS
    """

    def _p_analytic(self, rho):

        # Set the a_i parameters
        a = (
            0.0, 4.078, 7.587, 0.00839, 0.21695, 3.614, 11.942, 13.751, 1.3373, 3.606, -22.996, 1.6229,
            4.88, 14.274, 23.560, -1.5564, 2.095, 15.294, 0.084, 6.36, 11.67, -0.042, 14.8, 14.18
        )

        # Calculating xi
        xi = np.log10(rho * uconv.MASS_DENSITY_GU_TO_CGS)

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
        p = 10**(zeta) * uconv.PRESSURE_CGS_TO_GU

        return p


class SLy4EOS(InterpolatedEOS):
    """Class with the functions of the SLy4 EOS
    """

    def _p_analytic(self, rho):

        # Set the a_i parameters
        a = (
            0.0, 6.22, 6.121, 0.005925, 0.16326, 6.48, 11.4971, 19.105, 0.8938, 6.54,
            11.4950, -22.775, 1.5707, 4.3, 14.08, 27.80, -1.653, 1.50, 14.67
        )

        # Calculating xi
        xi = np.log10(rho * uconv.MASS_DENSITY_GU_TO_CGS)

        # Calculating zeta
        zeta = (
            ((a[1] + a[2] * xi + a[3] * xi**3) / (1 + a[4] * xi)) * (np.exp(a[5] * (xi - a[6])) + 1)**(-1)
            + (a[7] + a[8] * xi) * (np.exp(a[9] * (a[10] - xi)) + 1)**(-1)
            + (a[11] + a[12] * xi) * (np.exp(a[13] * (a[14] - xi)) + 1)**(-1)
            + (a[15] + a[16] * xi) * (np.exp(a[17] * (a[18] - xi)) + 1)**(-1)
        )

        # Calculating p
        p = 10**(zeta) * uconv.PRESSURE_CGS_TO_GU

        return p


def main():
    """Main logic
    """

    # Polytropic EOS test

    # Create the EOS object
    polytropic_eos = PolytropicEOS(k=1.0e8, n=1)

    # Set the p_space
    max_rho = 5.80e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Maximum density [m^-2]
    max_p = polytropic_eos.p(max_rho)                       # Maximum pressure [m^-2]
    p_space = max_p * np.logspace(-16.0, 0.0, 1000)

    # Print the minimum pressure calculated. Should be less than 10**21 [dyn cm^-2]
    print(f"PolytropicEOS minimum pressure calculated = {(p_space[0] * uconv.PRESSURE_GU_TO_CGS):e} [dyn cm^-2]")

    # Check the EOS
    polytropic_eos.check_eos(p_space)

    # Create the EOS graphs
    polytropic_eos.plot_all_curves(p_space)

    # Quark EOS test

    # Create the EOS object (values chosen to build a strange star)
    a2 = 100**2     # [MeV^2]
    a4 = 0.6        # [dimensionless]
    B = 130**4      # [MeV^4]
    quark_eos = QuarkEOS(a2, a4, B)

    # Set the p_space
    max_rho = 1.51e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Maximum density [m^-2]
    max_p = quark_eos.p(max_rho)                            # Maximum pressure [m^-2]
    p_space = max_p * np.logspace(-15.0, 0.0, 1000)

    # Print the minimum pressure calculated. Should be less than 10**21 [dyn cm^-2]
    print(f"QuarkEOS minimum pressure calculated = {(p_space[0] * uconv.PRESSURE_GU_TO_CGS):e} [dyn cm^-2]")

    # Check the EOS
    quark_eos.check_eos(p_space)

    # Create the EOS graphs
    quark_eos.plot_all_curves(p_space)

    # BSk20 EOS test

    # Set the rho_space
    max_rho = 2.19e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Maximum density [m^-2]
    rho_space = max_rho * np.logspace(-11.0, 0.0, 10000)

    # Create the EOS object
    bsk20_eos = BSk20EOS(rho_space)

    # Set the p_space
    p_space = bsk20_eos.p(rho_space)

    # Print the minimum pressure calculated. Should be less than 10**21 [dyn cm^-2]
    print(f"BSk20EOS minimum pressure calculated = {(p_space[0] * uconv.PRESSURE_GU_TO_CGS):e} [dyn cm^-2]")

    # Check the EOS
    bsk20_eos.check_eos(p_space)

    # Create the EOS graphs
    bsk20_eos.plot_all_curves(p_space)

    # SLy4 EOS test

    # Set the rho_space
    max_rho = 2.87e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Maximum density [m^-2]
    rho_space = max_rho * np.logspace(-11.0, 0.0, 10000)

    # Create the EOS object
    sly4_eos = SLy4EOS(rho_space)

    # Set the p_space
    p_space = sly4_eos.p(rho_space)

    # Print the minimum pressure calculated. Should be less than 10**21 [dyn cm^-2]
    print(f"SLy4EOS minimum pressure calculated = {(p_space[0] * uconv.PRESSURE_GU_TO_CGS):e} [dyn cm^-2]")

    # Check the EOS
    sly4_eos.check_eos(p_space)

    # Create the EOS graphs
    sly4_eos.plot_all_curves(p_space)

    # Table SLy4 EOS test

    # Create the EOS object
    table_sly4_eos = TableEOS(file_name="data/SLy4_EOS.csv", eos_name="TableSLy4EOS")

    # Set the p_space
    max_rho = 2.87e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Maximum density [m^-2]
    max_p = table_sly4_eos.p(max_rho)                       # Maximum pressure [m^-2]
    p_space = max_p * np.logspace(-11.0, 0.0, 1000)

    # Print the minimum pressure calculated. Should be less than 10**21 [dyn cm^-2]
    print(f"TableSLy4EOS minimum pressure calculated = {(p_space[0] * uconv.PRESSURE_GU_TO_CGS):e} [dyn cm^-2]")

    # Check the EOS
    table_sly4_eos.check_eos(p_space)

    # Create the EOS graphs
    table_sly4_eos.plot_all_curves(p_space)

    # Hybrid EOS test

    # Create the QuarkEOS object (values chosen to build a hybrid star)
    a2 = 100**2     # [MeV^2]
    a4 = 0.8        # [dimensionless]
    B = 160**4      # [MeV^4]
    quark_eos = QuarkEOS(a2, a4, B)

    # Set the p_space
    max_rho = 2.92e15 * uconv.MASS_DENSITY_CGS_TO_GU        # Maximum density [m^-2]
    max_p = quark_eos.p(max_rho)                            # Maximum pressure [m^-2]
    p_space = max_p * np.logspace(-15.0, 0.0, 5000)

    # Set the rho_space
    rho_space = max_rho * np.logspace(-15.0, 0.0, 5000)

    # Create the SLy4EOS object
    sly4_eos = SLy4EOS(rho_space)

    # Create the HybridEOS object
    sly4_maximum_stable_rho_center = 2.865e15 * uconv.MASS_DENSITY_CGS_TO_GU
    hybrid_eos = HybridEOS(quark_eos, sly4_eos, "data/SLy4_EOS.csv", sly4_maximum_stable_rho_center)

    # Print the transition pressure calculated
    if hybrid_eos.p_trans is not None:
        print(f"HybridEOS transition pressure calculated = {(hybrid_eos.p_trans * uconv.PRESSURE_GU_TO_CGS):e} [dyn cm^-2]")

    # Print the minimum pressure calculated. Should be less than 10**21 [dyn cm^-2]
    print(f"HybridEOS minimum pressure calculated = {(p_space[0] * uconv.PRESSURE_GU_TO_CGS):e} [dyn cm^-2]")

    # Check the EOS
    hybrid_eos.check_eos(p_space)

    # Plot the transition graph
    hybrid_eos.plot_transition_graph()

    # Create the EOS graphs
    hybrid_eos.plot_all_curves(rho_space=rho_space)


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
