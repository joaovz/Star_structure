import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants


# Defining some constants

# Conversion from CGS to SI
MASS_DENSITY_CGS_TO_SI = 10**3
PRESSURE_CGS_TO_SI = 0.1

# Conversion from NU (with E = MeV) to SI
MeV_to_SI = 10**6 * constants.e
ENERGY_DENSITY_NU_TO_SI = MeV_to_SI**4 * constants.hbar**(-3) * constants.c**(-3)

# Conversion from SI to GU
ENERGY_SI_TO_GU = constants.c**(-4) * constants.G
ENERGY_DENSITY_SI_TO_GU = constants.c**(-4) * constants.G
PRESSURE_SI_TO_GU = constants.c**(-4) * constants.G
MASS_DENSITY_SI_TO_GU = constants.c**(-2) * constants.G

# Conversion from CGS to GU
MASS_DENSITY_CGS_TO_GU = MASS_DENSITY_CGS_TO_SI * MASS_DENSITY_SI_TO_GU
PRESSURE_CGS_TO_GU = PRESSURE_CGS_TO_SI * PRESSURE_SI_TO_GU

# Conversion from NU to GU
ENERGY_DENSITY_NU_TO_GU = ENERGY_DENSITY_NU_TO_SI * ENERGY_DENSITY_SI_TO_GU


def dat_to_array(fname='file_name.dat', usecols=(0, 1), unit_conversion=(1.0, 1.0)):
    """Converts data in a .dat file into numpy arrays

    Args:
        fname (str, optional): Name of the .dat file, including the path. Defaults to 'file_name.dat'
        usecols (tuple, optional): Set the column numbers to read in the .dat file. Defaults to (0, 1)
        unit_conversion (tuple, optional): Conversion multiplicative factor for each column. Defaults to (1.0, 1.0)

    Returns:
        tuple of arrays: Numpy arrays with the data read from the .dat file
    """

    x, y = np.loadtxt(fname=fname, usecols=usecols, unpack=True)
    x_converted, y_converted = x * unit_conversion[0], y * unit_conversion[1]

    return x_converted, y_converted


# This logic is a simple example, only executed when this file is run directly in the command prompt
if __name__ == "__main__":

    # Open an example .dat file
    rho, p = dat_to_array(
        fname='data/EOSFull_GM1_BPS.dat',
        usecols=(0, 1),
        unit_conversion=(MASS_DENSITY_CGS_TO_GU, PRESSURE_CGS_TO_GU))

    # Plot the curve given in the example .dat file
    plt.figure()
    plt.plot(p, rho, linewidth=1, marker='.')
    plt.title("EOS from .dat file")
    plt.xlabel('$p ~ [m^{-2}]$')
    plt.ylabel('$\\rho ~ [m^{-2}]$')
    plt.show()
