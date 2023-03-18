import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants

# Defining some constants
DENSITY_CGS_TO_GU = 10**3 * constants.c**(-2) * constants.G
PRESSURE_CGS_TO_GU = 0.1 * constants.c**(-4) * constants.G


def dat_to_array(fname='file_name.dat', usecols=(0, 1), unit_convertion=(1.0, 1.0)):
    """Converts data in a .dat file into numpy arrays

    Args:
        fname (str, optional): Name of the .dat file, including the path. Defaults to 'file_name.dat'
        usecols (tuple, optional): Set the column numbers to read in the .dat file. Defaults to (0, 1)
        unit_convertion (tuple, optional): Conversion multiplicative factor for each column. Defaults to (1.0, 1.0)

    Returns:
        tuple of arrays: Numpy arrays with the data read from the .dat file
    """

    x, y = np.loadtxt(fname=fname, usecols=usecols, unpack=True)
    x_converted, y_converted = x * unit_convertion[0], y * unit_convertion[1]

    return x_converted, y_converted


# This logic is a simple example, only executed when this file is run directly in the command prompt
if __name__ == "__main__":

    # Open an example .dat file
    rho, p = dat_to_array(
        fname='data/EOSFull_GM1_BPS.dat',
        usecols=(0, 1),
        unit_convertion=(DENSITY_CGS_TO_GU, PRESSURE_CGS_TO_GU))

    # Plot the curve given in the example .dat file
    plt.figure()
    plt.plot(np.log10(rho), np.log10(p), linewidth=1)
    plt.title("EOS from .dat file")
    plt.xlabel('$\\log(\\rho[m^{-2}])$')
    plt.ylabel('$\\log(p[m^{-2}])$')
    plt.show()
