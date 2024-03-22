from astropy import constants as const
import numpy as np


class Constants:
    """Class with general purpose constants
    """

    C_SMALL = 0.005                     # Small compactness value, used to switch calculations to Taylor expansions for C -> 0


class DefaultValues:
    """Class with default parameter values
    """

    R_INIT = 0.1                        # Initial radial coordinate r of the IVP solve [m]
    R_FINAL = np.inf                    # Final radial coordinate r of the IVP solve [m]
    IVP_METHOD = "RK45"                 # Method used by the IVP solver
    MAX_STEP = np.inf                   # Maximum allowed step size for the IVP solver [m]
    ATOL_TOV = (1e-26, 1e-21, 1e-7)     # Absolute tolerances for the TOV system solution (ATOL = y_min * RTOL * 0.1, except pressure, with a larger tolerance)
    ATOL_TIDAL = 1e-7                   # Absolute tolerance for the tidal system solution (ATOL = y_min * RTOL * 0.1)
    RTOL = 1e-6                         # Relative tolerance of numerical values
    P_SURFACE = 1e-24                   # Surface pressure [m^-2] (Around 1.21e21 dyn ⋅ cm^-2)


class UnitConversion:
    """Class with unit conversion constants.

    Details of each unit system:
    * Geometrical Units (GU) (with the meter as the base unit): used throughout the code for all numerical calculations;
    * Centimetre-gram-second units (CGS): employed for prints, plots, and user inputs for pressure and density;
    * Natural Units (NU) (with the MeV as the base unit): utilized in parts of the code related to particle and nuclear physics,
      but always converted to GU before any numerical calculation;
    * International System of Units (SI): used for converting between other unit systems;
    * Other special astronomical units: applied in specific cases, such as mass in solar masses.

    Ensure consistency when applying these conversions.
    """

    # Universal constants in SI
    MeV = 10**6 * const.e.si.value      # [J]
    hbar = const.hbar.si.value          # [J ⋅ s]
    c = const.c.si.value                # [m ⋅ s^-1]
    G = const.G.si.value                # [m^3 ⋅ kg^-1 ⋅ s^-2]
    M_sun = const.M_sun.si.value        # [kg]

    # Conversion between SI and CGS
    PRESSURE_SI_TO_CGS = 10
    MASS_DENSITY_SI_TO_CGS = 10**(-3)

    # Conversion between NU (with E = MeV) and SI
    ENERGY_DENSITY_NU_TO_SI = MeV**4 * hbar**(-3) * c**(-3)
    ENERGY_DENSITY_SI_TO_NU = ENERGY_DENSITY_NU_TO_SI**(-1)

    # Conversion between GU and SI
    ENERGY_DENSITY_GU_TO_SI = c**4 * G**(-1)
    PRESSURE_GU_TO_SI = c**4 * G**(-1)
    MASS_DENSITY_GU_TO_SI = c**2 * G**(-1)
    MASS_GU_TO_SI = c**2 * G**(-1)
    ENERGY_DENSITY_SI_TO_GU = ENERGY_DENSITY_GU_TO_SI**(-1)
    PRESSURE_SI_TO_GU = PRESSURE_GU_TO_SI**(-1)
    MASS_DENSITY_SI_TO_GU = MASS_DENSITY_GU_TO_SI**(-1)
    MASS_SI_TO_GU = MASS_GU_TO_SI**(-1)

    # Conversion between GU and CGS
    PRESSURE_GU_TO_CGS = PRESSURE_GU_TO_SI * PRESSURE_SI_TO_CGS
    MASS_DENSITY_GU_TO_CGS = MASS_DENSITY_GU_TO_SI * MASS_DENSITY_SI_TO_CGS
    PRESSURE_CGS_TO_GU = PRESSURE_GU_TO_CGS**(-1)
    MASS_DENSITY_CGS_TO_GU = MASS_DENSITY_GU_TO_CGS**(-1)

    # Conversion between GU and NU
    ENERGY_DENSITY_GU_TO_NU = ENERGY_DENSITY_GU_TO_SI * ENERGY_DENSITY_SI_TO_NU
    ENERGY_DENSITY_NU_TO_GU = ENERGY_DENSITY_GU_TO_NU**(-1)

    # Conversion between astronomical units and GU
    MASS_SOLAR_MASS_TO_GU = M_sun * MASS_SI_TO_GU
    MASS_GU_TO_SOLAR_MASS = MASS_SOLAR_MASS_TO_GU**(-1)
