import os
import math
import multiprocessing
import pprint
import psutil
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.stats import qmc
from constants import Constants as const
from constants import UnitConversion as uconv
from data_handling import dataframe_to_csv, dict_to_json
from eos_library import QuarkEOS
from star_family_tides import DeformedStarFamily


# Constants
g0 = 930.0                                                              # Gibbs free energy per baryon of quark matter at null pressure [MeV]
alpha = ((1 / 3) - 8 / (3 * (1 + 2**(1 / 3))**3)) * g0**2               # alpha coefficient of the a2_max vs a4 curve [MeV^2]
a2_min = -3 * alpha                                                     # Minimum a2 parameter value [MeV^2]
a2_max = alpha                                                          # Maximum a2 parameter value [MeV^2]
a4_min = 0.05                                                           # Minimum a4 parameter value [dimensionless]
a4_max = 1.0                                                            # Maximum a4 parameter value [dimensionless]
B_min = 0.0                                                             # Minimum B parameter value [MeV^4]
B_max = (g0**2 / (108 * np.pi**2)) * (g0**2 * a4_max - 9 * a2_min)      # Maximum B parameter value [MeV^4]

# Observation data
M_max_inf_limit = 2.13                  # Inferior limit of the maximum mass [solar mass] (Romani - 2 sigma)
M_max_sup_limit = 2.33                  # Superior limit of the maximum mass [solar mass] (Rezzolla - 2 sigma)
R_canonical_inf_limit = 10.0            # Inferior limit of the radius of the canonical star [km] (Pang)
R_canonical_sup_limit = 13.25           # Superior limit of the radius of the canonical star [km] (Pang)
Lambda_canonical_sup_limit = 970.0      # Superior limit of the tidal deformability of the canonical star [dimensionless] (Abbott - 2 sigma)

# Quark star candidate data (XMMU J173203.3-344518)
QS_CANDIDATE_M = 0.77
QS_CANDIDATE_M_ERROR = [[0.17], [0.20]]
QS_CANDIDATE_R = 10.4
QS_CANDIDATE_R_ERROR = [[0.78], [0.86]]

# Numerical solver constants
QUARK_EOS_ANALYSIS_R_INIT = 0.01                        # Initial radial coordinate r of the IVP solve [m]
QUARK_EOS_ANALYSIS_ATOL_TOV = (1e-25, 1e-20, 1e-6)      # Absolute tolerances for the TOV system solution (ATOL = y_min * RTOL * 0.1, except pressure, with a larger tolerance)
QUARK_EOS_ANALYSIS_ATOL_TIDAL = 1e-6                    # Absolute tolerance for the tidal system solution (ATOL = y_min * RTOL * 0.1)
QUARK_EOS_ANALYSIS_RTOL = 1e-5                          # Relative tolerance of numerical values


def calc_B_max(a2, a4):
    """Function that calculates the maximum B parameter value

    Args:
        a2 (array of float): a2 parameter of the EOS [MeV^2]
        a4 (array of float): a4 parameter of the EOS [dimensionless]

    Returns:
        array of float: Maximum B value [MeV^4]
    """
    return (g0**2 / (108 * np.pi**2)) * (g0**2 * a4 - 9 * a2)


def calc_B_min(a2, a4):
    """Function that calculates the minimum B parameter value

    Args:
        a2 (array of float): a2 parameter of the EOS [MeV^2]
        a4 (array of float): a4 parameter of the EOS [dimensionless]

    Returns:
        array of float: Minimum B value [MeV^4]
    """
    return (g0**2 / (54 * np.pi**2)) * ((4 * g0**2 * a4) / ((1 + 2**(1 / 3))**3) - 3 * a2)


def generate_strange_stars(number_of_samples=10**4):
    """Function that generates a list of points in the parameter space of strange stars.
    This function also creates a dataframe with the parameters of strange stars

    Args:
        number_of_samples (int, optional): Number of samples used. Defaults to 10**4

    Returns:
        Arrays of float: Points in the (a2, a4, B) parameter space of strange stars
        Pandas dataframe of float: Dataframe with the parameters of strange stars
    """

    # 1. Create points in the (a2, a4, B) space using quasi-random Latin Hypercube sampler
    seed_value = 123                                            # Fix the seed value to generate the same pseudo-random values each time
    sampler = qmc.LatinHypercube(d=3, seed=seed_value)          # Set the sampler
    samples = sampler.random(n=number_of_samples)               # Create the samples
    l_bounds = [a2_min, a4_min, B_min**(1 / 4)]
    u_bounds = [a2_max, a4_max, B_max**(1 / 4)]
    scaled_samples = qmc.scale(samples, l_bounds, u_bounds)     # Scale the samples
    (a2, a4, B) = (scaled_samples[:, 0], scaled_samples[:, 1], scaled_samples[:, 2]**4)

    # Define the mask with all the constraints
    mask = (
        (a2 >= a2_min) &
        (a2 <= alpha * a4) &
        (a4 >= a4_min) &
        (a4 <= a4_max) &
        (B >= calc_B_min(a2, a4)) &
        (B <= calc_B_max(a2, a4))
    )

    # Apply the mask to extract valid points - inside the allowed volume
    (a2_inside, a4_inside, B_inside) = (a2[mask], a4[mask], B[mask])

    # 2. Create points in the surface B_min(a2, a4) using quasi-random Latin Hypercube sampler
    seed_value = 456                                                    # Fix the seed value to generate the same pseudo-random values each time
    sampler = qmc.LatinHypercube(d=2, seed=seed_value)                  # Set the sampler
    samples = sampler.random(n=2 * int(number_of_samples**(2 / 3)))     # Create the samples
    l_bounds = [a2_min, a4_min]
    u_bounds = [a2_max, a4_max]
    scaled_samples = qmc.scale(samples, l_bounds, u_bounds)             # Scale the samples
    a2 = scaled_samples[:, 0]
    a4 = scaled_samples[:, 1]
    B = calc_B_min(a2, a4)

    # Define the mask with all the constraints
    mask = (
        (a2 >= a2_min) &
        (a2 <= alpha * a4) &
        (a4 >= a4_min) &
        (a4 <= a4_max)
    )

    # Apply the mask to extract valid points - on the surface B_min
    (a2_surface_B_min, a4_surface_B_min, B_surface_B_min) = (a2[mask], a4[mask], B[mask])

    # 3. Create points in the surface B_max(a2, a4) using quasi-random Latin Hypercube sampler
    seed_value = 789                                                    # Fix the seed value to generate the same pseudo-random values each time
    sampler = qmc.LatinHypercube(d=2, seed=seed_value)                  # Set the sampler
    samples = sampler.random(n=2 * int(number_of_samples**(2 / 3)))     # Create the samples
    l_bounds = [a2_min, a4_min]
    u_bounds = [a2_max, a4_max]
    scaled_samples = qmc.scale(samples, l_bounds, u_bounds)             # Scale the samples
    a2 = scaled_samples[:, 0]
    a4 = scaled_samples[:, 1]
    B = calc_B_max(a2, a4)

    # Define the mask with all the constraints
    mask = (
        (a2 >= a2_min) &
        (a2 <= alpha * a4) &
        (a4 >= a4_min) &
        (a4 <= a4_max)
    )

    # Apply the mask to extract valid points - on the surface B_max
    (a2_surface_B_max, a4_surface_B_max, B_surface_B_max) = (a2[mask], a4[mask], B[mask])

    # 4. Group the sample points across all sources (interior, B_min surface, B_max surface)
    a2_masked = np.concatenate([a2_inside, a2_surface_B_min, a2_surface_B_max])
    a4_masked = np.concatenate([a4_inside, a4_surface_B_min, a4_surface_B_max])
    B_masked = np.concatenate([B_inside, B_surface_B_min, B_surface_B_max])

    # Create the dataframe with the parameter points of strange stars
    parameter_dataframe = pd.DataFrame({
        "a2 [10^3 MeV^2]": a2_masked * 10**(-3),
        "a4 [dimensionless]": a4_masked,
        "B^(1/4) [MeV]": B_masked**(1 / 4)
    })

    return (a2_masked, a4_masked, B_masked, parameter_dataframe)


def analyze_strange_star_family(dataframe_row):
    """Function that analyzes a star family, calculating the properties

    Args:
        dataframe_row (list): Row of the dataframe with the index and quark EOS parameters

    Returns:
        tuple: Tuple with index and star family properties calculated
    """

    # Unpack the row values
    (index, a2_1000, a4, B_1_4, *_) = dataframe_row
    a2 = a2_1000 * 10**3
    B = B_1_4**4

    # Create the EOS object
    quark_eos = QuarkEOS(a2, a4, B)

    # Get the surface density
    rho_surface = quark_eos.rho(0.0)

    # TOV and tidal analysis

    # Set the central pressure of the star
    rho_center = 1.0e16 * uconv.MASS_DENSITY_CGS_TO_GU      # Central density [m^-2]
    p_center = quark_eos.p(rho_center)                      # Central pressure [m^-2]

    # Set the p_center space that characterizes the star family
    p_center_space = p_center * np.logspace(-3.0, 0.0, 20)

    # Create the star family object
    star_family_object = DeformedStarFamily(
        quark_eos, p_center_space,
        r_init=QUARK_EOS_ANALYSIS_R_INIT,
        atol_tov=QUARK_EOS_ANALYSIS_ATOL_TOV,
        atol_tidal=QUARK_EOS_ANALYSIS_ATOL_TIDAL,
        rtol=QUARK_EOS_ANALYSIS_RTOL)

    # Find the maximum mass star
    star_family_object.find_maximum_mass_star()
    maximum_stable_rho_center = star_family_object.maximum_stable_rho_center
    maximum_mass = star_family_object.maximum_mass

    # Find the canonical star
    star_family_object.find_canonical_star()
    canonical_rho_center = star_family_object.canonical_rho_center
    canonical_radius = star_family_object.canonical_radius
    canonical_lambda = star_family_object.canonical_lambda

    # Find the maximum k2 star
    star_family_object.find_maximum_k2_star()
    maximum_k2_star_rho_center = star_family_object.maximum_k2_star_rho_center
    maximum_k2 = star_family_object.maximum_k2

    # Get the minimum and maximum sound speeds
    cs_rho_surface = quark_eos.c_s(rho_surface)
    cs_rho_center_max = quark_eos.c_s(maximum_stable_rho_center)
    minimum_cs = min(cs_rho_surface, cs_rho_center_max)
    maximum_cs = max(cs_rho_surface, cs_rho_center_max)

    # Return the index and results
    return (index, rho_surface, minimum_cs, maximum_stable_rho_center, maximum_mass, maximum_cs, canonical_rho_center, canonical_radius, canonical_lambda, maximum_k2_star_rho_center, maximum_k2)


def analyze_strange_stars(parameter_dataframe):
    """Function that analyzes the strange stars given by the parameters in the dataframe, calculating the properties

    Args:
        parameter_dataframe (Pandas dataframe of float): Dataframe with the parameters of strange stars

    Returns:
        Pandas dataframe of float: Dataframe with the parameters and properties of strange stars
        Pandas dataframe of float: Dataframe with the parameters and properties of strange stars filtered by the observation data restrictions
        dict: Dictionary with the minimum and maximum values of the parameters for each observation data restrictions
        dict: Dictionary with the minimum and maximum values of the properties of strange stars
    """

    # Import tqdm multiprocessing tool only inside the function it is used to ensure correct setup
    from tqdm.contrib.concurrent import process_map

    # Calculate the number of rows, number of processes and number of calculations per process (chunksize)
    n_rows = parameter_dataframe.shape[0]
    processes = psutil.cpu_count(logical=False)             # Number of processes are equal to the number of hardware cores
    chunksize = math.ceil(n_rows / (processes * 10))        # Create the chunksize smaller to provide some feedback on progress

    # Create a list with the rows of the dataframe
    rows_list = [list(row) for row in parameter_dataframe.itertuples()]

    # Check if DEBUG is activated
    if const.DEBUG is True:
        # Execute the analysis for each row sequentially
        results = [analyze_strange_star_family(row) for row in rows_list]
    else:
        # Execute the analysis for each row in parallel processes, using a progress bar from tqdm
        results = process_map(analyze_strange_star_family, rows_list, max_workers=processes, chunksize=chunksize)

    # Update the dataframe with the results
    for index, rho_surface, minimum_cs, maximum_stable_rho_center, maximum_mass, maximum_cs, canonical_rho_center, canonical_radius, canonical_lambda, maximum_k2_star_rho_center, maximum_k2 in results:
        parameter_dataframe.at[index, "rho_surface [10^15 g cm^-3]"] = rho_surface * uconv.MASS_DENSITY_GU_TO_CGS / 10**15
        parameter_dataframe.at[index, "cs_min [dimensionless]"] = minimum_cs
        parameter_dataframe.at[index, "rho_center_max [10^15 g cm^-3]"] = maximum_stable_rho_center * uconv.MASS_DENSITY_GU_TO_CGS / 10**15
        parameter_dataframe.at[index, "M_max [solar mass]"] = maximum_mass * uconv.MASS_GU_TO_SOLAR_MASS
        parameter_dataframe.at[index, "cs_max [dimensionless]"] = maximum_cs
        parameter_dataframe.at[index, "rho_center_canonical [10^15 g cm^-3]"] = canonical_rho_center * uconv.MASS_DENSITY_GU_TO_CGS / 10**15
        parameter_dataframe.at[index, "R_canonical [km]"] = canonical_radius / 10**3
        parameter_dataframe.at[index, "Lambda_canonical [dimensionless]"] = canonical_lambda
        parameter_dataframe.at[index, "rho_center_k2_max [10^15 g cm^-3]"] = maximum_k2_star_rho_center * uconv.MASS_DENSITY_GU_TO_CGS / 10**15
        parameter_dataframe.at[index, "k2_max [dimensionless]"] = maximum_k2

    # Determine the EOS parameters limits based on observation data and create filtered dataframes
    M_max_query = f"(`M_max [solar mass]` > {M_max_inf_limit}) & (`M_max [solar mass]` < {M_max_sup_limit})"
    R_canonical_query = f"(`R_canonical [km]` > {R_canonical_inf_limit}) & (`R_canonical [km]` < {R_canonical_sup_limit})"
    Lambda_canonical_query = f"(`Lambda_canonical [dimensionless]` < {Lambda_canonical_sup_limit})"
    combined_query = f"{M_max_query} & {R_canonical_query} & {Lambda_canonical_query}"
    filtered_M_max_dataframe = parameter_dataframe.query(M_max_query)
    filtered_R_canonical_dataframe = parameter_dataframe.query(R_canonical_query)
    filtered_Lambda_canonical_dataframe = parameter_dataframe.query(Lambda_canonical_query)
    filtered_dataframe = parameter_dataframe.query(combined_query)

    # Create a dictionary with the minimum and maximum values of the parameters for each observation data restrictions
    parameters_limits = {
        "a2 [10^3 MeV^2]": {
            "M_max [solar mass]": (np.min(filtered_M_max_dataframe.loc[:, "a2 [10^3 MeV^2]"]), np.max(filtered_M_max_dataframe.loc[:, "a2 [10^3 MeV^2]"])),
            "R_canonical [km]": (np.min(filtered_R_canonical_dataframe.loc[:, "a2 [10^3 MeV^2]"]), np.max(filtered_R_canonical_dataframe.loc[:, "a2 [10^3 MeV^2]"])),
            "Lambda_canonical [dimensionless]": (np.min(filtered_Lambda_canonical_dataframe.loc[:, "a2 [10^3 MeV^2]"]), np.max(filtered_Lambda_canonical_dataframe.loc[:, "a2 [10^3 MeV^2]"])),
            "combined": (np.min(filtered_dataframe.loc[:, "a2 [10^3 MeV^2]"]), np.max(filtered_dataframe.loc[:, "a2 [10^3 MeV^2]"])),
        },
        "a4 [dimensionless]": {
            "M_max [solar mass]": (np.min(filtered_M_max_dataframe.loc[:, "a4 [dimensionless]"]), np.max(filtered_M_max_dataframe.loc[:, "a4 [dimensionless]"])),
            "R_canonical [km]": (np.min(filtered_R_canonical_dataframe.loc[:, "a4 [dimensionless]"]), np.max(filtered_R_canonical_dataframe.loc[:, "a4 [dimensionless]"])),
            "Lambda_canonical [dimensionless]": (np.min(filtered_Lambda_canonical_dataframe.loc[:, "a4 [dimensionless]"]), np.max(filtered_Lambda_canonical_dataframe.loc[:, "a4 [dimensionless]"])),
            "combined": (np.min(filtered_dataframe.loc[:, "a4 [dimensionless]"]), np.max(filtered_dataframe.loc[:, "a4 [dimensionless]"])),
        },
        "B^(1/4) [MeV]": {
            "M_max [solar mass]": (np.min(filtered_M_max_dataframe.loc[:, "B^(1/4) [MeV]"]), np.max(filtered_M_max_dataframe.loc[:, "B^(1/4) [MeV]"])),
            "R_canonical [km]": (np.min(filtered_R_canonical_dataframe.loc[:, "B^(1/4) [MeV]"]), np.max(filtered_R_canonical_dataframe.loc[:, "B^(1/4) [MeV]"])),
            "Lambda_canonical [dimensionless]": (np.min(filtered_Lambda_canonical_dataframe.loc[:, "B^(1/4) [MeV]"]), np.max(filtered_Lambda_canonical_dataframe.loc[:, "B^(1/4) [MeV]"])),
            "combined": (np.min(filtered_dataframe.loc[:, "B^(1/4) [MeV]"]), np.max(filtered_dataframe.loc[:, "B^(1/4) [MeV]"])),
        },
    }

    # Create a dictionary with the minimum and maximum values of the properties of strange stars
    properties_limits = {
        "rho_surface [10^15 g cm^-3]": (np.min(filtered_dataframe.loc[:, "rho_surface [10^15 g cm^-3]"]), np.max(filtered_dataframe.loc[:, "rho_surface [10^15 g cm^-3]"])),
        "cs_min [dimensionless]": (np.min(filtered_dataframe.loc[:, "cs_min [dimensionless]"]), np.max(filtered_dataframe.loc[:, "cs_min [dimensionless]"])),
        "rho_center_max [10^15 g cm^-3]": (np.min(filtered_dataframe.loc[:, "rho_center_max [10^15 g cm^-3]"]), np.max(filtered_dataframe.loc[:, "rho_center_max [10^15 g cm^-3]"])),
        "M_max [solar mass]": (np.min(filtered_dataframe.loc[:, "M_max [solar mass]"]), np.max(filtered_dataframe.loc[:, "M_max [solar mass]"])),
        "cs_max [dimensionless]": (np.min(filtered_dataframe.loc[:, "cs_max [dimensionless]"]), np.max(filtered_dataframe.loc[:, "cs_max [dimensionless]"])),
        "rho_center_canonical [10^15 g cm^-3]": (np.min(filtered_dataframe.loc[:, "rho_center_canonical [10^15 g cm^-3]"]), np.max(filtered_dataframe.loc[:, "rho_center_canonical [10^15 g cm^-3]"])),
        "R_canonical [km]": (np.min(filtered_dataframe.loc[:, "R_canonical [km]"]), np.max(filtered_dataframe.loc[:, "R_canonical [km]"])),
        "Lambda_canonical [dimensionless]": (np.min(filtered_dataframe.loc[:, "Lambda_canonical [dimensionless]"]), np.max(filtered_dataframe.loc[:, "Lambda_canonical [dimensionless]"])),
        "rho_center_k2_max [10^15 g cm^-3]": (np.min(filtered_dataframe.loc[:, "rho_center_k2_max [10^15 g cm^-3]"]), np.max(filtered_dataframe.loc[:, "rho_center_k2_max [10^15 g cm^-3]"])),
        "k2_max [dimensionless]": (np.min(filtered_dataframe.loc[:, "k2_max [dimensionless]"]), np.max(filtered_dataframe.loc[:, "k2_max [dimensionless]"])),
    }

    # Print the parameter dataframe, parameters limits, and properties limits
    print(parameter_dataframe)
    pprint.pp(parameters_limits)
    pprint.pp(properties_limits)

    # Return the parameter dataframe, the filtered dataframe, the parameters limits, and the properties limits
    return (parameter_dataframe, filtered_dataframe, parameters_limits, properties_limits)


def plot_parameter_points_scatter(a2, a4, B, figure_path="figures/app_quark_eos"):
    """Function that plots the scatter graph of the parameter points

    Args:
        a2 (Array of float): Array with the a2 parameter values of the EOS [MeV^2]
        a4 (Array of float): Array with the a4 parameter values of the EOS [dimensionless]
        B (Array of float): Array with the B parameter values of the EOS [MeV^4]
        figure_path (str, optional): Path used to save the figure. Defaults to "figures/app_quark_eos"
    """

    # Create figure and change properties
    (fig, ax) = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5.0, 4.0), constrained_layout=True)
    ax.view_init(elev=15, azim=-70, roll=0)
    ax.set_xlim3d(a2_min * 10**(-3), a2_max * 10**(-3))
    ax.set_ylim3d(a4_min, a4_max)
    ax.set_zlim3d(B_min**(1 / 4), B_max**(1 / 4))
    ax.set_xlabel("$a_2 ~ [10^3 ~ MeV^2]$", fontsize=10, rotation=0)
    ax.set_ylabel("$a_4 ~ [dimensionless]$", fontsize=10, rotation=0)
    ax.set_zlabel("$B^{1/4} ~ [MeV]$", fontsize=10, rotation=90)
    ax.zaxis.set_rotate_label(False)
    ax.set_position([-0.06, -0.05, 1.05, 1.2])      # Adjust plot position and size inside image to remove excessive whitespaces

    # Add each scatter point
    ax.scatter(a2 * 10**(-3), a4, B**(1 / 4), s=2.0**2, rasterized=True)

    # Create the folder if necessary and save the figure
    os.makedirs(figure_path, exist_ok=True)
    figure_name = "quark_eos_parameter_points.pdf"
    complete_path = os.path.join(figure_path, figure_name)
    plt.savefig(complete_path, dpi=400)

    # Show graph
    plt.show()


def plot_parameter_space(mesh_size=1000, figure_path="figures/app_quark_eos"):
    """Function that plots the graph of the parameter space

    Args:
        mesh_size (int, optional): Size of the mesh used to create the plot. Defaults to 1000
        figure_path (str, optional): Path used to save the figure. Defaults to "figures/app_quark_eos"
    """

    # Define the (a2, a4) rectangular meshgrid
    a2_range = np.linspace(a2_min, a2_max, mesh_size)
    a4_range = np.linspace(a4_min, a4_max, mesh_size)
    (a2, a4) = np.meshgrid(a2_range, a4_range)

    # Create the B_max and B_min surfaces
    B_max_surface = calc_B_max(a2, a4)
    B_min_surface = calc_B_min(a2, a4)

    # Apply the triangular mask to the meshgrid
    mesh_mask = (a2 > alpha * a4)
    a2_masked = np.ma.masked_where(mesh_mask, a2)
    a4_masked = np.ma.masked_where(mesh_mask, a4)
    B_max_surface_masked = np.ma.masked_where(mesh_mask, B_max_surface)
    B_min_surface_masked = np.ma.masked_where(mesh_mask, B_min_surface)

    # Create figure and change properties
    (fig, ax) = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5.0, 4.0), constrained_layout=True)
    ax.view_init(elev=15, azim=-70, roll=0)
    ax.set_xlim3d(a2_min * 10**(-3), a2_max * 10**(-3))
    ax.set_ylim3d(a4_min, a4_max)
    ax.set_zlim3d(B_min**(1 / 4), B_max**(1 / 4))
    ax.set_xlabel("$a_2 ~ [10^3 ~ MeV^2]$", fontsize=10, rotation=0)
    ax.set_ylabel("$a_4 ~ [dimensionless]$", fontsize=10, rotation=0)
    ax.set_zlabel("$B^{1/4} ~ [MeV]$", fontsize=10, rotation=90)
    ax.zaxis.set_rotate_label(False)
    ax.set_position([-0.06, -0.05, 1.05, 1.2])      # Adjust plot position and size inside image to remove excessive whitespaces

    # Add each surface plot
    B_1_4_max_surface_masked = B_max_surface_masked**(1 / 4)
    B_1_4_min_surface_masked = B_min_surface_masked**(1 / 4)
    ax.plot_surface(a2_masked * 10**(-3), a4_masked, B_1_4_max_surface_masked, cmap=cm.Reds, rstride=10, cstride=10, alpha=0.8, label="$B_{max}^{1/4}$", rasterized=True)
    ax.plot_surface(a2_masked * 10**(-3), a4_masked, B_1_4_min_surface_masked, cmap=cm.Blues, rstride=10, cstride=10, alpha=0.8, label="$B_{min}^{1/4}$", rasterized=True)

    # Create custom legend handles using Patches, and add the legend
    red_patch = Patch(color=cm.Reds(0.5), label="$B_{max}^{1/4}$")
    blue_patch = Patch(color=cm.Blues(0.5), label="$B_{min}^{1/4}$")
    ax.legend(handles=[red_patch, blue_patch], loc=(0.3, 0.22))

    # Add each contour plot (grey projections on each plane)
    ax.contourf(a2_masked * 10**(-3), a4_masked, B_1_4_max_surface_masked, levels=0, zdir="x", offset=a2_min * 10**(-3), colors="gray", alpha=0.7, antialiased=True)
    ax.contourf(a2_masked * 10**(-3), a4_masked, B_1_4_max_surface_masked, levels=0, zdir="z", offset=0, colors="gray", alpha=0.7, antialiased=True)

    # Create the folder if necessary and save the figure
    os.makedirs(figure_path, exist_ok=True)
    figure_name = "quark_eos_parameter_space.pdf"
    complete_path = os.path.join(figure_path, figure_name)
    plt.savefig(complete_path, dpi=400)

    # Show graph
    plt.show()


def plot_analysis_graphs(parameter_dataframe, parameters_limits, figures_path="figures/app_quark_eos/analysis"):
    """Function that creates all the analysis graphs

    Args:
        parameter_dataframe (Pandas dataframe of float): Dataframe with the parameters of strange stars
        parameters_limits (dict): Dictionary with the minimum and maximum values of the parameters for each observation data restrictions
        figures_path (str, optional): Path used to save the figures. Defaults to "figures/app_quark_eos/analysis"
    """

    # Create a dictionary with all the functions used in plotting, with each name and label description
    plot_dict = {
        "a2 [10^3 MeV^2]": {
            "name": "a2",
            "label": "$a_2 ~ [10^3 ~ MeV^2]$",
            "value": parameter_dataframe.loc[:, "a2 [10^3 MeV^2]"],
        },
        "a4 [dimensionless]": {
            "name": "a4",
            "label": "$a_4 ~ [dimensionless]$",
            "value": parameter_dataframe.loc[:, "a4 [dimensionless]"],
        },
        "B^(1/4) [MeV]": {
            "name": "B",
            "label": "$B^{1/4} ~ [MeV]$",
            "value": parameter_dataframe.loc[:, "B^(1/4) [MeV]"],
        },
        "M_max [solar mass]": {
            "name": "Maximum mass",
            "label": "$M_{max} ~ [M_{\\odot}]$",
            "value": parameter_dataframe.loc[:, "M_max [solar mass]"],
            "inf_limit": M_max_inf_limit,
            "sup_limit": M_max_sup_limit,
        },
        "R_canonical [km]": {
            "name": "Canonical radius",
            "label": "$R_{1.4 M_{\\odot}} ~ [km]$",
            "value": parameter_dataframe.loc[:, "R_canonical [km]"],
            "inf_limit": R_canonical_inf_limit,
            "sup_limit": R_canonical_sup_limit,
        },
        "Lambda_canonical [dimensionless]": {
            "name": "Canonical deformability",
            "label": "$\\log_{10} \\left( \\Lambda_{1.4 M_{\\odot}} ~ [dimensionless] \\right)$",
            "value": np.log10(parameter_dataframe.loc[:, "Lambda_canonical [dimensionless]"]),
            "inf_limit": None,
            "sup_limit": np.log10(Lambda_canonical_sup_limit),
        },
    }

    # Create a list with all the graphs to be plotted
    graphs_list = [
        ["a2 [10^3 MeV^2]", "M_max [solar mass]"],
        ["a4 [dimensionless]", "M_max [solar mass]"],
        ["B^(1/4) [MeV]", "M_max [solar mass]"],
        ["a2 [10^3 MeV^2]", "R_canonical [km]"],
        ["a4 [dimensionless]", "R_canonical [km]"],
        ["B^(1/4) [MeV]", "R_canonical [km]"],
        ["a2 [10^3 MeV^2]", "Lambda_canonical [dimensionless]"],
        ["a4 [dimensionless]", "Lambda_canonical [dimensionless]"],
        ["B^(1/4) [MeV]", "Lambda_canonical [dimensionless]"],
    ]

    # Plot all graphs specified in graphs_list
    for (x_axis, y_axis) in graphs_list:

        # Create the plot
        plt.figure(figsize=(6.0, 4.5))
        plt.scatter(plot_dict[x_axis]["value"], plot_dict[y_axis]["value"], s=2.0**2, zorder=2, rasterized=True)
        plt.xlabel(plot_dict[x_axis]["label"], fontsize=10)
        plt.ylabel(plot_dict[y_axis]["label"], fontsize=10)

        # Add the y axis limit lines and text
        xlim0, xlim1 = plt.xlim()
        if plot_dict[y_axis]["sup_limit"] is not None:
            plt.axhline(y=plot_dict[y_axis]["sup_limit"], linewidth=1, color="#d62728", zorder=3)
            text = f" {plot_dict[y_axis]["sup_limit"]:.2f} "
            plt.text(xlim1, plot_dict[y_axis]["sup_limit"], text, horizontalalignment="left", verticalalignment="bottom", color="#d62728")
        if plot_dict[y_axis]["inf_limit"] is not None:
            plt.axhline(y=plot_dict[y_axis]["inf_limit"], linewidth=1, color="#2ca02c", zorder=3)
            text = f" {plot_dict[y_axis]["inf_limit"]:.2f} "
            plt.text(xlim1, plot_dict[y_axis]["inf_limit"], text, horizontalalignment="left", verticalalignment="top", color="#2ca02c")

        # Create a shaded region between y axis limit lines
        ylim0, ylim1 = plt.ylim()       # Get current y limits to use for the shaded region if necessary
        span_min = plot_dict[y_axis]["inf_limit"]
        span_max = plot_dict[y_axis]["sup_limit"]
        if span_min is None:
            span_min = ylim0
        if span_max is None:
            span_max = ylim1
        plt.axhspan(span_min, span_max, facecolor="#2ca02c", alpha=0.25, zorder=1)
        plt.ylim(ylim0, ylim1)          # Set original y limits after creating the shaded region

        # Add the x axis limit lines and text
        (x_inf_limit, x_sup_limit) = parameters_limits[x_axis][y_axis]
        # Superior limit
        plt.axvline(x=x_sup_limit, linewidth=1, color="#d62728", zorder=4)
        text = f" {x_sup_limit:.2f} "
        plt.text(x_sup_limit, ylim1, text, horizontalalignment="center", verticalalignment="bottom", color="#d62728")
        # Inferior limit
        plt.axvline(x=x_inf_limit, linewidth=1, color="#2ca02c", zorder=4)
        text = f" {x_inf_limit:.2f} "
        plt.text(x_inf_limit, ylim1, text, horizontalalignment="center", verticalalignment="bottom", color="#2ca02c")

        # Create the folder if necessary and save the figure
        os.makedirs(figures_path, exist_ok=True)
        x_axis_name = plot_dict[x_axis]["name"].lower().replace(" ", "_").replace("/", "_")
        y_axis_name = plot_dict[y_axis]["name"].lower().replace(" ", "_").replace("/", "_")
        figure_name = f"{y_axis_name}_vs_{x_axis_name}_graph.pdf"
        complete_path = os.path.join(figures_path, figure_name)
        plt.savefig(complete_path, bbox_inches="tight", dpi=400)

    # Show graphs at the end
    plt.show()


def plot_mass_radius_graph(filtered_dataframe, figures_path="figures/app_quark_eos/analysis"):
    """Function that creates the mass vs radius graph

    Args:
        filtered_dataframe (Pandas dataframe of float): Filtered dataframe with the parameters of strange stars
        figures_path (str, optional): Path used to save the figures. Defaults to "figures/app_quark_eos/analysis"
    """

    # Create a simple plot
    plt.figure(figsize=(6.0, 4.5))
    plt.xlabel("$R ~ [km]$", fontsize=10)
    plt.ylabel("$M ~ [M_{\\odot}]$", fontsize=10)

    # Create a list with the rows of the dataframe
    rows_list = [list(row) for row in filtered_dataframe.itertuples()]

    # Execute the TOV solver and add the mass vs radius curve to the graph
    for row in rows_list:

        # Unpack the row values
        (index, a2_1000, a4, B_1_4, rho_surface, cs_min, rho_center_max, *_) = row
        a2 = a2_1000 * 10**3
        B = B_1_4**4

        # Create the EOS object
        quark_eos = QuarkEOS(a2, a4, B)

        # Set the central pressure of the star
        rho_center = rho_center_max * 10**15 * uconv.MASS_DENSITY_CGS_TO_GU     # Central density [m^-2]
        p_center = quark_eos.p(rho_center)                                      # Central pressure [m^-2]

        # Set the p_center space that characterizes the star family
        p_center_space = p_center * np.logspace(-3.0, 0.0, 50)

        # Create the star family object
        star_family_object = DeformedStarFamily(
            quark_eos, p_center_space,
            r_init=QUARK_EOS_ANALYSIS_R_INIT,
            atol_tov=QUARK_EOS_ANALYSIS_ATOL_TOV,
            atol_tidal=QUARK_EOS_ANALYSIS_ATOL_TIDAL,
            rtol=QUARK_EOS_ANALYSIS_RTOL)

        # Solve the TOV system
        star_family_object.solve_tov(False)

        # Add the mass vs radius curve to the graph
        plt.plot(star_family_object.radius_array / 10**3, star_family_object.mass_array * uconv.MASS_GU_TO_SOLAR_MASS, linewidth=1, color="tab:blue", rasterized=True)

    # Add the quark star candidate error bars
    plt.errorbar(QS_CANDIDATE_R, QS_CANDIDATE_M, xerr=QS_CANDIDATE_R_ERROR, yerr=QS_CANDIDATE_M_ERROR, capsize=10, fmt="o", color="tab:orange")

    # Create the folder if necessary and save the figure
    os.makedirs(figures_path, exist_ok=True)
    figure_name = f"mass_vs_radius_curve.pdf"
    complete_path = os.path.join(figures_path, figure_name)
    plt.savefig(complete_path, bbox_inches="tight", dpi=400)

    # Show graph at the end
    plt.show()


def main():
    """Main logic
    """

    # Constants
    figures_path = "figures/app_quark_eos/extended_analysis"                                    # Path of the figures folder
    dataframe_csv_path = "results"                                                              # Path of the results folder
    dataframe_csv_name = "quark_eos_extended_analysis.csv"                                      # Name of the csv file with the results
    filtered_dataframe_csv_name = "quark_eos_extended_analysis_filtered.csv"                    # Name of the csv file with the results after filtering with observation data restrictions
    dictionary_json_path = "results"                                                            # Path of the results folder
    dictionary_json_name = "quark_eos_extended_analysis_parameters_limits.json"                 # Name of the json file with the parameters limits
    properties_dictionary_json_name = "quark_eos_extended_analysis_properties_limits.json"      # Name of the json file with the properties limits
    parameter_space_mesh_size = 2001                                                            # Number of points used in the meshgrid for the parameter space plot
    number_of_samples = 3 * 10**6                                                               # Number of samples used

    # Create the parameter space plot
    plot_parameter_space(parameter_space_mesh_size, figures_path)

    # Generate parameters for strange stars
    (a2_masked, a4_masked, B_masked, parameter_dataframe) = generate_strange_stars(number_of_samples)

    # Plot the parameter points generated for strange stars
    plot_parameter_points_scatter(a2_masked, a4_masked, B_masked, figures_path)

    # Analize the strange stars generated
    (parameter_dataframe, filtered_dataframe, parameters_limits, properties_limits) = analyze_strange_stars(parameter_dataframe)

    # Create all the analysis graphs
    plot_analysis_graphs(parameter_dataframe, parameters_limits, figures_path)

    # Create the combined mass vs radius graph
    plot_mass_radius_graph(filtered_dataframe, figures_path)

    # Save the dataframes to csv files
    dataframe_to_csv(dataframe=parameter_dataframe, file_path=dataframe_csv_path, file_name=dataframe_csv_name)
    dataframe_to_csv(dataframe=filtered_dataframe, file_path=dataframe_csv_path, file_name=filtered_dataframe_csv_name)

    # Save the dictionaries to json files
    dict_to_json(dictionary=parameters_limits, file_path=dictionary_json_path, file_name=dictionary_json_name)
    dict_to_json(dictionary=properties_limits, file_path=dictionary_json_path, file_name=properties_dictionary_json_name)


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)       # Force the multiprocessing start method to "spawn" on every environment
    main()
