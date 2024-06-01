import os
import math
import pprint
import psutil
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm.contrib.concurrent import process_map
from constants import Constants as const
from constants import UnitConversion as uconv
from data_handling import dataframe_to_csv, dict_to_json
from eos_library import QuarkEOS, SLy4EOS, HybridEOS
from star_family_tides import DeformedStarFamily


# Constants
g0 = 930.0                                                      # Gibbs free energy per baryon of quark matter at null pressure [MeV]
alpha = ((1 / 3) - 8 / (3 * (1 + 2**(1 / 3))**3)) * g0**2       # alpha coefficient of the a2_3f_max vs a4 curve, used in the parameter space graph [MeV^2]
beta = (4 / (3 * (1 + 2**(1 / 3))**3)) * g0**2                  # beta coefficient of the a2_2f_max vs a4 curve, used in the parameter space graph [MeV^2]
a2_min = 0.0                                                    # Minimum a2 parameter value [MeV^2]
a2_max = 300**2                                                 # Maximum a2 parameter value [MeV^2]
a4_min = 0.0                                                    # Minimum a4 parameter value [dimensionless]
a4_max = 1.0                                                    # Maximum a4 parameter value [dimensionless]
B_min = 0.0                                                     # Minimum B parameter value [MeV^4]
B_max = 180**4                                                  # Maximum B parameter value [MeV^4]
MAX_RHO = 1e16 * uconv.MASS_DENSITY_CGS_TO_GU                   # Maximum density [m^-2]
EOS_LOGSPACE = np.logspace(-15.0, 0.0, 10000)                   # Logspace used to create the EOS
STARS_LOGSPACE = np.logspace(-2.0, 0.0, 20)                     # Logspace used to create the star family
HADRON_EOS_PROPERTIES = {                                       # Hadron EOS properties in GU, used to avoid calculations when the Hybrid EOS turns out to be a Hadron EOS
    "maximum_stable_rho_center": 2.121779980021422e-09,
    "maximum_mass": 3024.21699470999,
    "maximum_cs": 0.9880562481221495,
    "canonical_rho_center": 7.31938864684075e-10,
    "canonical_radius": 11699.848065179254,
    "canonical_lambda": 297.0426007982706,
    "maximum_k2_star_rho_center": 4.673405942362452e-10,
    "maximum_k2": 0.10564216488608084,
}

# Observation data
M_max_inf_limit = 2.13                                          # Inferior limit of the maximum mass [solar mass] (Romani - 2 sigma)
R_canonical_inf_limit = 10.0                                    # Inferior limit of the radius of the canonical star [km]
R_canonical_sup_limit = 13.25                                   # Superior limit of the radius of the canonical star [km]
Lambda_canonical_sup_limit = 970.0                              # Superior limit of the tidal deformability of the canonical star [dimensionless] (Abbott - 2 sigma)

# EOS type limits (hadron = 0 / hybrid = 1 / quark = 2)
eos_type_inf_limit = 0.5                                        # Inferior limit of the EOS type
eos_type_sup_limit = 1.5                                        # Inferior limit of the EOS type


def calc_B_3f_lim(a2, a4):
    """Function that calculates the 3-flavor limit of the B parameter

    Args:
        a2 (array of float): a2 parameter of the EOS [MeV^2]
        a4 (array of float): a4 parameter of the EOS [dimensionless]

    Returns:
        array of float: 3-flavor limit of B [MeV^4]
    """
    return (g0**2 / (108 * np.pi**2)) * (g0**2 * a4 - 9 * a2)


def calc_B_2f_lim(a2, a4):
    """Function that calculates the 2-flavor limit of the B parameter

    Args:
        a2 (array of float): a2 parameter of the EOS [MeV^2]
        a4 (array of float): a4 parameter of the EOS [dimensionless]

    Returns:
        array of float: 2-flavor limit of B [MeV^4]
    """
    return (g0**2 / (54 * np.pi**2)) * ((4 * g0**2 * a4) / ((1 + 2**(1 / 3))**3) - 3 * a2)


def calc_a4_2f_lim(a2, B):
    """Function that calculates the 2-flavor limit of the a4 parameter

    Args:
        a2 (array of float): a2 parameter of the EOS [MeV^2]
        B (array of float): B parameter of the EOS [MeV^4]

    Returns:
        array of float: 2-flavor limit of a4 [dimensionless]
    """
    return (((1 + 2**(1 / 3))**3) / (4 * g0**2)) * (((54 * np.pi**2) / g0**2) * B + 3 * a2)


def generate_hybrid_stars(mesh_size=21):
    """Function that generates a list of meshgrids representing the parameters of hybrid stars.
    This function also creates a dataframe with the parameters of hybrid stars

    Args:
        mesh_size (int, optional): Size of the mesh used to represent the parameters. Defaults to 21

    Returns:
        Arrays of float: Masked meshgrids representing the parameters
        Pandas dataframe of float: Dataframe with the parameters of hybrid stars
    """

    # Define the (a2, a4, B) rectangular random meshgrid using Latin Hypercube sampler
    seed_value = 123                                            # Fix the seed value to generate the same pseudo-random values each time
    sampler = qmc.LatinHypercube(d=3, seed=seed_value)          # Set the sampler
    samples = sampler.random(n=mesh_size**3)                    # Create the samples
    l_bounds = [a2_min**(1 / 2), a4_min, B_min**(1 / 4)]
    u_bounds = [a2_max**(1 / 2), a4_max, B_max**(1 / 4)]
    scaled_samples = qmc.scale(samples, l_bounds, u_bounds)     # Scale the samples
    (a2, a4, B) = (scaled_samples[:, 0]**2, scaled_samples[:, 1], scaled_samples[:, 2]**4)

    # Create the mesh masks according to each parameter minimum and maximum allowed values
    a2_max_mesh_mask = (a2 > a2_max)
    a2_min_mesh_mask = (a2 <= a2_min)
    a4_max_mesh_mask = (a4 > a4_max)
    a4_min_mesh_mask = (a4 <= a4_min)
    B_max_mesh_mask = (B > B_max)
    B_min_mesh_mask = (B <= B_min) | (B <= calc_B_2f_lim(a2, a4)) | (B <= calc_B_3f_lim(a2, a4))

    # Create the combined mask and apply to each mesh grid
    mesh_mask = a2_max_mesh_mask | a2_min_mesh_mask | a4_max_mesh_mask | a4_min_mesh_mask | B_max_mesh_mask | B_min_mesh_mask
    a2_masked = np.ma.masked_where(mesh_mask, a2)
    a4_masked = np.ma.masked_where(mesh_mask, a4)
    B_masked = np.ma.masked_where(mesh_mask, B)

    # Loop over the mask and store the parameter points of the hybrid stars in a dataframe
    iterator = np.nditer(mesh_mask, flags=["multi_index"])
    parameter_points = []
    for x in iterator:
        if bool(x) is False:
            index = iterator.multi_index
            star_parameters = (a2[index]**(1 / 2), a4[index], B[index]**(1 / 4))
            parameter_points.append(star_parameters)
    parameter_dataframe = pd.DataFrame(parameter_points, columns=["a2^(1/2) [MeV]", "a4 [dimensionless]", "B^(1/4) [MeV]"])

    return (a2_masked, a4_masked, B_masked, parameter_dataframe)


def analyze_hybrid_star_family(dataframe_row):
    """Function that analyzes a star family, calculating the properties

    Args:
        dataframe_row (list): Row of the dataframe with the index and hybrid EOS parameters

    Returns:
        tuple: Tuple with index and star family properties calculated
    """

    # Unpack the row values
    (index, a2_1_2, a4, B_1_4, *_) = dataframe_row
    a2 = a2_1_2**2
    B = B_1_4**4

    # Create the QuarkEOS object
    quark_eos = QuarkEOS(a2, a4, B)

    # Create the HadronEOS object
    sly4_eos_rho_space = MAX_RHO * EOS_LOGSPACE
    sly4_eos = SLy4EOS(sly4_eos_rho_space)

    # Create the HybridEOS object
    sly4_maximum_stable_rho_center = 2.865e15 * uconv.MASS_DENSITY_CGS_TO_GU
    hybrid_eos = HybridEOS(quark_eos, sly4_eos, "data/SLy4_EOS.csv", sly4_maximum_stable_rho_center, "HybridSLy4EOS")

    # EOS analysis

    # Set the p_space
    if hybrid_eos.is_hadron_eos:
        max_p = hybrid_eos.p_max
    else:
        max_rho = MAX_RHO                   # Maximum density [m^-2]
        max_p = hybrid_eos.p(max_rho)       # Maximum pressure [m^-2]
    p_space = max_p * EOS_LOGSPACE

    # Check the EOS
    hybrid_eos.check_eos(p_space, debug_msg=False)

    # Get the surface pressure and minimum sound speed
    rho_surface = hybrid_eos.rho_min
    if hybrid_eos.is_hybrid_eos is True:
        # Sound speed is zero at the phase transition
        minimum_cs = 0.0
    else:
        minimum_cs = hybrid_eos.c_s(rho_surface)

    # Get the EOS type (hadron = 0 / hybrid = 1 / quark = 2)
    if hybrid_eos.is_hadron_eos is True:
        eos_type = 0
    elif hybrid_eos.is_hybrid_eos is True:
        eos_type = 1
    else:
        eos_type = 2

    # TOV and tidal analysis

    # Check if the EOS is a Hadron EOS
    if hybrid_eos.is_hadron_eos is True:

        # Get the EOS properties from the constant dict to avoid unnecessary calculations
        maximum_stable_rho_center = HADRON_EOS_PROPERTIES["maximum_stable_rho_center"]
        maximum_mass = HADRON_EOS_PROPERTIES["maximum_mass"]
        maximum_cs = HADRON_EOS_PROPERTIES["maximum_cs"]
        canonical_rho_center = HADRON_EOS_PROPERTIES["canonical_rho_center"]
        canonical_radius = HADRON_EOS_PROPERTIES["canonical_radius"]
        canonical_lambda = HADRON_EOS_PROPERTIES["canonical_lambda"]
        maximum_k2_star_rho_center = HADRON_EOS_PROPERTIES["maximum_k2_star_rho_center"]
        maximum_k2 = HADRON_EOS_PROPERTIES["maximum_k2"]

    # Calculate the EOS properties if it is a Hybrid EOS or a Quark EOS
    else:

        # Set the central pressure of the star
        p_center = max_p        # Central pressure [m^-2]

        # Set the p_center space that characterizes the star family
        p_center_space = p_center * STARS_LOGSPACE

        # Create the star family object
        star_family_object = DeformedStarFamily(hybrid_eos, p_center_space)

        # Find the maximum mass star
        star_family_object.find_maximum_mass_star()
        maximum_stable_rho_center = star_family_object.maximum_stable_rho_center
        maximum_mass = star_family_object.maximum_mass
        maximum_cs = hybrid_eos.c_s(maximum_stable_rho_center)

        # Find the canonical star
        star_family_object.find_canonical_star()
        canonical_rho_center = star_family_object.canonical_rho_center
        canonical_radius = star_family_object.canonical_radius
        canonical_lambda = star_family_object.canonical_lambda

        # Find the maximum k2 star
        star_family_object.find_maximum_k2_star()
        maximum_k2_star_rho_center = star_family_object.maximum_k2_star_rho_center
        maximum_k2 = star_family_object.maximum_k2

    # Return the index and results
    return (index, eos_type, rho_surface, minimum_cs, maximum_stable_rho_center, maximum_mass, maximum_cs, canonical_rho_center, canonical_radius, canonical_lambda, maximum_k2_star_rho_center, maximum_k2)


def analyze_hybrid_stars(parameter_dataframe):
    """Function that analyzes the hybrid stars given by the parameters in the dataframe, calculating the properties

    Args:
        parameter_dataframe (Pandas dataframe of float): Dataframe with the parameters of hybrid stars

    Returns:
        Pandas dataframe of float: Dataframe with the parameters and properties of hybrid stars
        Pandas dataframe of float: Dataframe with the parameters and properties of hybrid stars filtered by the observation data restrictions
        dict: Dictionary with the minimum and maximum values of the parameters for each observation data restrictions
        dict: Dictionary with the minimum and maximum values of the properties of hybrid stars
    """

    # Calculate the number of rows, number of processes and number of calculations per process (chunksize)
    n_rows = parameter_dataframe.shape[0]
    processes = psutil.cpu_count(logical=False)             # Number of processes are equal to the number of hardware cores
    chunksize = math.ceil(n_rows / (processes * 10))        # Create the chunksize smaller to provide some feedback on progress

    # Create a list with the rows of the dataframe
    rows_list = [list(row) for row in parameter_dataframe.itertuples()]

    # Check if DEBUG is activated
    if const.DEBUG is True:
        # Execute the analysis for each row sequentially
        results = [analyze_hybrid_star_family(row) for row in rows_list]
    else:
        # Execute the analysis for each row in parallel processes, using a progress bar from tqdm
        results = process_map(analyze_hybrid_star_family, rows_list, max_workers=processes, chunksize=chunksize)

    # Update the dataframe with the results
    for index, eos_type, rho_surface, minimum_cs, maximum_stable_rho_center, maximum_mass, maximum_cs, canonical_rho_center, canonical_radius, canonical_lambda, maximum_k2_star_rho_center, maximum_k2 in results:
        parameter_dataframe.at[index, "eos_type [0, 1, or 2]"] = eos_type
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
    eos_type_query = f"`eos_type [0, 1, or 2]` > {eos_type_inf_limit} & `eos_type [0, 1, or 2]` < {eos_type_sup_limit}"
    M_max_query = f"`M_max [solar mass]` > {M_max_inf_limit}"
    R_canonical_query = f"`R_canonical [km]` > {R_canonical_inf_limit} & `R_canonical [km]` < {R_canonical_sup_limit}"
    Lambda_canonical_query = f"`Lambda_canonical [dimensionless]` < {Lambda_canonical_sup_limit}"
    combined_query = f"{eos_type_query} & {M_max_query} & {R_canonical_query} & {Lambda_canonical_query}"
    filtered_eos_type_dataframe = parameter_dataframe.query(eos_type_query)
    filtered_M_max_dataframe = parameter_dataframe.query(M_max_query)
    filtered_R_canonical_dataframe = parameter_dataframe.query(R_canonical_query)
    filtered_Lambda_canonical_dataframe = parameter_dataframe.query(Lambda_canonical_query)
    filtered_dataframe = parameter_dataframe.query(combined_query)

    # Create a dictionary with the minimum and maximum values of the parameters for each observation data restrictions
    parameters_limits = {
        "a2^(1/2)": {
            "eos_type": (np.min(filtered_eos_type_dataframe.loc[:, "a2^(1/2) [MeV]"]), np.max(filtered_eos_type_dataframe.loc[:, "a2^(1/2) [MeV]"])),
            "M_max": (np.min(filtered_M_max_dataframe.loc[:, "a2^(1/2) [MeV]"]), np.max(filtered_M_max_dataframe.loc[:, "a2^(1/2) [MeV]"])),
            "R_canonical": (np.min(filtered_R_canonical_dataframe.loc[:, "a2^(1/2) [MeV]"]), np.max(filtered_R_canonical_dataframe.loc[:, "a2^(1/2) [MeV]"])),
            "Lambda_canonical": (np.min(filtered_Lambda_canonical_dataframe.loc[:, "a2^(1/2) [MeV]"]), np.max(filtered_Lambda_canonical_dataframe.loc[:, "a2^(1/2) [MeV]"])),
        },
        "a4": {
            "eos_type": (np.min(filtered_eos_type_dataframe.loc[:, "a4 [dimensionless]"]), np.max(filtered_eos_type_dataframe.loc[:, "a4 [dimensionless]"])),
            "M_max": (np.min(filtered_M_max_dataframe.loc[:, "a4 [dimensionless]"]), np.max(filtered_M_max_dataframe.loc[:, "a4 [dimensionless]"])),
            "R_canonical": (np.min(filtered_R_canonical_dataframe.loc[:, "a4 [dimensionless]"]), np.max(filtered_R_canonical_dataframe.loc[:, "a4 [dimensionless]"])),
            "Lambda_canonical": (np.min(filtered_Lambda_canonical_dataframe.loc[:, "a4 [dimensionless]"]), np.max(filtered_Lambda_canonical_dataframe.loc[:, "a4 [dimensionless]"])),
        },
        "B^(1/4)": {
            "eos_type": (np.min(filtered_eos_type_dataframe.loc[:, "B^(1/4) [MeV]"]), np.max(filtered_eos_type_dataframe.loc[:, "B^(1/4) [MeV]"])),
            "M_max": (np.min(filtered_M_max_dataframe.loc[:, "B^(1/4) [MeV]"]), np.max(filtered_M_max_dataframe.loc[:, "B^(1/4) [MeV]"])),
            "R_canonical": (np.min(filtered_R_canonical_dataframe.loc[:, "B^(1/4) [MeV]"]), np.max(filtered_R_canonical_dataframe.loc[:, "B^(1/4) [MeV]"])),
            "Lambda_canonical": (np.min(filtered_Lambda_canonical_dataframe.loc[:, "B^(1/4) [MeV]"]), np.max(filtered_Lambda_canonical_dataframe.loc[:, "B^(1/4) [MeV]"])),
        },
    }

    # Add the combined limits to the dictionary
    a2_1_2_min = np.max([parameters_limits["a2^(1/2)"]["eos_type"][0], parameters_limits["a2^(1/2)"]["M_max"][0], parameters_limits["a2^(1/2)"]["R_canonical"][0], parameters_limits["a2^(1/2)"]["Lambda_canonical"][0]])
    a2_1_2_max = np.min([parameters_limits["a2^(1/2)"]["eos_type"][1], parameters_limits["a2^(1/2)"]["M_max"][1], parameters_limits["a2^(1/2)"]["R_canonical"][1], parameters_limits["a2^(1/2)"]["Lambda_canonical"][1]])
    a4_min = np.max([parameters_limits["a4"]["eos_type"][0], parameters_limits["a4"]["M_max"][0], parameters_limits["a4"]["R_canonical"][0], parameters_limits["a4"]["Lambda_canonical"][0]])
    a4_max = np.min([parameters_limits["a4"]["eos_type"][1], parameters_limits["a4"]["M_max"][1], parameters_limits["a4"]["R_canonical"][1], parameters_limits["a4"]["Lambda_canonical"][1]])
    B_1_4_min = np.max([parameters_limits["B^(1/4)"]["eos_type"][0], parameters_limits["B^(1/4)"]["M_max"][0], parameters_limits["B^(1/4)"]["R_canonical"][0], parameters_limits["B^(1/4)"]["Lambda_canonical"][0]])
    B_1_4_max = np.min([parameters_limits["B^(1/4)"]["eos_type"][1], parameters_limits["B^(1/4)"]["M_max"][1], parameters_limits["B^(1/4)"]["R_canonical"][1], parameters_limits["B^(1/4)"]["Lambda_canonical"][1]])
    parameters_limits["a2^(1/2)"]["combined"] = (a2_1_2_min, a2_1_2_max)
    parameters_limits["a4"]["combined"] = (a4_min, a4_max)
    parameters_limits["B^(1/4)"]["combined"] = (B_1_4_min, B_1_4_max)

    # Create a dictionary with the minimum and maximum values of the properties of hybrid stars
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


def plot_parameter_points_scatter(a2, a4, B, figure_path="figures/app_hybrid_eos"):
    """Function that plots the scatter graph of the parameter points

    Args:
        a2 (3D array of float): Meshgrid with the a2 parameter of the EOS [MeV^2]
        a4 (3D array of float): Meshgrid with the a4 parameter of the EOS [dimensionless]
        B (3D array of float): Meshgrid with the B parameter of the EOS [MeV^4]
        figure_path (str, optional): Path used to save the figure. Defaults to "figures/app_hybrid_eos"
    """

    # Create figure and change properties
    (fig, ax) = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5.0, 4.0), constrained_layout=True)
    ax.view_init(elev=15, azim=-115, roll=0)
    ax.set_xlim3d(a2_min**(1 / 2), a2_max**(1 / 2))
    ax.set_ylim3d(a4_min, a4_max)
    ax.set_zlim3d(B_min**(1 / 4), B_max**(1 / 4))
    ax.set_xlabel("$a_2^{1/2} ~ [MeV]$", fontsize=10, rotation=0)
    ax.set_ylabel("$a_4 ~ [dimensionless]$", fontsize=10, rotation=0)
    ax.set_zlabel("$B^{1/4} ~ [MeV]$", fontsize=10, rotation=90)
    ax.zaxis.set_rotate_label(False)
    ax.set_position([0.0, -0.05, 1.05, 1.2])        # Adjust plot position and size inside image to remove excessive whitespaces

    # Add each scatter point
    ax.scatter(a2**(1 / 2), a4, B**(1 / 4), s=2.5**2)

    # Create the folder if necessary and save the figure
    os.makedirs(figure_path, exist_ok=True)
    figure_name = "hybrid_eos_parameter_points.pdf"
    complete_path = os.path.join(figure_path, figure_name)
    plt.savefig(complete_path)

    # Show graph
    plt.show()


def plot_parameter_space(mesh_size=1000, figure_path="figures/app_hybrid_eos"):
    """Function that plots the graph of the parameter space

    Args:
        mesh_size (int, optional): Size of the mesh used to create the plot. Defaults to 1000
        figure_path (str, optional): Path used to save the figure. Defaults to "figures/app_hybrid_eos"
    """

    # Define the (a2, a4) and (a2, B) rectangular meshgrids, using nonlinear scales to make the graph look better
    a2_1_2_range = np.linspace(a2_min**(1 / 2), a2_max**(1 / 2), mesh_size)
    a2_range = a2_1_2_range**2
    a4_1_2_range = np.linspace(a4_min**(1 / 2), a4_max**(1 / 2), mesh_size)
    a4_range = a4_1_2_range**2
    B_1_4_range = np.linspace(B_min**(1 / 4), B_max**(1 / 4), mesh_size)
    B_range = B_1_4_range**4
    (a2, a4) = np.meshgrid(a2_range, a4_range)
    (a2, B) = np.meshgrid(a2_range, B_range)

    # Create the B_3f_lim and a4_2f_lim surfaces. Using a4 surface for the 2f limit to make the graph look better
    B_3f_lim_surface = calc_B_3f_lim(a2, a4)
    a4_2f_lim_surface = calc_a4_2f_lim(a2, B)

    # Apply the mask to the meshgrids
    mesh_mask_3f = (a2 > alpha * a4)
    B_2f_3f_line = calc_B_3f_lim(a2, a2 / alpha)
    mesh_mask_2f = ((B < B_min) | (B > B_2f_3f_line))
    a2_3f_masked = np.ma.masked_where(mesh_mask_3f, a2)
    a2_2f_masked = np.ma.masked_where(mesh_mask_2f, a2)
    a4_3f_masked = np.ma.masked_where(mesh_mask_3f, a4)
    a4_2f_lim_surface_masked = np.ma.masked_where(mesh_mask_2f, a4_2f_lim_surface)
    B_3f_lim_surface_masked = np.ma.masked_where(mesh_mask_3f, B_3f_lim_surface)
    B_2f_masked = np.ma.masked_where(mesh_mask_2f, B)

    # Create figure and change properties
    (fig, ax) = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5.0, 4.0), constrained_layout=True)
    ax.view_init(elev=15, azim=-115, roll=0)
    ax.set_xlim3d(a2_min**(1 / 2), a2_max**(1 / 2))
    ax.set_ylim3d(a4_min, a4_max)
    ax.set_zlim3d(B_min**(1 / 4), B_max**(1 / 4))
    ax.set_xlabel("$a_2^{1/2} ~ [MeV]$", fontsize=10, rotation=0)
    ax.set_ylabel("$a_4 ~ [dimensionless]$", fontsize=10, rotation=0)
    ax.set_zlabel("$B^{1/4} ~ [MeV]$", fontsize=10, rotation=90)
    ax.zaxis.set_rotate_label(False)
    ax.set_position([0.0, -0.05, 1.05, 1.2])        # Adjust plot position and size inside image to remove excessive whitespaces

    # Add each surface plot
    a2_1_2_3f_masked = a2_3f_masked**(1 / 2)
    a2_1_2_2f_masked = a2_2f_masked**(1 / 2)
    B_1_4_3f_lim_surface_masked = B_3f_lim_surface_masked**(1 / 4)
    B_1_4_2f_masked = B_2f_masked**(1 / 4)
    ax.plot_surface(a2_1_2_3f_masked, a4_3f_masked, B_1_4_3f_lim_surface_masked, cmap=cm.Reds, rstride=10, cstride=10, alpha=0.8, label="$B_{3f}^{1/4}$")
    ax.plot_surface(a2_1_2_2f_masked, a4_2f_lim_surface_masked, B_1_4_2f_masked, cmap=cm.Blues, rstride=10, cstride=10, alpha=0.8, label="$B_{2f}^{1/4}$")
    ax.legend(loc=(0.7, 0.25))

    # Add each contour plot (grey projections on each plane)
    # B_3f_lim surface
    ax.contourf(a2_1_2_3f_masked, a4_3f_masked, B_1_4_3f_lim_surface_masked, levels=0, zdir="x", offset=a2_max**(1 / 2), colors="gray", alpha=0.7, antialiased=True)
    ax.contourf(a2_1_2_3f_masked, a4_3f_masked, B_1_4_3f_lim_surface_masked, levels=0, zdir="y", offset=a4_max, colors="gray", alpha=0.7, antialiased=True)
    ax.contourf(a2_1_2_3f_masked, a4_3f_masked, B_1_4_3f_lim_surface_masked, levels=0, zdir="z", offset=0, colors="gray", alpha=0.7, antialiased=True)
    # B_2f_lim surface
    ax.contourf(a2_1_2_2f_masked, a4_2f_lim_surface_masked, B_1_4_2f_masked, levels=0, zdir="x", offset=a2_max**(1 / 2), colors="gray", alpha=0.7, antialiased=True)
    ax.contourf(a2_1_2_2f_masked, a4_2f_lim_surface_masked, B_1_4_2f_masked, levels=0, zdir="y", offset=a4_max, colors="gray", alpha=0.7, antialiased=True)
    ax.contourf(a2_1_2_2f_masked, a4_2f_lim_surface_masked, B_1_4_2f_masked, levels=0, zdir="z", offset=0, colors="gray", alpha=0.7, antialiased=True)

    # Create the folder if necessary and save the figure
    os.makedirs(figure_path, exist_ok=True)
    figure_name = "hybrid_eos_parameter_space.pdf"
    complete_path = os.path.join(figure_path, figure_name)
    plt.savefig(complete_path)

    # Show graph
    plt.show()


def plot_analysis_graphs(parameter_dataframe, parameters_limits, figures_path="figures/app_hybrid_eos/analysis"):
    """Function that creates all the analysis graphs

    Args:
        parameter_dataframe (Pandas dataframe of float): Dataframe with the parameters of hybrid stars
        parameters_limits (dict): Dictionary with the minimum and maximum values of the parameters for each observation data restrictions
        figures_path (str, optional): Path used to save the figures. Defaults to "figures/app_hybrid_eos/analysis"
    """

    # Create a dictionary with all the functions used in plotting, with each name and label description
    plot_dict = {
        "a2^(1/2)": {
            "name": "a2^(1/2)",
            "label": "$a_2^{1/2} ~ [MeV]$",
            "value": parameter_dataframe.loc[:, "a2^(1/2) [MeV]"],
        },
        "a4": {
            "name": "a4",
            "label": "$a_4 ~ [dimensionless]$",
            "value": parameter_dataframe.loc[:, "a4 [dimensionless]"],
        },
        "B^(1/4)": {
            "name": "B^(1/4)",
            "label": "$B^{1/4} ~ [MeV]$",
            "value": parameter_dataframe.loc[:, "B^(1/4) [MeV]"],
        },
        "eos_type": {
            "name": "EOS type",
            "label": "EOS type [0 (hadron), 1 (hybrid), or 2 (quark)]",
            "value": parameter_dataframe.loc[:, "eos_type [0, 1, or 2]"],
            "inf_limit": eos_type_inf_limit,
            "sup_limit": eos_type_sup_limit,
        },
        "M_max": {
            "name": "Maximum mass",
            "label": "$M_{max} ~ [M_{\\odot}]$",
            "value": parameter_dataframe.loc[:, "M_max [solar mass]"],
            "inf_limit": M_max_inf_limit,
            "sup_limit": None,
        },
        "R_canonical": {
            "name": "Canonical radius",
            "label": "$R_{canonical} ~ [km]$",
            "value": parameter_dataframe.loc[:, "R_canonical [km]"],
            "inf_limit": R_canonical_inf_limit,
            "sup_limit": R_canonical_sup_limit,
        },
        "Lambda_canonical": {
            "name": "Canonical deformability",
            "label": "$\\log_{10} \\left( \\Lambda_{canonical} [dimensionless] \\right)$",
            "value": np.log10(parameter_dataframe.loc[:, "Lambda_canonical [dimensionless]"]),
            "inf_limit": None,
            "sup_limit": np.log10(Lambda_canonical_sup_limit),
        },
    }

    # Create a list with all the graphs to be plotted
    graphs_list = [
        ["a2^(1/2)", "eos_type"],
        ["a4", "eos_type"],
        ["B^(1/4)", "eos_type"],
        ["a2^(1/2)", "M_max"],
        ["a4", "M_max"],
        ["B^(1/4)", "M_max"],
        ["a2^(1/2)", "R_canonical"],
        ["a4", "R_canonical"],
        ["B^(1/4)", "R_canonical"],
        ["a2^(1/2)", "Lambda_canonical"],
        ["a4", "Lambda_canonical"],
        ["B^(1/4)", "Lambda_canonical"],
    ]

    # Plot all graphs specified in graphs_list
    for (x_axis, y_axis) in graphs_list:

        # Create the plot
        plt.figure(figsize=(6.0, 4.5))
        plt.scatter(plot_dict[x_axis]["value"], plot_dict[y_axis]["value"], s=2.5**2, zorder=2)
        plt.xlabel(plot_dict[x_axis]["label"], fontsize=10)
        plt.ylabel(plot_dict[y_axis]["label"], fontsize=10)

        # Add the y axis limit lines and text
        xlim0, xlim1 = plt.xlim()
        if plot_dict[y_axis]["sup_limit"] is not None:
            plt.axhline(y=plot_dict[y_axis]["sup_limit"], linewidth=1, color="#d62728", zorder=3)
            text = f" {plot_dict[y_axis]["sup_limit"]:.2f} "
            plt.text(xlim1, plot_dict[y_axis]["sup_limit"], text, horizontalalignment="left", verticalalignment="center", color="#d62728")
        if plot_dict[y_axis]["inf_limit"] is not None:
            plt.axhline(y=plot_dict[y_axis]["inf_limit"], linewidth=1, color="#2ca02c", zorder=3)
            text = f" {plot_dict[y_axis]["inf_limit"]:.2f} "
            plt.text(xlim1, plot_dict[y_axis]["inf_limit"], text, horizontalalignment="left", verticalalignment="center", color="#2ca02c")

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
        plt.savefig(complete_path, bbox_inches="tight")

    # Show graphs at the end
    plt.show()


def main():
    """Main logic
    """

    # Constants
    figures_path = "figures/app_hybrid_sly4_eos/analysis"                           # Path of the figures folder
    dataframe_csv_path = "results"                                                  # Path of the results folder
    dataframe_csv_name = "hybrid_sly4_eos_analysis.csv"                             # Name of the csv file with the results
    filtered_dataframe_csv_name = "hybrid_sly4_eos_analysis_filtered.csv"           # Name of the csv file with the results after filtering with observation data restrictions
    dictionary_json_path = "results"                                                # Path of the results folder
    dictionary_json_name = "hybrid_sly4_eos_parameters_limits.json"                 # Name of the json file with the parameters limits
    properties_dictionary_json_name = "hybrid_sly4_eos_properties_limits.json"      # Name of the json file with the properties limits
    parameter_space_mesh_size = 2001                                                # Number of points used in the meshgrid for the parameter space plot
    scatter_plot_mesh_size = 11                                                     # Number of points used in the meshgrid for the scatter plot

    # Create the parameter space plot
    plot_parameter_space(parameter_space_mesh_size, figures_path)

    # Generate parameters for hybrid stars
    (a2_masked, a4_masked, B_masked, parameter_dataframe) = generate_hybrid_stars(scatter_plot_mesh_size)

    # Plot the parameter points generated for hybrid stars
    plot_parameter_points_scatter(a2_masked, a4_masked, B_masked, figures_path)

    # Analize the hybrid stars generated
    (parameter_dataframe, filtered_dataframe, parameters_limits, properties_limits) = analyze_hybrid_stars(parameter_dataframe)

    # Create all the analysis graphs
    plot_analysis_graphs(parameter_dataframe, parameters_limits, figures_path)

    # Save the dataframes to csv files
    dataframe_to_csv(dataframe=parameter_dataframe, file_path=dataframe_csv_path, file_name=dataframe_csv_name)
    dataframe_to_csv(dataframe=filtered_dataframe, file_path=dataframe_csv_path, file_name=filtered_dataframe_csv_name)

    # Save the dictionaries to json files
    dict_to_json(dictionary=parameters_limits, file_path=dictionary_json_path, file_name=dictionary_json_name)
    dict_to_json(dictionary=properties_limits, file_path=dictionary_json_path, file_name=properties_dictionary_json_name)


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
