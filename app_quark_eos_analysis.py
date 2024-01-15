import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from constants import UnitConversion as uconv
from eos_library import QuarkEOS
from star_family_structure import StarFamily


# Constants
g0 = 930.0                                                      # Gibbs free energy per baryon of quark matter at null pressure [MeV]
alpha = ((1 / 3) - 8 / (3 * (1 + 2**(1 / 3))**3)) * g0**2       # alpha coefficient of the a2_max vs a4 curve [MeV^2]
a2_min = 0.0                                                    # Minimum a2 parameter value [MeV^2]
a2_max = alpha                                                  # Maximum a2 parameter value [MeV^2]
a4_min = 0.0                                                    # Minimum a4 parameter value [dimensionless]
a4_max = 1.0                                                    # Maximum a4 parameter value [dimensionless]
B_min = 0.0                                                     # Minimum B parameter value [MeV^4]
B_max = g0**4 / (108 * np.pi**2)                                # Maximum B parameter value [MeV^4]


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


def generate_strange_stars(mesh_size=21):
    """Function that generates a list of meshgrids representing the parameters of strange stars.
    This function also creates a dataframe with the parameters of strange stars

    Args:
        mesh_size (int, optional): Size of the mesh used to represent the parameters. Defaults to 21

    Returns:
        Arrays of float: Masked meshgrids representing the parameters
        Pandas dataframe of float: Dataframe with the parameters of strange stars
    """

    # Define the (a2, a4, B) rectangular meshgrid
    a2_1_2_range = np.linspace(a2_min**(1 / 2), a2_max**(1 / 2), mesh_size)
    a2_range = a2_1_2_range**2
    a4_range = np.linspace(a4_min, a4_max, mesh_size)
    B_1_4_range = np.linspace(B_min**(1 / 4), B_max**(1 / 4), mesh_size)
    B_range = B_1_4_range**4
    (a2, a4, B) = np.meshgrid(a2_range, a4_range, B_range)

    # Create the mesh masks according to each parameter minimum and maximum allowed values
    a2_max_mesh_mask = (a2 >= alpha * a4)
    a2_min_mesh_mask = (a2 <= a2_min)
    a4_max_mesh_mask = (a4 > a4_max)
    a4_min_mesh_mask = (a4 <= a4_min)
    B_max_mesh_mask = (B >= calc_B_max(a2, a4))
    B_min_mesh_mask = (B <= calc_B_min(a2, a4))

    # Create the combined mask and apply to each mesh grid
    mesh_mask = a2_max_mesh_mask | a2_min_mesh_mask | a4_max_mesh_mask | a4_min_mesh_mask | B_max_mesh_mask | B_min_mesh_mask
    a2_masked = np.ma.masked_where(mesh_mask, a2)
    a4_masked = np.ma.masked_where(mesh_mask, a4)
    B_masked = np.ma.masked_where(mesh_mask, B)

    # Loop over the mask and store the parameter points of the strange stars in a dataframe
    iterator = np.nditer(mesh_mask, flags=['multi_index'])
    parameter_points = []
    for x in iterator:
        if bool(x) is False:
            index = iterator.multi_index
            star_parameters = (a2[index], a4[index], B[index])
            parameter_points.append(star_parameters)
    parameter_dataframe = pd.DataFrame(parameter_points, columns=["a2 [MeV^2]", "a4 [dimensionless]", "B [MeV^4]"])

    return (a2_masked, a4_masked, B_masked, parameter_dataframe)


def analyze_strange_stars(parameter_dataframe):

    # Pre-allocate the new dataframe columns with NaN
    parameter_dataframe['rho_center_max [g ⋅ cm^-3]'] = np.nan
    parameter_dataframe['M_max [solar mass]'] = np.nan

    # Iterate over each strange star
    n_rows = parameter_dataframe.shape[0]
    for row in parameter_dataframe.itertuples():

        # Unpack the row values
        (index, a2, a4, B, *_) = row

        # Print a message at the beginning to separate each star
        print(f"({index + 1} / {n_rows}) {'#' * 100}")

        # Create the EOS object
        quark_eos = QuarkEOS(a2, a4, B)

        # EOS analysis

        # Set the p_space
        max_rho = 1.0e16 * uconv.MASS_DENSITY_CGS_TO_GU     # Maximum density [m^-2]
        max_p = quark_eos.p(max_rho)                        # Maximum pressure [m^-2]
        p_space = max_p * np.logspace(-15.0, 0.0, 1000)

        # Check the EOS
        quark_eos.check_eos(p_space)

        # TOV analysis

        # Set the central pressure of the star
        rho_center = 1.0e16 * uconv.MASS_DENSITY_CGS_TO_GU      # Central density [m^-2]
        p_center = quark_eos.p(rho_center)                      # Central pressure [m^-2]

        # Set the p_center space that characterizes the star family
        p_center_space = p_center * np.logspace(-3.0, 0.0, 20)

        # Define the object
        star_family_object = StarFamily(quark_eos, p_center_space)

        # Find the maximum mass star and add the central density and mass to the dataframe
        star_family_object.find_maximum_mass()
        parameter_dataframe.at[index, 'rho_center_max [g ⋅ cm^-3]'] = star_family_object.maximum_stable_rho_center * uconv.MASS_DENSITY_GU_TO_CGS
        parameter_dataframe.at[index, 'M_max [solar mass]'] = star_family_object.maximum_mass * uconv.MASS_GU_TO_SOLAR_MASS

    # Print the parameter dataframe at the end
    print(parameter_dataframe)


def plot_parameter_points_scatter(a2, a4, B, figure_path="figures/app_quark_eos"):
    """Function that plots the scatter graph of the parameter points

    Args:
        a2 (3D array of float): Meshgrid with the a2 parameter of the EOS [MeV^2]
        a4 (3D array of float): Meshgrid with the a4 parameter of the EOS [dimensionless]
        B (3D array of float): Meshgrid with the B parameter of the EOS [MeV^4]
        figure_path (str, optional): Path used to save the figure. Defaults to "figures/app_quark_eos"
    """

    # Create figure and change properties
    (fig, ax) = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.0, 4.8), constrained_layout=True)
    ax.set_title("Quark EOS parameter points", y=1.0)
    ax.view_init(elev=15, azim=-115, roll=0)
    ax.set_xlim3d(a2_min**(1 / 2), a2_max**(1 / 2))
    ax.set_ylim3d(a4_min, a4_max)
    ax.set_zlim3d(B_min**(1 / 4), B_max**(1 / 4))
    ax.set_xlabel("$a_2^{1/2} ~ [MeV]$", fontsize=10, rotation=0)
    ax.set_ylabel("$a_4 ~ [dimensionless]$", fontsize=10, rotation=0)
    ax.set_zlabel("$B^{1/4} ~ [MeV]$", fontsize=10, rotation=90)
    ax.zaxis.set_rotate_label(False)

    # Add each scatter point
    ax.scatter(a2**(1 / 2), a4, B**(1 / 4))

    # Create the folder if necessary and save the figure
    os.makedirs(figure_path, exist_ok=True)
    figure_name = "quark_eos_parameter_points.png"
    complete_path = os.path.join(figure_path, figure_name)
    plt.savefig(complete_path)

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
    (fig, ax) = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.0, 4.8), constrained_layout=True)
    ax.set_title("Quark EOS parameter space for stable strange stars", y=1.0)
    ax.view_init(elev=15, azim=-115, roll=0)
    ax.set_xlim3d(a2_min**(1 / 2), a2_max**(1 / 2))
    ax.set_ylim3d(a4_min, a4_max)
    ax.set_zlim3d(B_min**(1 / 4), B_max**(1 / 4))
    ax.set_xlabel("$a_2^{1/2} ~ [MeV]$", fontsize=10, rotation=0)
    ax.set_ylabel("$a_4 ~ [dimensionless]$", fontsize=10, rotation=0)
    ax.set_zlabel("$B^{1/4} ~ [MeV]$", fontsize=10, rotation=90)
    ax.zaxis.set_rotate_label(False)

    # Add each surface plot
    a2_1_2_masked = a2_masked**(1 / 2)
    B_1_4_max_surface_masked = B_max_surface_masked**(1 / 4)
    B_1_4_min_surface_masked = B_min_surface_masked**(1 / 4)
    ax.plot_surface(a2_1_2_masked, a4_masked, B_1_4_max_surface_masked, cmap=cm.Reds, rstride=10, cstride=10, alpha=0.8, label="$B_{max}^{1/4}$")
    ax.plot_surface(a2_1_2_masked, a4_masked, B_1_4_min_surface_masked, cmap=cm.Blues, rstride=10, cstride=10, alpha=0.8, label="$B_{min}^{1/4}$")
    ax.legend(loc=(0.7, 0.25))

    # Add each contour plot (grey projections on each plane)
    ax.contourf(a2_1_2_masked, a4_masked, B_1_4_max_surface_masked, levels=0, zdir='x', offset=a2_max**(1 / 2), colors="gray", alpha=0.7, antialiased=True)
    ax.contourf(a2_1_2_masked, a4_masked, B_1_4_max_surface_masked, levels=0, zdir='y', offset=a4_max, colors="gray", alpha=0.7, antialiased=True)
    ax.contourf(a2_1_2_masked, a4_masked, B_1_4_max_surface_masked, levels=0, zdir='z', offset=0, colors="gray", alpha=0.7, antialiased=True)

    # Create the folder if necessary and save the figure
    os.makedirs(figure_path, exist_ok=True)
    figure_name = "quark_eos_parameter_space.png"
    complete_path = os.path.join(figure_path, figure_name)
    plt.savefig(complete_path)

    # Show graph
    plt.show()


def main():
    """Main logic
    """

    # Constants
    figure_path = "figures/app_quark_eos"
    parameter_space_mesh_size = 1001        # Number of points used in the meshgrid for the parameter space plot
    scatter_plot_mesh_size = 11             # Number of points used in the meshgrid for the scatter plot

    # Create the parameter space plot
    plot_parameter_space(parameter_space_mesh_size, figure_path)

    # Generate parameters for strange stars
    (a2_masked, a4_masked, B_masked, parameter_dataframe) = generate_strange_stars(scatter_plot_mesh_size)

    # Plot the parameter points generated for strange stars
    plot_parameter_points_scatter(a2_masked, a4_masked, B_masked, figure_path)

    # Analize the strange stars generated
    analyze_strange_stars(parameter_dataframe)


# This logic is only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
