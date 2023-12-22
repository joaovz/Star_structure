import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def calc_alpha(g0):
    """Function that calculates the alpha coefficient of the a2_max vs a4 curve

    Args:
        g0 (float): Gibbs free energy per baryon of quark matter at null pressure [MeV]

    Returns:
        float: alpha coefficient of the a2_max vs a4 curve [MeV^2]
    """
    alpha = ((1 / 3) - 8 / (3 * (1 + 2**(1 / 3))**3)) * g0**2
    return alpha

def calc_B_max(g0, a2, a4):
    """Function that calculates the maximum B value

    Args:
        g0 (float): Gibbs free energy per baryon of quark matter at null pressure [MeV]
        a2 (array of float): a2 parameter of the EOS [MeV^2]
        a4 (array of float): a4 parameter of the EOS [dimensionless]

    Returns:
        array of float: Maximum B value [MeV^4]
    """
    B_max = (g0**2 / (108 * np.pi**2)) * (g0**2 * a4 - 9 * a2)
    return B_max

def calc_B_min(g0, a2, a4):
    """Function that calculates the minimum B value

    Args:
        g0 (float): Gibbs free energy per baryon of quark matter at null pressure [MeV]
        a2 (array of float): a2 parameter of the EOS [MeV^2]
        a4 (array of float): a4 parameter of the EOS [dimensionless]

    Returns:
        array of float: Minimum B value [MeV^4]
    """
    B_min = (g0**2 / (54 * np.pi**2)) * ((4 * g0**2 * a4) / ((1 + 2**(1 / 3))**3) - 3 * a2)
    return B_min

def plot_parameter_space(figure_path="figures/app_quark_eos", mesh_size=1000):
    """Function that plots the graph of the parameter space

    Args:
        figure_path (str, optional): Path used to save the figure. Defaults to "figures/app_quark_eos"
        mesh_size (int, optional): Size of the mesh used to create the plot. Defaults to 1000
    """

    # Constants
    g0 = 930.0                      # Gibbs free energy per baryon of quark matter at null pressure [MeV]
    a4_max = 1.0                    # Maximum a4 value [dimensionless]
    alpha = calc_alpha(g0)          # Coefficient of the a2_max vs a4 curve [MeV^2]
    a2_max = alpha * a4_max         # Maximum a2 value [MeV^2]

    # Define the (a4, a2) rectangular meshgrid
    a4_range = np.linspace(0, a4_max, mesh_size)
    a2_range = np.linspace(0, a2_max, mesh_size)
    a4, a2 = np.meshgrid(a4_range, a2_range)

    # Create the B_max and B_min surfaces
    B_max = calc_B_max(g0, a2, a4)
    B_min = calc_B_min(g0, a2, a4)

    # Apply the triangular mask to the meshgrid
    mesh_mask = (a2 > alpha * a4)
    a4_masked = np.ma.masked_where(mesh_mask, a4)
    a2_masked = np.ma.masked_where(mesh_mask, a2)
    B_max_masked = np.ma.masked_where(mesh_mask, B_max)
    B_min_masked = np.ma.masked_where(mesh_mask, B_min)
    B_max_value = np.max(B_max_masked)

    # Create figure and change properties
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.0, 4.8), constrained_layout=True)
    ax.set_title("Quark EOS parameter space for stable strange stars", y=1.0)
    ax.view_init(elev=15, azim=-155, roll=0)
    ax.set_xlim3d(0, a4_max)
    ax.set_ylim3d(0, a2_max**(1 / 2))
    ax.set_zlim3d(0, B_max_value**(1 / 4))
    ax.set_xlabel("$a_4 ~ [dimensionless]$", fontsize=10, rotation=0)
    ax.set_ylabel("$a_2^{1/2} ~ [MeV]$", fontsize=10, rotation=0)
    ax.set_zlabel("$B^{1/4} ~ [MeV]$", fontsize=10, rotation=90)
    ax.zaxis.set_rotate_label(False)

    # Add each surface plot
    a2_1_2_masked = a2_masked**(1 / 2)
    B_1_4_max_masked = B_max_masked**(1 / 4)
    B_1_4_min_masked = B_min_masked**(1 / 4)
    ax.plot_surface(a4_masked, a2_1_2_masked, B_1_4_max_masked, cmap=cm.Reds, rstride=10, cstride=10, alpha=0.8, label="$B_{max}^{1/4}$")
    ax.plot_surface(a4_masked, a2_1_2_masked, B_1_4_min_masked, cmap=cm.Blues, rstride=10, cstride=10, alpha=0.8, label="$B_{min}^{1/4}$")
    ax.legend(loc=(0.15, 0.3))

    # Add each contour plot (grey projections on each plane)
    ax.contourf(a4_masked, a2_1_2_masked, B_1_4_max_masked, levels=0, zdir='x', offset=a4_max, colors="gray", alpha=0.7, antialiased=True)
    ax.contourf(a4_masked, a2_1_2_masked, B_1_4_max_masked, levels=0, zdir='y', offset=a2_max**(1 / 2), colors="gray", alpha=0.7, antialiased=True)
    ax.contourf(a4_masked, a2_1_2_masked, B_1_4_max_masked, levels=0, zdir='z', offset=0, colors="gray", alpha=0.7, antialiased=True)

    # Create the folder if necessary and save the figure
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    plt.savefig(f"{figure_path}/quark_eos_parameter_space.png")

    # Show graph
    plt.show()

def main():
    """Main logic
    """

    # Constants
    figure_path = "figures/app_quark_eos"
    mesh_size = 1000     # Number of points used in the meshgrid

    # Create the parameter space plot
    plot_parameter_space(figure_path, mesh_size)


# This logic is a simple example, only executed when this file is run directly in the command prompt
if __name__ == "__main__":
    main()
