import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Set the path of the figures
figure_path = "figures/app_quark_eos"

# Constants
n_points = 1000     # Number of points used in the meshgrid
g0 = 930            # Gibbs free energy per baryon of quark matter at null pressure [MeV]
a4_max = 1          # Maximum a4 value [dimensionless]
a2_max = ((1 / 3) - 8 / (3 * (1 + 2**(1 / 3))**3)) * g0**2      # Maximum a2 value [MeV^2]

# Define the (a4, a2) rectangular meshgrid
a4_range = np.linspace(0, a4_max, n_points)
a2_range = np.linspace(0, a2_max, n_points)
a4, a2 = np.meshgrid(a4_range, a2_range)

# Create the B_max and B_min surfaces
B_max = (g0**2 / (108 * np.pi**2)) * (g0**2 * a4 - 9 * a2)
B_min = (g0**2 / (54 * np.pi**2)) * ((4 * g0**2 * a4) / ((1 + 2**(1 / 3))**3) - 3 * a2)

# Apply the triangular mask to the meshgrid
mesh_mask = (a2 > a2_max * a4)
B_max_masked = np.ma.masked_where(mesh_mask, B_max)
B_min_masked = np.ma.masked_where(mesh_mask, B_min)

# Create figure and change properties
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.0, 4.8), constrained_layout=True)
ax.set_title("Quark EOS parameter space for stable strange stars", y=1.0)
ax.view_init(elev=15, azim=-155, roll=0)
ax.set_xlim3d(0, a4_max)
ax.set_ylim3d(0, a2_max**(1 / 2))
ax.set_zlim3d(0, np.max(B_max_masked)**(1 / 4))
ax.set_xlabel("$a_4 ~ [dimensionless]$", fontsize=10, rotation=0)
ax.set_ylabel("$a_2^{1/2} ~ [MeV]$", fontsize=10, rotation=0)
ax.set_zlabel("$B^{1/4} ~ [MeV]$", fontsize=10, rotation=90)
ax.zaxis.set_rotate_label(False)

# Add each surface plot
a2_1_2 = a2**(1 / 2)
B_1_4_max_masked = B_max_masked**(1 / 4)
B_1_4_min_masked = B_min_masked**(1 / 4)
ax.plot_surface(a4, a2_1_2, B_1_4_max_masked, cmap=cm.Reds, alpha=0.8, label="$B_{max}^{1/4}$")
ax.plot_surface(a4, a2_1_2, B_1_4_min_masked, cmap=cm.Blues, alpha=1.0, label="$B_{min}^{1/4}$")
ax.legend(loc=(0.15, 0.3))

# Create the folder if necessary and save the figure
if not os.path.exists(figure_path):
    os.makedirs(figure_path)
plt.savefig(f"{figure_path}/quark_eos_parameter_space.png")

# Show graph
plt.show()
