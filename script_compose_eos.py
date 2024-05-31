import os
import pandas as pd
from constants import UnitConversion as uconv
from data_handling import dataframe_to_csv


# Get the EOS name from the user input
eos_name = input("Type the name of the EOS to be generated from the CompOSE files (default: BSk24): ")
if eos_name == "":
    eos_name = "BSk24"

# Create the path names used. Check CompOSE manual for the description of these files
output_files_path = "data"
eos_files_path = os.path.join(output_files_path, "CompOSE", eos_name)
thermo_file_path = os.path.join(eos_files_path, "eos.thermo")
nb_file_path = os.path.join(eos_files_path, "eos.nb")
mr_file_path = os.path.join(eos_files_path, "eos.mr")

# eos.thermo file parsing

# Read the entire thermo file to get the first line separately
with open(thermo_file_path, 'r') as file:
    lines = file.readlines()

# Extract the first line separately, and get the neutron mass [MeV], needed for the scaling calculations
first_line = lines[0].strip().split()
first_values = [float(value) for value in first_line]
neutron_mass = first_values[0]

# Read the rest of the data into a DataFrame, and assign the column names
thermo_dataframe = pd.read_csv(thermo_file_path, delim_whitespace=True, skiprows=1, header=None)
thermo_dataframe.columns = [
    "col1",
    "index",
    "col3",
    "Q1 = p / n_b [MeV]",
    "Q2 = s / n_b [dimensionless]",
    "Q3 = mu_b / m_n - 1 [dimensionless]",
    "Q4 = mu_q / m_n [dimensionless]",
    "Q5 = mu_l / m_n [dimensionless]",
    "Q6 = f / (n_b m_n) - 1 [dimensionless]",
    "Q7 = e / (n_b m_n) - 1 [dimensionless]",
    "col11"
]
thermo_dataframe.drop(["col1", "index", "col3", "col11"], inplace=True, axis=1)     # Remove not used columns

# eos.nb file parsing

# Read the entire nb file into a DataFrame, and assign the column name
nb_dataframe = pd.read_csv(nb_file_path, delim_whitespace=True, skiprows=2, header=None)
nb_dataframe.columns = ["n_b [fm^-3]"]

# eos.mr file parsing

# Read the entire mr file into a DataFrame, and assign the column names (Lambda and n_c are optional)
mr_dataframe = pd.read_csv(mr_file_path, delim_whitespace=True, skiprows=0, header=None)
column_names = ["R [km]", "M [M_solar]", "Lambda [dimensionless]", "n_c [fm^-3]"]
mr_dataframe.columns = column_names[:mr_dataframe.shape[1]]

# csv files creation

# Create a dictionary with the necessary converted data for the EOS file
MeV_fm3_to_si = uconv.MeV * (10**-15)**(-3)     # [MeV / fm^3] to [J / m^3] conversion constant
density_MeV_fm3_to_cgs = MeV_fm3_to_si * uconv.ENERGY_DENSITY_SI_TO_GU * uconv.MASS_DENSITY_GU_TO_CGS
pressure_MeV_fm3_to_cgs = MeV_fm3_to_si * uconv.ENERGY_DENSITY_SI_TO_GU * uconv.PRESSURE_GU_TO_CGS
eos_data = {
    "rho [g cm^-3]": (thermo_dataframe["Q7 = e / (n_b m_n) - 1 [dimensionless]"] + 1) * nb_dataframe["n_b [fm^-3]"] * neutron_mass * density_MeV_fm3_to_cgs,
    "p [dyn cm^-2]": thermo_dataframe["Q1 = p / n_b [MeV]"] * nb_dataframe["n_b [fm^-3]"] * pressure_MeV_fm3_to_cgs,
    "nb [fm^-3]": nb_dataframe["n_b [fm^-3]"]
}

# Create the EOS dataframe and save as a csv file
eos_dataframe = pd.DataFrame(eos_data)
eos_file_name = eos_name + "_EOS.csv"
dataframe_to_csv(dataframe=eos_dataframe, file_path=output_files_path, file_name=eos_file_name)
print(f"{os.path.join(output_files_path, eos_file_name)} file created!")

# Create the Mass vs Radius dataframe and save as a csv file
m_vs_r_data = {
    "R [km]": mr_dataframe["R [km]"],
    "M [M_solar]": mr_dataframe["M [M_solar]"]
}
m_vs_r_dataframe = pd.DataFrame(m_vs_r_data)
m_vs_r_file_name = eos_name + "_M_vs_R.csv"
dataframe_to_csv(dataframe=m_vs_r_dataframe, file_path=output_files_path, file_name=m_vs_r_file_name)
print(f"{os.path.join(output_files_path, m_vs_r_file_name)} file created!")

# Create the Lambda vs Compactness dataframe and save as a a csv file
lambda_vs_c_data = {
    "C": (mr_dataframe["M [M_solar]"] * uconv.MASS_SOLAR_MASS_TO_GU) / (mr_dataframe["R [km]"] * 10**3),
    "Lambda": mr_dataframe["Lambda [dimensionless]"]
}
lambda_vs_c_dataframe = pd.DataFrame(lambda_vs_c_data)
lambda_vs_c_file_name = eos_name + "_Lambda_vs_C.csv"
dataframe_to_csv(dataframe=lambda_vs_c_dataframe, file_path=output_files_path, file_name=lambda_vs_c_file_name)
print(f"{os.path.join(output_files_path, lambda_vs_c_file_name)} file created!")
