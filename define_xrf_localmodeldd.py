import numpy as np
import xraylib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from common_modules import readcol
from get_xrf_lines_V1 import get_xrf_lines
from get_constants_xrf_new_V2 import get_constants_xrf
from xrf_comp_new_V2 import xrf_comp

# Read static parameters
static_parameter_file = "static_par_localmodel.txt"
fid = open(static_parameter_file, "r")
finfo_full = fid.read()
finfo_split = finfo_full.split('\n')
solar_file = finfo_split[0]
solar_zenith_angle = float(finfo_split[1])
emiss_angle = float(finfo_split[2])
altitude = float(finfo_split[3])
exposure = float(finfo_split[4])

# Define the XRF model function for fitting
def xrf_localmodel(energy, *parameters):
    # Define proper energy axis
    energy_mid = 0.5 * (energy[:-1] + energy[1:])
    
    # Define input parameters
    at_no = np.array([26, 22, 20, 14, 13, 12, 11, 8])  # Atomic numbers of elements
    weight = np.array(parameters)  # The weights of the elements
    
    # Solar data (assuming the file format is compatible)
    i_angle = 90.0 - solar_zenith_angle
    e_angle = 90.0 - emiss_angle
    energy_solar, tmp1_solar, counts_solar = readcol(solar_file, format='F,F,F')
    
    # Get XRF line data
    k_lines = np.array([xraylib.KL1_LINE, xraylib.KL2_LINE, xraylib.KL3_LINE, xraylib.KM1_LINE])
    l1_lines = np.array([xraylib.L1L2_LINE, xraylib.L1L3_LINE, xraylib.L1M1_LINE, xraylib.L1M2_LINE])
    l2_lines = np.array([xraylib.L2L3_LINE, xraylib.L2M1_LINE, xraylib.L2M2_LINE, xraylib.L2M3_LINE])
    l3_lines = np.array([xraylib.L3M1_LINE, xraylib.L3M2_LINE, xraylib.L3M3_LINE])
    xrf_lines = get_xrf_lines(at_no, xraylib.K_SHELL, k_lines, xraylib.L1_SHELL, l1_lines, xraylib.L2_SHELL, l2_lines, xraylib.L3_SHELL, l3_lines)
    
    # Calculate constants
    const_xrf = get_constants_xrf(energy_solar, at_no, weight, xrf_lines)
    
    # Generate XRF spectrum using the computed structure
    xrf_struc = xrf_comp(energy_solar, counts_solar, i_angle, e_angle, at_no, weight, xrf_lines, const_xrf)
    
    # Compute the XRF spectrum for the given energy bins
    bin_size = energy[1] - energy[0]
    ebin_left = energy_mid - 0.5 * bin_size
    ebin_right = energy_mid + 0.5 * bin_size
    
    no_elements = len(xrf_lines.lineenergy)
    n_lines = len(xrf_lines.lineenergy[0])
    n_ebins = len(energy_mid)
    
    spectrum_xrf = np.zeros(n_ebins)
    for i in range(no_elements):
        for j in range(n_lines):
            line_energy = xrf_lines.lineenergy[i, j]
            bin_index = np.where((ebin_left <= line_energy) & (ebin_right >= line_energy))
            spectrum_xrf[bin_index] += xrf_struc.total_xrf[i, j]
    
    # Scale the spectrum by the exposure and altitude constants
    scaling_factor = (12.5 * 1e4 * 12.5 * (round(exposure / 8.0) + 1) * 1e4) / (exposure * 4 * np.pi * (altitude * 1e4) ** 2)
    spectrum_xrf_scaled = scaling_factor * spectrum_xrf
    
    return spectrum_xrf_scaled

# Example energy data
energy = np.linspace(0.1, 10.0, 100)  # Energy bins from 0.1 to 10 keV

# Define initial guesses for the parameters (weights of the elements)
initial_parameters = [5, 1, 9, 21, 14, 5, 0.5, 45]  # Example values

# Fit the model to the data (using synthetic data as an example)
synthetic_data = xrf_localmodel(energy, *initial_parameters) + np.random.normal(0, 0.1, len(energy))  # Adding noise

# Perform the fitting
popt, pcov = curve_fit(xrf_localmodel, energy, synthetic_data, p0=initial_parameters)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(energy, synthetic_data, label="Synthetic Data", color='blue', linestyle='dashed')
plt.plot(energy, xrf_localmodel(energy, *popt), label="Fitted Model", color='red')
plt.xlabel('Energy (keV)')
plt.ylabel('Flux (counts/s)')
plt.legend()
plt.title('XRF Spectrum Fitting')
plt.grid(True)
plt.show()

# Print fitted parameters
print("Fitted Parameters:", popt)
