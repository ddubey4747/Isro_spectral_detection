#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to check the model fitting and data processing without using XSPEC.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from scipy.optimize import curve_fit

# Specifying input files
class_l1_data = 'ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits'
bkg_file = 'ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits'
scatter_atable = 'tbmodel_20210827T210316000_20210827T210332000.fits'
solar_model_file = 'modelop_20210827T210316000_20210827T210332000.txt'
response_path = 'test/'

# Load FITS files and extract data
def load_fits_data(fits_file):
    with fits.open(fits_file) as hdu:
        data = hdu[1].data
        header = hdu[1].header
    return data, header
# data = fits.open(class_l1_data)
# header = data[1].header
# data.close()
# Read the data from the FITS files
data, header = load_fits_data(class_l1_data)
bkg_data, _ = load_fits_data(bkg_file)
scatter_data, _ = load_fits_data(scatter_atable)

# Solar zenith angle, emission angle, satellite altitude, and exposure time from header
solar_zenith_angle = header['SOLARANG']
emiss_angle = header['EMISNANG']
sat_alt = header['SAT_ALT']
tint = header['EXPOSURE']

# Define the xrf_localmodel (Example)
def xrf_localmodel(energy, param1, param2):
    # Implement your model function (this is just an example)
    return param1 * np.exp(-param2 * energy)

# Simulating the response from the scatter table (simplified)
def scatter_response(energy):
    # Here we assume the scatter table provides some form of energy response
    return np.interp(energy, scatter_data['ENERGY'], scatter_data['FLUX'])

# Prepare data for fitting
energy = data['ENERGY']
observed_flux = data['FLUX']
background_flux = bkg_data['FLUX']
corrected_flux = observed_flux - background_flux  # Subtract background

# Fit model to data
def model_to_fit(energy, param1, param2):
    scatter_flux = scatter_response(energy)
    xrf_flux = xrf_localmodel(energy, param1, param2)
    return scatter_flux + xrf_flux

# Initial guesses for parameters
initial_guess = [1.0, 0.1]

# Perform the fitting
params, params_covariance = curve_fit(model_to_fit, energy, corrected_flux, p0=initial_guess)

# Plotting the fit results
pdf_plot = PdfPages('test/plots_x2abund_test.pdf')

# Calculate model flux for the fitted parameters
fitted_flux = model_to_fit(energy, *params)
delchi = (corrected_flux - fitted_flux) / np.sqrt(fitted_flux)

# Create the plot
fig, (axis1, axis2) = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1], 'height_ratios': [3, 1]})
fig.suptitle('Data Model Comparison')

axis1.plot(energy, corrected_flux, label='Data')
axis1.plot(energy, fitted_flux, label='Model', linestyle='--')
axis1.set_yscale("log")
axis1.set_xlabel('Energy (keV)')
axis1.set_ylabel('Counts/s')
axis1.legend()

axis2.plot(energy, delchi)
axis2.set_xlabel('Energy (keV)')
axis2.set_ylabel('Delchi')

# Save the plot to a PDF
pdf_plot.savefig(fig, bbox_inches='tight', dpi=300)
plt.close(fig)

pdf_plot.close()

# Output fitted parameters
print("Fitted parameters:", params)

# Save the results
with open('static_par_localmodel.txt', 'w') as f:
    f.write(f"Solar zenith angle: {solar_zenith_angle}\n")
    f.write(f"Emission angle: {emiss_angle}\n")
    f.write(f"Satellite altitude: {sat_alt}\n")
    f.write(f"Exposure time: {tint}\n")
    f.write(f"Fitted parameters: {params}\n")
