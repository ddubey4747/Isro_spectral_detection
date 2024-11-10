"""
;====================================================================================================
;                              X2ABUNDANCE
;
; Package for determining elemental weight percentages from XRF line fluxes
;
; Algorithm developed by P. S. Athiray (Athiray et al. 2015)
; Codes in IDL written by Netra S Pillai
; Codes for XSPEC localmodel developed by Ashish Jacob Sam and Netra S Pillai
;
; Developed at Space Astronomy Group, U.R.Rao Satellite Centre, Indian Space Research Organisation
;
;====================================================================================================

This is the local model defined for fitting CLASS data using PyXspec

"""
# Importing necessary modules
import numpy as np
from xspec import *
import xraylib
from common_modules import *
from get_xrf_lines_V1 import get_xrf_lines
from get_constants_xrf_new_V2 import get_constants_xrf
from xrf_comp_new_V2 import xrf_comp

# Getting the static parameters for the local model
static_parameter_file = "static_par_localmodel.txt"
fid = open(static_parameter_file,"r")
finfo_full = fid.read()
finfo_split = finfo_full.split('\n')
solar_file = finfo_split[0]
solar_zenith_angle = float(finfo_split[1])
emiss_angle = float(finfo_split[2])
altitude = float(finfo_split[3])
exposure = float(finfo_split[4])

# Defining the model function
def xrf_localmodel(energy, parameters, flux):
    
    # Defining proper energy axis
    energy_mid = np.zeros(np.size(energy)-1)
    for i in np.arange(np.size(energy)-1):
        energy_mid[i] = 0.5*(energy[i+1] + energy[i])
        
    # Defining some input parameters required for x2abund xrf computation modules
    at_no = np.array([26,22,20,14,13,12,11,8])
    
    weight = list(parameters)
    
    i_angle = 90.0 - solar_zenith_angle
    e_angle = 90.0 - emiss_angle
    (energy_solar,tmp1_solar,counts_solar) = readcol(solar_file,format='F,F,F')
    
    # Computing the XRF line intensities
    k_lines = np.array([xraylib.KL1_LINE, xraylib.KL2_LINE, xraylib.KL3_LINE, xraylib.KM1_LINE, xraylib.KM2_LINE, xraylib.KM3_LINE, xraylib.KM4_LINE, xraylib.KM5_LINE])
    l1_lines = np.array([xraylib.L1L2_LINE, xraylib.L1L3_LINE, xraylib.L1M1_LINE, xraylib.L1M2_LINE, xraylib.L1M3_LINE, xraylib.L1M4_LINE, xraylib.L1M5_LINE, xraylib.L1N1_LINE, xraylib.L1N2_LINE, xraylib.L1N3_LINE, xraylib.L1N4_LINE, xraylib.L1N5_LINE, xraylib.L1N6_LINE, xraylib.L1N7_LINE])
    l2_lines = np.array([xraylib.L2L3_LINE, xraylib.L2M1_LINE, xraylib.L2M2_LINE, xraylib.L2M3_LINE, xraylib.L2M4_LINE, xraylib.L2M5_LINE, xraylib.L2N1_LINE, xraylib.L2N2_LINE, xraylib.L2N3_LINE, xraylib.L2N4_LINE, xraylib.L2N5_LINE, xraylib.L2N6_LINE, xraylib.L2N7_LINE])
    l3_lines = [xraylib.L3M1_LINE, xraylib.L3M2_LINE, xraylib.L3M3_LINE, xraylib.L3M4_LINE, xraylib.L3M5_LINE, xraylib.L3N1_LINE,xraylib.L3N2_LINE, xraylib.L3N3_LINE, xraylib.L3N4_LINE, xraylib.L3N5_LINE, xraylib.L3N6_LINE, xraylib.L3N7_LINE]
    xrf_lines = get_xrf_lines(at_no, xraylib.K_SHELL, k_lines, xraylib.L1_SHELL, l1_lines, xraylib.L2_SHELL, l2_lines, xraylib.L3_SHELL, l3_lines)
    const_xrf = get_constants_xrf(energy_solar, at_no, weight, xrf_lines)
    xrf_struc = xrf_comp(energy_solar,counts_solar,i_angle,e_angle,at_no,weight,xrf_lines,const_xrf)
    
    # Generating XRF spectrum
    bin_size = energy[1] - energy[0]
    ebin_left = energy_mid - 0.5*bin_size
    ebin_right = energy_mid + 0.5*bin_size
    
    no_elements = (np.shape(xrf_lines.lineenergy))[0]
    n_lines = (np.shape(xrf_lines.lineenergy))[1]
    n_ebins = np.size(energy_mid)
    
    spectrum_xrf = dblarr(n_ebins)
    for i in range(0, no_elements):
        for j in range(0, n_lines):
            line_energy = xrf_lines.lineenergy[i,j]
            bin_index = np.where((ebin_left <= line_energy) & (ebin_right >= line_energy))
            spectrum_xrf[bin_index] = spectrum_xrf[bin_index] + xrf_struc.total_xrf[i,j]
            
    # Defining the flux array required for XSPEC
    scaling_factor = (12.5*1e4*12.5*(round(exposure/8.0)+1)*1e4)/(exposure*4*np.pi*(altitude*1e4)**2)
    spectrum_xrf_scaled = scaling_factor*spectrum_xrf
    for i in range(0, n_ebins):
        flux[i] = spectrum_xrf_scaled[i]
        
# Specifying parameter information
xrf_localmodel_ParInfo = ("Wt_Fe \"\" 5 1 1 20 20 1e-2","Wt_Ti \"\" 1 1e-6 1e-6 20 20 1e-2","Wt_Ca \"\" 9 5 5 20 20 1e-2","Wt_Si \"\" 21 15 15 35 35 1e-2","Wt_Al \"\" 14 5 5 20 20 1e-2","Wt_Mg \"\" 5 1e-6 1e-6 20 20 1e-2","Wt_Na \"\" 0.5 1e-6 1e-6 5 5 1e-2","Wt_O \"\" 45 30 30 60 60 1e-2")

# Creating the local model in PyXspec
AllModels.addPyMod(xrf_localmodel, xrf_localmodel_ParInfo, 'add')
    

        
    