'''
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

This file contains the function get_constants_xrf that interpolates the cross-sections from the database to the input energy axis and also takes into account inter-element effects
'''

from common_modules import *
from scipy.interpolate import interp1d

def get_constants_xrf(energy:list,at_no:list,weight:list,xrf_lines:Xrf_Lines) -> Const_Xrf:
    #; Function to compute the different cross sections necessary for computing XRF lines

	#Original: weight = weight/total(weight) ; To confirm that weight fractions are taken
    totalweight = np.sum(weight)
    weight = weight/totalweight
    no_elements = n_elements(at_no)
    n_ebins = n_elements(energy)
    
    #; Identify the number of lines for which xrf computation is done - just by checking array sizes of xrf_lines
    tmp2 = xrf_lines.edgeenergy
    n_lines = np.shape(tmp2)[1]
    
    #; Computing total attenuation of sample at characteristic line energies
    musampletotal_echarline = dblarr(no_elements,n_lines)
    for i in range(0, no_elements):
        for j in range(0, n_lines):
            line_energy = xrf_lines.lineenergy[i,j]
            rad_rate = xrf_lines.radrate[i,j]
            if ((line_energy > 0) and (rad_rate > 0)):
                for k in range(0,no_elements):
                    tmp3 = np.where(xrf_lines.energy_nist[k,:] != 0)
                    x_interp = (xrf_lines.energy_nist[k,tmp3])[0,:]
                    y_interp = (xrf_lines.totalcs_nist[k,tmp3])[0,:]
                    func_interp = interp1d(x_interp, y_interp, fill_value='extrapolate')
                    muelement_echarline = func_interp(line_energy)
                    musampletotal_echarline[i,j] = musampletotal_echarline[i,j] + weight[k]*muelement_echarline
                    
    #; Computing total attenuation of sample for incident energy, but only if incident energy is greater than the edge corresponding for transition
    musampletotal_eincident = dblarr(no_elements,n_lines,n_ebins)
    for i in range(0,no_elements):
        for j in range(0,n_lines):
            line_energy = xrf_lines.lineenergy[i,j]
            rad_rate = xrf_lines.radrate[i,j]
            edge_energy = xrf_lines.edgeenergy[i,j]
            if ((line_energy > 0) and (rad_rate > 0)):#; This is to ensure that only proper XRF lines are taken
                for k in range(0,no_elements):
                    tmp3 = np.where(xrf_lines.energy_nist[k,:] != 0)
                    x_interp = (xrf_lines.energy_nist[k,tmp3])[0,:]
                    y_interp = (xrf_lines.totalcs_nist[k,tmp3])[0,:]
                    func_interp = interp1d(x_interp, y_interp, fill_value='extrapolate')
                    muelement_eincident = func_interp(energy)
                    musampletotal_eincident[i,j,:] = musampletotal_eincident[i,j,:] + weight[k]*muelement_eincident 
                tmp4 = np.where(energy < edge_energy)
                if (n_elements(tmp4) != 0):
                    musampletotal_eincident[i,j,tmp4] = 0.0
    
    #; Computing photoelectric attenuation of element for incident energy, but only if incident energy is greater than the edge corresponding to the transition
    muelementphoto_eincident = dblarr(no_elements,n_lines,n_ebins)
    for i in range(0,no_elements):
        for j in range(0,n_lines):
            line_energy = xrf_lines.lineenergy[i,j]
            rad_rate = xrf_lines.radrate[i,j]
            edge_energy = xrf_lines.edgeenergy[i,j]
            if ((line_energy > 0) and (rad_rate > 0)):#; This is to ensure that only proper XRF lines are taken
                tmp3 = np.where(xrf_lines.energy_nist[i,:] != 0)
                x_interp = (xrf_lines.energy_nist[i,tmp3])[0,:]
                y_interp = (xrf_lines.photoncs_nist[i,tmp3])[0,:]
                func_interp = interp1d(x_interp, y_interp, fill_value='extrapolate')
                muelement_eincident = func_interp(energy)
                muelementphoto_eincident[i,j,:] = muelement_eincident
                tmp4 = np.where(energy < edge_energy)
                if (n_elements(tmp4) != 0):
                    muelementphoto_eincident[i,j,tmp4] = 0.0 #; Setting values less than the edge energy as 0
                    
    #; Creating a structure to return the data
    return Const_Xrf(musampletotal_echarline, musampletotal_eincident,muelementphoto_eincident)

