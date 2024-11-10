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

This file contains the function get_xrf_lines which derives various constants (cross sections, fluorescent yields, jump factors etc) for the elements of interest

'''
from common_modules import *
import xraylib
import os

def get_xrf_lines(at_no, k_shell, k_lines, l1_shell, l1_lines, l2_shell, l2_lines, l3_shell, l3_lines) -> Xrf_Lines:
    no_elements = n_elements(at_no)
    edgeenergy = dblarr(no_elements, 5)
    fluoryield = dblarr(no_elements, 5)
    jumpfactor = dblarr(no_elements, 5)
    radrate = dblarr(no_elements, 5)
    lineenergy = dblarr(no_elements, 5)
    
    energy_nist = dblarr(no_elements, 100)
    photoncs_nist = dblarr(no_elements, 100)
    totalcs_nist = dblarr(no_elements, 100)
    elename_string = strarr(no_elements)
    (atomic_number_list, kalpha_list, ele_list, be_list, density_list, kbeta_list) = readcol('/home/netra/CLASS/X2ABUNDANCE/x2abundance_python/data_constants/kalpha_be_density_kbeta.txt', format='I,F,A,F,F,F')

    fullpath = os.path.abspath(__file__)
    script_path, filename = os.path.split(fullpath)
    
    for i in range(0, no_elements):
        #; Getting the NIST cross-sections for all the elements
        tmp1 = np.where(atomic_number_list == at_no[i])
        elename_string[i] = ele_list[tmp1]
        
        filename = script_path + '/data_constants/ffast/ffast_'+str(int(at_no[i])).strip()+'_'+(ele_list[tmp1])[0]+'.txt'#; Getting the attenuation coefficients from FFAST database
        (column1, column2, column3, column4, column5, column6, column7, column8) = readcol(filename, format = 'D,F,F,F,F,F,F,F')
        
        n = n_elements(column1)
        energy_nist[i, 0:n] = column1
        photoncs_nist[i, 0:n] = column4
        totalcs_nist[i, 0:n] = column6
        
        #; Getting the edge energy for each shell
        edgeenergy[i, 0:2] = xraylib.EdgeEnergy(at_no[i], k_shell)#; edge energy is defined for shells so it will be same for kalpha and kbeta
        edgeenergy[i, 2] = xraylib.EdgeEnergy(at_no[i], l1_shell)
        edgeenergy[i, 3] = xraylib.EdgeEnergy(at_no[i], l2_shell)
        edgeenergy[i, 4] = xraylib.EdgeEnergy(at_no[i], l3_shell)
        
        #; Getting the fluorescent yields for each shell
        fluoryield[i, 0:2] = xraylib.FluorYield(at_no[i], k_shell)#; fluorescent yield is defined for shells so it will be same for kalpha and kbeta
        try:
            fluoryield[i, 2] = xraylib.FluorYield(at_no[i], l1_shell)
        except:
            fluoryield[i, 2] = 0.0
        try:
            fluoryield[i, 3] = xraylib.FluorYield(at_no[i], l2_shell)
        except:
            fluoryield[i, 3] = 0.0
        try:
            fluoryield[i, 4] = xraylib.FluorYield(at_no[i], l3_shell)
        except:
            fluoryield[i, 4] = 0.0
       
        #; Getting the jump factors for each shell
        jumpfactor[i, 0:2] = xraylib.JumpFactor(at_no[i], k_shell)#; jump factor is defined for shells so it will be same for kalpha and kbeta
        try:
            jumpfactor[i, 2] = xraylib.JumpFactor(at_no[i], l1_shell)
        except:
            jumpfactor[i, 2] = 0.0
        try:
            jumpfactor[i, 3] = xraylib.JumpFactor(at_no[i], l2_shell)
        except:
            jumpfactor[i, 3] = 0.0
        try:
            jumpfactor[i, 4] = xraylib.JumpFactor(at_no[i], l3_shell)
        except:
            jumpfactor[i, 4] = 0.0
        
        #; Getting the radiative rates and energy for kbeta
        kbeta_lines = k_lines[3:8]
        kbeta_lines_length = n_elements(kbeta_lines)
        radrate_kbeta = dblarr(kbeta_lines_length)
        lineenergy_kbeta = dblarr(kbeta_lines_length)
        for j in range(0, kbeta_lines_length):
            try:
                radrate_kbeta[j] = xraylib.RadRate(at_no[i], kbeta_lines[j])
                lineenergy_kbeta[j] = xraylib.LineEnergy(at_no[i], kbeta_lines[j])
            except:
                radrate_kbeta[j] = 0.0
                lineenergy_kbeta[j] = 0.0

        allowed_lines_index_kbeta = np.where(radrate_kbeta > 0)
        if (n_elements(allowed_lines_index_kbeta) != 0):# then begin
            lineenergy[i, 0] = total(radrate_kbeta[allowed_lines_index_kbeta]*lineenergy_kbeta[allowed_lines_index_kbeta])/total(radrate_kbeta[allowed_lines_index_kbeta])#; weighted average of kbeta energies
            radrate[i, 0] = total(radrate_kbeta[allowed_lines_index_kbeta])
		# If no kbeta line is possible then the energy and the radrate will be set as 0.

        #; Getting the radiative rates and energy for kalpha
        kalpha_lines = k_lines[0:3]
        radrate_kalpha = dblarr(n_elements(kalpha_lines))
        lineenergy_kalpha = dblarr(n_elements(kalpha_lines))
        for j in range(0, n_elements(kalpha_lines)):#-1 do begin
            try:
                radrate_kalpha[j] = xraylib.RadRate(at_no[i], kalpha_lines[j])
                lineenergy_kalpha[j] = xraylib.LineEnergy(at_no[i], kalpha_lines[j])
            except:
                radrate_kalpha[j] = 0.0
                lineenergy_kalpha[j] = 0.0

        allowed_lines_index_kalpha = np.where(radrate_kalpha > 0)
        if (n_elements(allowed_lines_index_kalpha) != 0):# then begin
            lineenergy[i, 1] = total(radrate_kalpha[allowed_lines_index_kalpha]*lineenergy_kalpha[allowed_lines_index_kalpha])/total(radrate_kalpha[allowed_lines_index_kalpha])#; weighted average of kalpha energies
            radrate[i, 1] = total(radrate_kalpha[allowed_lines_index_kalpha])

        #; Getting the radiative rates and energy for l1lines
        radrate_l1 = dblarr(n_elements(l1_lines))
        lineenergy_l1 = dblarr(n_elements(l1_lines))
        for j in range(0, n_elements(l1_lines)):#-1 do begin
            try:
                radrate_l1[j] = xraylib.RadRate(at_no[i], l1_lines[j])
                lineenergy_l1[j] = xraylib.LineEnergy(at_no[i], l1_lines[j])
            except:
                radrate_l1[j] = 0.0
                lineenergy_l1[j] = 0.0

        allowed_lines_index_l1 = np.where(radrate_l1 > 0)
        if (n_elements(allowed_lines_index_l1) != 0):# then begin
            lineenergy[i, 2] = total(radrate_l1[allowed_lines_index_l1]*lineenergy_l1[allowed_lines_index_l1])/total(radrate_l1[allowed_lines_index_l1])#; weighted average of l1 energies
            radrate[i, 2] = total(radrate_l1[allowed_lines_index_l1])
		# If no l1 line is possible then the energy and the radrate will be set as 0.

        #; Getting the radiative rates and energy for l2lines
        radrate_l2 = dblarr(n_elements(l2_lines))
        lineenergy_l2 = dblarr(n_elements(l2_lines))
        for j in range(0, n_elements(l2_lines)):#-1 do begin
            try:
                radrate_l2[j] = xraylib.RadRate(at_no[i], l2_lines[j])
                lineenergy_l2[j] = xraylib.LineEnergy(at_no[i], l2_lines[j])
            except:
                radrate_l2[j] = 0.0
                lineenergy_l2[j] = 0.0

        allowed_lines_index_l2 = np.where(radrate_l2 > 0)
        if (n_elements(allowed_lines_index_l2) != 0):# then begin
            lineenergy[i, 3] = total(radrate_l2[allowed_lines_index_l2]*lineenergy_l2[allowed_lines_index_l2])/total(radrate_l2[allowed_lines_index_l2])#; weighted average of l2 energies
            radrate[i, 3] = total(radrate_l2[allowed_lines_index_l2])
		# If no l2 line is possible then the energy and the radrate will be set as 0.

        #; Getting the radiative rates and energy for l3lines
        radrate_l3 = dblarr(n_elements(l3_lines))
        lineenergy_l3 = dblarr(n_elements(l3_lines))
        for j in range(0, n_elements(l3_lines)):#-1 do begin
            try:
                radrate_l3[j] = xraylib.RadRate(at_no[i], l3_lines[j])
                lineenergy_l3[j] = xraylib.LineEnergy(at_no[i], l3_lines[j])
            except:
                radrate_l3[j] = 0.0
                lineenergy_l3[j] = 0.0

        allowed_lines_index_l3 = np.where(radrate_l3 > 0)
        if (n_elements(allowed_lines_index_l3[0]) != 0):# then begin
            lineenergy[i, 4] = total(radrate_l3[allowed_lines_index_l3]*lineenergy_l3[allowed_lines_index_l3])/total(radrate_l3[allowed_lines_index_l3])#; weighted average of l3 energies
            radrate[i, 4] = total(radrate_l3[allowed_lines_index_l3])
		# If no l3 line is possible then the energy and the radrate will be set as 0.

    #; Creating a structure to return the data
    return Xrf_Lines(edgeenergy,fluoryield,jumpfactor,radrate,lineenergy,energy_nist,photoncs_nist,totalcs_nist,elename_string)

    
