"""
Ari Silburt, March 2018 Changes: 
- Made ACI-b (PSU cluster) compatible. 
- Translated French->English
- Growth/decay of spots (see spot_functions.py)
- Allow for multiple spots (see setup_soap_play.py)
- Improved naming convention
- Cleaned up code
Must load a python 2.7 environment for this to run! E.g. source activate soapenv

Play with Xavier D's example SOAP script to understand SOAP's workings better.

Created May 4, 2017 by Tom Loredo, mostly from XD's example driver script, 
soap_with_real_solar_spectra_for_Eric.py
"""

import matplotlib
matplotlib.use('Agg')

import os, time, sys
import h5py
import ConfigParser
import resource

from soap2 import _starspot, ooapi
from soap2 import pyrdb, calculate_spectrum, read_spectrum_star, functions

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import *

import spot_functions as sf
import spot_phase_config as spc

# def main(spectral_regions=[3923.87, 4010.]):
def main(spectral_regions):
    # start_time = time.time()
    ######################################################################
    ###          Constants             ###################################
    ######################################################################

    c = 299792458. # speed of light in m/s

    vrad_ccf2 = arange(-20000.,20001.,250.) #vrad for the CCF

    # Reading HARPS G2 mask to calculate CCFs
    # X_G2 = loadtxt('data/G2.mas')  # lambda_1  lambda_2  val
    mask_store = h5py.File('data/G2Mask.h5', 'r')
    X_G2 = mask_store['mask_raw'].value
    mask_store.close()
    contrast = X_G2[:,2]

    # Get center wavelengths and widths for mask bins
    lambda0_line_G2, contrast_G2 = [], []
    dlambda0 = empty_like(contrast)
    for i in arange(len(contrast)):
        lambda0_start_G2 = X_G2[i,0] # Angstrom
        lambda0_end_G2 = X_G2[i,1] # Angstrom
        lambda0_line_G2.append(1/2.*(lambda0_start_G2+lambda0_end_G2)) # Angstrom
        dlambda0[i] = lambda0_end_G2 - lambda0_start_G2
        contrast_G2.append(contrast[i])
    freq_line_G2 = array(lambda0_line_G2) + functions.Delta_wavelength(-4138.8915572235164,array(lambda0_line_G2))    
    contrast_line_G2 = array(contrast_G2)

    ######################################################################
    ###           Read Solar FTS spectra    ##############################
    ######################################################################

    print '#############################'
    print 'Reading Sun data'
    print '#############################'

    # Read the FTS spectra of the quiet photosphere and of a spot
    if True:  # read HDF5 data
        sun_store = h5py.File('data/SunLowHigh.h5', 'r')
        # Note that HDF5 node order may not be preserved; access by name, not order.
        names = ['freq_Sun_high', 'flux_Sun_high', 'freq_Sun_low', 'flux_Sun_low', 'freq_Sun', \
            'high_activity_Sun', 'low_activity_Sun']
        values = [sun_store[name].value for name in names]
        freq_Sun_high, flux_Sun_high, freq_Sun_low, flux_Sun_low, freq_Sun, \
            high_activity_Sun, low_activity_Sun = values
        sun_store.close()
    else:  # read ASCII data
        values0 = read_spectrum_star.read_sun_spectrum('data/spot_spectrum.rdb','data/solar_spectrum.rdb')
        freq_Sun_high, flux_Sun_high, freq_Sun_low, flux_Sun_low, freq_Sun, \
        high_activity_Sun, low_activity_Sun = values0

    # Remove negative values in the FTS spectra
    for i in arange(len(low_activity_Sun)):
        if low_activity_Sun[i] <= 0:
            low_activity_Sun[i] = low_activity_Sun[i-1]
        if high_activity_Sun[i] <= 0:
            high_activity_Sun[i] = high_activity_Sun[i-1]

    #Select the spectra to have some continuum on the sides and not spectral line, for extrapolation of the edge (not necessary, but better)
    # select_sun = (freq_Sun > 3923.87) & (freq_Sun < 6663.56)  #  total freq range, takes ~77 sec
    # select_sun = (freq_Sun > 3923.87) & (freq_Sun < 4500.)  #  Ca II K, Balmer; itot takes ~22 sec
    # select_sun = (freq_Sun > 3923.87) & (freq_Sun < 4010.)  #  itot takes ~3 sec; edge effects?
    select_sun = (freq_Sun > spectral_regions[0]) & (freq_Sun < spectral_regions[1])
    wavelength = freq_Sun[select_sun]
    spectrum = low_activity_Sun[select_sun]
    spectrum_magn_region = high_activity_Sun[select_sun]

    # Cut 2 angstrom on both sides of the spectrum
    max_shift           = 2. #in Angstrom
    select = (wavelength >= wavelength[0]+max_shift) & (wavelength < wavelength[-1]-max_shift)
    wavelength_select = wavelength[select]

    # Select the lines in the HARPS G2 mask that are in the range of "wavelength_select"
    select_line_mask = (freq_line_G2 > min(wavelength_select)) & (freq_line_G2 < max(wavelength_select))
    freq_line_G2 = freq_line_G2[select_line_mask]
    contrast_line_G2 = contrast_line_G2[select_line_mask]

    # Shifting the wavelength of the solar FTS spectra so that the spectra and the G2 mask have a difference in velocity of 0.
    wavelength_shift = wavelength + functions.Delta_wavelength(+2100.,wavelength) # calculate the position of the spectral lines as a function of the template speed

    # Interpolation to get the shifted wavelength on the same scale as the original one
    # C optimization, ATTENTION wavelength_extrapol_tmp and wavelength_shift must be in type "float64"
    # the 2nd vector must have more points than the first, and the 3rd vector must match the flow of the 2nd vector (which is the wavelength solution)
    similar_sampling = calculate_spectrum.get_same_sampling(wavelength_select,wavelength_shift,spectrum)
    # if the last value is 2, the value "depth_from_continuum" is the difference between the actual sampling and the one we want
    master_value_at_spectrum_position = calculate_spectrum.get_spectrum_blueshifted(similar_sampling[0], similar_sampling[1], similar_sampling[3],array([0.,0.,0.]),2)

    ####################################################################################
    #                    INITITIALIZATION FOR THE STAR IN SOAP           ###############
    ####################################################################################

    try:
        seed_id = int(sys.argv[1])      # unique id for the simulation
    except:
        print("*** You must pass a unique ID as the first argument ***")
        sys.exit()

    config = ConfigParser.ConfigParser()
    config.read("config.cfg")

    GRID        = int(config.get('main','grid' ))
    NRHO        = int(config.get('main','nrho' ))
    INST_RESO   = int(config.get('main','instrument_reso' ))
    if INST_RESO > 0:
        inst_profile_FWHM = c/INST_RESO # Resolution = c/Delta_v -> Delta_v = c/R
        inst_profile_sigma = inst_profile_FWHM/(2*sqrt(2*log(2)))
    RAD_Sun     = int(config.get('star','radius_sun' ))
    RAD         = float(config.get('star','radius' )) * RAD_Sun
    PROT        = float(config.get('star','prot' ))          # rotation period of sun (at equator)
    PROT_POLE   = float(config.get('star','prot_pole' ))     # rotation period of sun at pole
    STAR        = ooapi.Star(vrot   = (2.*pi*RAD)/(PROT*86400.),      \
                            vrot_pole = (2.*pi*RAD)/(PROT_POLE*86400.), \
                            incl   = float(config.get('star','I')),\
                            limba1 = float(config.get('star','limb1'   )),  \
                            limba2 = float(config.get('star','limb2'   )),  \
                            modif_bis_quad = eval(config.get('star','modif_bis_quad'   )),  \
                            modif_bis_lin  = eval(config.get('star','modif_bis_lin'   )),  \
                            modif_bis_cte  = 0.0,  \
                            psi   = float(config.get('star','psi'    )),  \
                            rad   = RAD,\
                            gamma = float(config.get('star','gamma'  )),  \
                            Temp  = int(config.get('star','Tstar')),\
                            Temp_diff_spot = int(config.get('star','Tdiff_spot')))
    CCF         = ooapi.Ccf(vrad  = wavelength,
                           intensity = spectrum,
                           width = max(wavelength),
                           step  = median(wavelength[1:]-wavelength[:-1]),
                           star  = STAR)

    STAR.calc_maps(GRID)

    print '#############################'
    print 'Calculating integrated spectrum for the quiet Sun'
    print '#############################'
    t0 = time.time()
    fstar_quiet_sun, sstar = _starspot.itot(STAR.vrot, STAR.incl, STAR.limba1, STAR.limba2, STAR.modif_bis_quad, STAR.modif_bis_lin, STAR.modif_bis_cte, GRID, CCF.vrad, CCF.intensity, CCF.v_interval, CCF.n_v, CCF.n)
    fstar_quiet_sun_raw = fstar_quiet_sun
    t1 = time.time()
    print '#############################'
    print 'ENDED in %.1f seconds' % (t1-t0)
    print '#############################'
    print

    calculate_CCFs = int(config.get('extra','calculate_CCFs'))
    calculate_flux_and_bconv = int(config.get('extra','calculate_flux_and_bconv'))

    ####################################################################################
    #            INITITIALIZATION FOR THE ACTIVE REGION IN SOAP          ###############
    ####################################################################################

    # long = longitue
    # lat = latitude
    # size = active region size, in stellar radius. If you want a size S in area of the visible hemisphere, then put sqrt(2*s) for the size poarameter
    # I = intensity of the active region, this is arbitrary as it is recalculated after in the code
    # magn_feature_type = 0 for spot, 1 for plage

    CCF_spot     = ooapi.Ccf(vrad  = wavelength,
                           intensity = spectrum_magn_region,
                           width = max(wavelength),
                           step  = median(wavelength[1:]-wavelength[:-1]),
                           star  = STAR)

    # A.S. Custom functions that get phases and create star spots
    PSI = spc.obs_get_phases_nightly(PROT, seed_id)
    SPOTS, msh_ph, n_spots_ph = spc.get_spots(PSI, STAR, GRID, NRHO, PROT, seed_id)

    ################################################################################

    print '#############################'
    print 'Calculating active region spectrum'
    print '#############################'
    # print 'start SPOTS'
    # print SPOTS
    # print 'end SPOTS'
    t2 = time.time()
    # get initial arrays from first spot
    SPOT = SPOTS[0]
    if SPOT.visible <4:
        xyz = _starspot.spot_init(np.max(SPOT.s), SPOT.long, SPOT.lat, STAR.incl, NRHO)
        fspot_flux,fspot_bconv,fspot_tot,sspot = _starspot.spot_scan_npsi(xyz, NRHO, PSI, len(PSI), STAR.vrot, STAR.vrot_pole, STAR.incl,\
                                                                          STAR.limba1, STAR.limba2, STAR.modif_bis_quad, STAR.modif_bis_lin, STAR.modif_bis_cte, GRID,\
                                                                          CCF.vrad, CCF.intensity, CCF_spot.intensity, CCF.v_interval, CCF.n_v, CCF.n,\
                                                                          SPOT.s, SPOT.long, SPOT.lat, SPOT.magn_feature_type, STAR.Temp, STAR.Temp_diff_spot)
    # iterate over rest of spots, update arrays
    for i in range(1, len(SPOTS)):
        SPOT = SPOTS[i]
        if SPOT.visible <4:
            xyz = _starspot.spot_init(np.max(SPOT.s), SPOT.long, SPOT.lat, STAR.incl, NRHO)
            fspot_flux_,fspot_bconv_,fspot_tot_,sspot_ = _starspot.spot_scan_npsi(xyz, NRHO, PSI, len(PSI), STAR.vrot, STAR.vrot_pole, STAR.incl,\
                                                                STAR.limba1, STAR.limba2, STAR.modif_bis_quad, STAR.modif_bis_lin, STAR.modif_bis_cte, GRID,\
                                                                CCF.vrad, CCF.intensity, CCF_spot.intensity, CCF.v_interval, CCF.n_v, CCF.n,\
                                                                SPOT.s, SPOT.long, SPOT.lat, SPOT.magn_feature_type, STAR.Temp, STAR.Temp_diff_spot)
            fspot_flux += fspot_flux_
            fspot_bconv += fspot_bconv_
            fspot_tot += fspot_tot_
            sspot += sspot_

    t3 = time.time()
    print '#############################'
    print 'ENDED in %.1f seconds' % (t3-t2)
    print '#############################'
    print

    # Convolving with the resolution instrumental profile
    if INST_RESO > 0:
        print '#############################'
        print 'Lowering resoluion of integrated quiet Sun'
        print '#############################'
        t4 = time.time()
        #We do not need to put some zeros on the sides because they already are there (we cutted at min(wavelength)+2 and max(wavelength)-2)
        fstar_quiet_sun_tmp = calculate_spectrum.func_lower_resolution(wavelength,-fstar_quiet_sun+1,inst_profile_sigma)
        fstar_quiet_sun = 1-fstar_quiet_sun_tmp*(1-min(fstar_quiet_sun))/max(fstar_quiet_sun_tmp) # normalization
        t5 = time.time()
        
        plt.figure('quiet')
        plt.plot(wavelength,fstar_quiet_sun,label='Integrated spectrum no activity at selected resolution')
        print '#############################'
        print 'ENDED in %.1f seconds' % (t5-t4)
        print '#############################'

    # Calculates the CCF for the quiet Sun intgrated spectrum and plot it
    if calculate_CCFs == 1:
        CCF_quiet_Sun = functions.calculate_CCF(vrad_ccf2,wavelength,fstar_quiet_sun,freq_line_G2,contrast_line_G2)
        CCF_quiet_Sun /= max(CCF_quiet_Sun)

        #appodisation parameters, same as for the CCF of the quiet photosphere and the spot (see "create_Sun_CCF.py")
        nb_zeros_on_sides = 5
        period_appodisation = ((len(vrad_ccf2)-1)/2.)
        len_appodisation = period_appodisation/2.
        #appodisation, same as for the CCF of the quiet photosphere and spot (see "create_Sun_CCF.py")
        a = arange(len(vrad_ccf2))
        b = 0.5*cos(2*pi/period_appodisation*a-pi)+0.5
        appod = concatenate([zeros(nb_zeros_on_sides),b[:len_appodisation],ones(len(vrad_ccf2)-period_appodisation-2*nb_zeros_on_sides),b[:len_appodisation][::-1],zeros(nb_zeros_on_sides)])
        CCF_quiet_Sun  = 1-((-CCF_quiet_Sun+1)*appod)
        #######################

        fit_CCF_quiet_Sun = functions.compute_bis(vrad_ccf2,CCF_quiet_Sun)

    print '###############################################################'
    print 'Calculating integrated spectrum including active region'
    print '###############################################################'

    # Removing from the integrated quiet solar spectrum the contribution of the active region
    sstar_active,fstar_active_sun,fstar_active_sun_flux,fstar_active_sun_bconv,CCF_active_sun,CCF_active_sun_flux,CCF_active_sun_bconv,fit_CCF_active_sun,fit_CCF_active_sun_flux,fit_CCF_active_sun_bconv = [],[],[],[],[],[],[],[],[],[]
    for i in arange(len(fspot_tot)):
        print i,' over ',len(fspot_tot)
        if len(nonzero(fspot_tot[i])[0]) > 0:
            sstar_active = append(sstar_active,sstar - sspot[i])
            fstar_active_sun.append(fstar_quiet_sun_raw - fspot_tot[i])
            if calculate_flux_and_bconv == 1:
                fstar_active_sun_flux.append(fstar_quiet_sun_raw - fspot_flux[i])
                fstar_active_sun_bconv.append(fstar_quiet_sun_raw - fspot_bconv[i])
            
            # Convolving with the resolution instrumental profile
            if INST_RESO > 0:
                t6 = time.time()
                print '#############################'
                print 'Lowering resoluion of integrated Sun'
                print '#############################'
                ##Convolving with the HARPS instrumental profile
                fstar_active_sun_tmp = calculate_spectrum.func_lower_resolution(wavelength,-fstar_active_sun[i]+1,inst_profile_sigma)
                fstar_active_sun[i] = 1-fstar_active_sun_tmp*(1-min(fstar_active_sun[i]))/max(fstar_active_sun_tmp) # normalization
                if calculate_flux_and_bconv == 1:
                    fstar_active_sun_flux_tmp = calculate_spectrum.func_lower_resolution(wavelength,-fstar_active_sun_flux[i]+1,inst_profile_sigma)
                    fstar_active_sun_flux[i] = 1-fstar_active_sun_flux_tmp*(1-min(fstar_active_sun_flux[i]))/max(fstar_active_sun_flux_tmp) # normalization
                    fstar_active_sun_bconv_tmp = calculate_spectrum.func_lower_resolution(wavelength,-fstar_active_sun_bconv[i]+1,inst_profile_sigma)
                    fstar_active_sun_bconv[i] = 1-fstar_active_sun_bconv_tmp*(1-min(fstar_active_sun_bconv[i]))/max(fstar_active_sun_bconv_tmp) # normalization
                t7 = time.time()
                print '#############################'
                print 'ENDED in %.1f seconds' % (t7-t6)
                print '#############################'
        
            # Calculates the CCF for the intgrated spectrum including the active region
            if calculate_CCFs == 1:
                print 'calculating CCF'
                CCF_active_sun.append(functions.calculate_CCF(vrad_ccf2,wavelength,fstar_active_sun[i],freq_line_G2,contrast_line_G2))
                CCF_active_sun[i] /= max(CCF_active_sun[i])
                if calculate_flux_and_bconv == 1:
                    CCF_active_sun_flux.append(functions.calculate_CCF(vrad_ccf2,wavelength,fstar_active_sun_flux[i],freq_line_G2,contrast_line_G2))
                    CCF_active_sun_flux[i] /= max(CCF_active_sun_flux[i])
                    CCF_active_sun_bconv.append(functions.calculate_CCF(vrad_ccf2,wavelength,fstar_active_sun_bconv[i],freq_line_G2,contrast_line_G2))
                    CCF_active_sun_bconv[i] /= max(CCF_active_sun_bconv[i])
                
                #appodisation, same as for the CCF of the quiet photosphere and spot (see "create_Sun_CCF.py")
                CCF_active_sun[i]  = 1-((-CCF_active_sun[i]+1)*appod)
                if calculate_flux_and_bconv == 1:
                    CCF_active_sun_flux[i]  = 1-((-CCF_active_sun_flux[i]+1)*appod)
                    CCF_active_sun_bconv[i]  = 1-((-CCF_active_sun_bconv[i]+1)*appod)
                #######################

                print 'fitting CCF'
                fit_CCF_active_sun.append(functions.compute_bis(vrad_ccf2,CCF_active_sun[i]))
                if calculate_flux_and_bconv == 1:
                    fit_CCF_active_sun_flux.append(functions.compute_bis(vrad_ccf2,CCF_active_sun_flux[i]))
                    fit_CCF_active_sun_bconv.append(functions.compute_bis(vrad_ccf2,CCF_active_sun_bconv[i]))
        else:
            sstar_active = append(sstar_active,sstar - sspot[i])
            fstar_active_sun.append(fstar_quiet_sun)
            if calculate_flux_and_bconv == 1:
                fstar_active_sun_flux.append(fstar_quiet_sun)
                fstar_active_sun_bconv.append(fstar_quiet_sun)
            if calculate_CCFs == 1:
                CCF_active_sun.append(CCF_quiet_Sun)
                fit_CCF_active_sun.append(fit_CCF_quiet_Sun)
                if calculate_flux_and_bconv == 1:
                    CCF_active_sun_flux.append(CCF_quiet_Sun)
                    fit_CCF_active_sun_flux.append(fit_CCF_quiet_Sun)
                    CCF_active_sun_bconv.append(CCF_quiet_Sun)
                    fit_CCF_active_sun_bconv.append(fit_CCF_quiet_Sun)


    print '###############################################################'
    print 'ENDED calculating integrated spectrum including active region'
    print '###############################################################'
    print

    # Calculates the RV, BIS SPAN and FWHM and plot them as a function of phase
    if calculate_CCFs == 1:
        #functions.calculate_CCF() returns "model,continuum,contrast,span,vrad,fwhm,depth,bis"
        vrad_spectrum = array([fit_CCF_active_sun[i][4] for i in arange(len(fspot_tot))])
        vrad_spectrum -= vrad_spectrum[0]
        bis_span_spectrum = array([fit_CCF_active_sun[i][3] for i in arange(len(fspot_tot))])
        bis_span_spectrum -= mean(bis_span_spectrum)
        fwhm_spectrum = array([fit_CCF_active_sun[i][5] for i in arange(len(fspot_tot))])
        fwhm_spectrum -= min(fwhm_spectrum)
        if calculate_flux_and_bconv == 1:
            vrad_flux_spectrum = array([fit_CCF_active_sun_flux[i][4] for i in arange(len(fspot_tot))])
            vrad_flux_spectrum -= vrad_flux_spectrum[0]
            vrad_bconv_spectrum = array([fit_CCF_active_sun_bconv[i][4] for i in arange(len(fspot_tot))])
            vrad_bconv_spectrum -= vrad_bconv_spectrum[0]
            bis_span_flux_spectrum = array([fit_CCF_active_sun_flux[i][3] for i in arange(len(fspot_tot))])
            bis_span_flux_spectrum -= mean(bis_span_flux_spectrum)
            bis_span_bconv_spectrum = array([fit_CCF_active_sun_bconv[i][3] for i in arange(len(fspot_tot))])
            bis_span_bconv_spectrum -= mean(bis_span_bconv_spectrum)
            fwhm_flux_spectrum = array([fit_CCF_active_sun_flux[i][5] for i in arange(len(fspot_tot))])
            fwhm_flux_spectrum -= min(fwhm_flux_spectrum)
            fwhm_bconv_spectrum = array([fit_CCF_active_sun_bconv[i][5] for i in arange(len(fspot_tot))])
            fwhm_bconv_spectrum -= min(fwhm_bconv_spectrum)


    # print '###############################################################'
    # print 'Generating Figures'
    # print '###############################################################'
    # try:
    #     fig = plt.figure(figsize=(12,3))
    #     fig.subplots_adjust(bottom=0.2)
    #     # plot(freq_line_G2, contrast_line_G2, '.b')
    #     mask_hw = 0.5*dlambda0
    #     plt.errorbar(freq_line_G2, contrast_line_G2, xerr=mask_hw, fmt='o', ms=1.5)
    #     plt.xlabel(r'$\lambda$')
    #     plt.ylabel('Mask contrast')
    #     plt.savefig('lambda.png')
    # except:
    #     print "couldnt make lambda.png"
    #     pass

    # try:
    #     f, (ax1, ax2) = plt.subplots(1,2, figsize=[12, 5])
    #     l, u = 4000, 4010  # range to plot
    #     ax1.set_title('Comparison quiet photosphere and spot spectra')
    #     pselect = (wavelength >= l)  &  (wavelength <= u)
    #     ax1.plot(wavelength[pselect], spectrum[pselect], 'b', label='Quiet')
    #     ax1.plot(wavelength[pselect], spectrum_magn_region[pselect], 'r', label='Spot')
    #     ax1.set_ylabel('Normalized flux')
    #     ax1.set_xlabel('wavelength [A]')
    #     ax1.legend()
    #     ax1.set_xlim(l, u)

    #     mask_hw = mask_hw[select_line_mask]
    #     fselect = (freq_line_G2 >= l)  &  (freq_line_G2 <= u)
    #     ax2.errorbar(freq_line_G2[fselect], contrast_line_G2[fselect], xerr=mask_hw[fselect], fmt='o', ms=1.5)
    #     for l in freq_line_G2[fselect]:
    #         ax2.axvline(l, c='k', alpha=0.5, lw=.5)
    #     ax2.set_ylabel('Mask contrast')
    #     plt.savefig('QuietVsActiveStar.png')
    # except:
    #     print "couldnt make QuietVsActiveStar.png"
    #     pass

    # try:
    #     fig = plt.figure('quiet', figsize=(12,3))
    #     fig.subplots_adjust(left=.1, right=.9, bottom=0.21, top=.9)
    #     plt.title('Quiet Sun integrated spectrum')
    #     plt.plot(wavelength,fstar_quiet_sun,label='Integrated spectrum no activity at full resolution')

    #     plt.figure('quiet')
    #     plt.ylabel('Normalized flux')
    #     plt.xlabel(r'$\lambda$ ($\AA$)')
    #     plt.legend(loc=0)
    #     plt.xlim(4000,4010)

    #     fig = plt.figure('FTS-vs-integrated', figsize=(12,3))
    #     fig.subplots_adjust(left=.1, right=.9, bottom=0.21, top=.9)
    #     plt.title('Comparison quiet solar FTS spectrum and integrated quiet solar spectrum')
    #     plt.plot(wavelength,fstar_quiet_sun/sort(fstar_quiet_sun)[-50],label='Integrated spectrum no activity at selected resolution')
    #     plt.plot(wavelength,spectrum,'r',label='FTS spectrum quiet Sun, full resolution')
    #     plt.ylabel('Normalized flux')
    #     plt.xlabel('wavelength [A]')
    #     plt.legend(loc=0)
    #     plt.xlim(4000,4010)
    #     plt.savefig('QuietSunComp.png')
    # except:
    #     print "couldnt make QuietSunComp.png"
    #     pass

    # if calculate_CCFs == 1:
    #     plt.figure('CCF-quiet')
    #     plt.title('CCF quiet Sun')
    #     plt.plot(vrad_ccf2,CCF_quiet_Sun,color='b',marker='o',ls='')
    #     plt.plot(vrad_ccf2,fit_CCF_quiet_Sun[0],color='b',ls='-')
    #     plt.plot(fit_CCF_quiet_Sun[7],fit_CCF_quiet_Sun[6],'bo')
    #     plt.ylabel('CCF flux')
    #     plt.xlabel('CCF RV [m/s]')
    #     plt.savefig('figure4.png')

    #     plt.figure(1000)
    #     plt.plot(PSI,vrad_spectrum,marker='o',ls='',label='tot')
    #     if calculate_flux_and_bconv == 1:
    #         plt.plot(PSI,vrad_flux_spectrum,marker='o',ls='',label='flux')
    #         plt.plot(PSI,vrad_bconv_spectrum,marker='o',ls='',label='conv. blue.')
    #     plt.ylabel('RV [m/s]')
    #     plt.xlabel('Phase')
    #     plt.legend()
    #     plt.savefig('figure5.png')

    print '###############################################################'
    print 'Writing spectra in file'
    print '###############################################################'

    if SPOT.magn_feature_type == 0:
        str_magn_feature_type = 'spot'
    elif SPOT.magn_feature_type == 1:
        str_magn_feature_type = 'plage'
    lats, longs, Is, amps, starts, lens, gfs = [], [], [], [], [], [], []
    for SPOT in SPOTS:
        lats.append(SPOT.lat)
        longs.append(SPOT.long)
        Is.append(SPOT.I)
        starts.append(SPOT.setup_data[0])
        lens.append(SPOT.setup_data[1])
        amps.append(SPOT.setup_data[2])
        gfs.append(SPOT.setup_data[3])

    # folder location
    # folder = '%s_n%d_maxsize_%.3f'%(str_magn_feature_type,len(SPOTS),max(amps))
    # folder = '/gpfs/group/ebf11/default/SOAP_output/May_runs'
    folder = 'test_runs'
    if os.path.exists(folder):
        pass
    else:
        os.mkdir(folder)

    # output file name
    # str_magn_feature_type, STAR.incl, PROT,SPOT.s, SPOT.lat
    fname = 'lambda-%d-%d-%dyears_%dspots_diffrot_id%d.h5'%(int(floor(wavelength[0])),
        int(ceil(wavelength[-1])), int(round(max(PSI)*PROT/(2*np.pi*365))), len(SPOTS), seed_id)

    sname = folder + '/' + fname

    store = h5py.File(sname, 'w')

    # Star Attrs
    store.attrs['region_type'] = str_magn_feature_type
    store.attrs['incl'] = STAR.incl
    store.attrs['Prot_eq'] = PROT
    store.attrs['Prot_pole'] = PROT_POLE
    store.attrs['Rad_Sun'] = STAR.rad
    store.attrs['Temp'] = STAR.Temp
    store.attrs['Temp_diff_spot'] = STAR.Temp_diff_spot
    store.attrs['limba1'] = STAR.limba1
    store.attrs['limba2'] = STAR.limba2

    # Spot Attrs
    store.attrs['spot_lats'] = lats
    store.attrs['spot_longs'] = longs
    store.attrs['spot_intensities'] = Is
    store.attrs['spot_start_phases'] = starts
    store.attrs['spot_phase_lifetimes'] = lens
    store.attrs['spot_max_sizes'] = amps
    store.attrs['spot_growth_frac'] = gfs

    store.create_dataset('lambdas', data=wavelength, compression=9)
    store['lambdas'].attrs['units'] = 'angstrom'

    # TODO:  Convert the ==1 option to HDF5.
    if calculate_flux_and_bconv == 1:
        ##selection of the file format to write the data
        keys = ['wave','intensity_tot','intensity_flux','intensity_bconv']
        dd = {'wave':wavelength.tolist(),'intensity_tot':fstar_quiet_sun.tolist(),'intensity_flux':fstar_quiet_sun.tolist(),'intensity_bconv':fstar_quiet_sun.tolist()}
        format = '%f\t%f\t%f\t%f\n'
        pyrdb.write_rdb('outputs/'+folder+'/integrated_spectrum_R%i_no_active_region_incl_%.1f_prot_%.1f.csv' % (INST_RESO,STAR.incl,PROT),dd,keys,format) #write the data

        ##selection of the file format to write the data
        for i in arange(len(PSI)):
            keys = ['wave','intensity_tot','intensity_flux','intensity_bconv']
            dd = {'wave':wavelength.tolist(),'intensity_tot':fstar_active_sun[i].tolist(),'intensity_flux':fstar_active_sun_flux[i].tolist(),'intensity_bconv':fstar_active_sun_bconv[i].tolist()}
            format = '%f\t%f\t%f\t%f\n'
            pyrdb.write_rdb('outputs/'+folder+'/integrated_spectrum_R%i_%s_incl_%.1f_prot_%.1f_size_%.3f_lat_%i_phase_%.4f.csv'%(INST_RESO,str_magn_feature_type,STAR.incl,PROT,SPOT.s[0],SPOT.lat,PSI[i]),dd,keys,format) #ecriture les donnees
    else:
        store.create_dataset('quiet', data=fstar_quiet_sun, compression=9)
        store.create_dataset('phases', data=PSI, compression=9)
        store.create_dataset('active', data=fstar_active_sun, compression=9)
        store.create_dataset('msh_covered', data=msh_ph, compression=9)  # MSH covered by spots (not necessarily on visible side)
        store.create_dataset('n_spots', data=n_spots_ph, compression=9)  # number of spots on surface (not necessarily on visible side)
        store.close()

        # Check the store.
        store = h5py.File(sname, 'r')
        print('HDF5 store {}:'.format(sname))
        for name, value in store.attrs.items():
            print '  {}: {}'.format(name, value)
        names = ['lambdas', 'quiet', 'phases', 'active', 'msh_covered', 'n_spots']
        values = [wavelength, fstar_quiet_sun, PSI, fstar_active_sun, msh_ph, n_spots_ph]
        for name, value in zip(names, values):
            print store[name]
            print name, 'match:', (store[name].value == value).all()
        store.close()


if __name__ == '__main__':
    start_time = time.time()
    profile = 0     # do you want to profile the code?
    
    spectral_regions = [3923.87, 6663.56]  # min=3923.87, max=6663.56
    # spectral_regions = [3923.87, 4010.]  # min=3923.87, max=6663.56
    
    if profile == 1:
        from memory_profiler import memory_usage
        mem_usage = memory_usage(main(spectral_regions))
        print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
        print('Maximum memory usage: %s' % max(mem_usage))
    else:
        main(spectral_regions)

    end_time = time.time()
    print "total soap_run.py time used: %f s"%(end_time - start_time)
