#
#  funtions.py
#
#
#  Created by Dumusque Xavier on 3/23/10.
#  Copyright (c) 2010 Private. All rights reserved.
#

import sys,string,glob
import fit2
from numpy import *

"""
def Delta_wavelength
def calc_mask
def calculate_CCF
def compute_bis
"""

#---------------------------------------------------------------------------------------------------------------------

c = 299792458. # speed of light in m / s

#We deal with the light, so we have to use the relativistic case, so that we have a symmetry (if we take a wavelength, it is shifted by 20 km / s, then we shifted the result of -20 km / s, we must find the same thing)
# nu_recu = sqrt ((1-beta) / (1 + beta)) * nu_emmi see wiki doppler effect
# c = lambda * naked
# c / lambda_recu = sqrt ((1-beta) / (1 + beta)) * c / lambda_emmi
# lambda_recu = sqrt ((1 + beta) / (1-beta)) * lambda_emmi
# Delta_lambda = lambda_emmi - lambda_recu = lambda_emmi * (1 - sqrt ((1 + beta) / (1-beta)))
def Delta_wavelength(v,wavelength0):
    beta = v/c
    #we need a - because if the star approaches us, the radiation is emitted with a positive speed, but by difinition of the radial velocity, it corresponds to a nigative VR
    delta_wavelength = wavelength0 * (sqrt((1+beta)/(1-beta))-1)
    return delta_wavelength

#---------------------------------------------------------------------------------------------------------------------


###########################################################################
# We move the template mask, that have holes of 1 pixels, with steps smaller than 1 pixel. The spectrum is also dicontinuous because one have a point each pixel of the detector.
# This code allows to calculate the proportion of flux that will pass through the template when the spectrum sampling is not aligned with the template sampling.
# If one hole of the template is centered on one pixel of the spectrum, the flux returned will be 100% of the spectrum. If the one hole is centered in between 2 pixels, we take 50%
# of the flux of the pixel before and 50% of the flux of the pixel after.
# See the program "test_function_mask.py" for an example.
###########################################################################



###########################################################################
# Move the mask, which has 1 pixel holes with steps smaller than 1 pixel. On the discontinuous values ​​of the spectrum (the spectrum is measured using the CCD
# which is discontinuous (pixels)). This routine, implementee in "synthetic_spectrum.py" allows to calculate the proportion that the mask will pass according to the proportion
# that the hole covers a pixel. If the hole is centered on the pixel, value = pixel flux. If the hole is centered on the intersection between 2 pixels,
# value = 50% flow of the pixel before and 50% of the flux of the pixel after.
# Look at the function test_function_mask.py for an example.
# See the program "test_function_mask.py" for an example.
# Watch the function "test_fit_gauss_with_variation_bissecteur2.py"
###########################################################################

#mask_width = 820 # mask hole in m/s (= pixel width)

def calc_mask(wavelength,wavelength_line,contrast_line,mask_width=820,hole_width=0):

    if type(hole_width) == int:
        hole_width = array([Delta_wavelength(mask_width,wavelength_line[i]) for i in arange(len(wavelength_line))])
    begining_mask_hole = wavelength_line-hole_width/2.
    end_mask_hole = wavelength_line+hole_width/2.

    index_begining_mask_hole,index_end_mask_hole = [],[]
    freq_step_before_mask_hole,freq_step_after_mask_hole = [],[]
    for i in arange(len(wavelength_line)):
        # We select the wavelength (border of the pixel) which is the closest of the position of the center of the hole
        # We select the discontinuous wave length (= edge of the pixels) that is closest to the center position of the mask.
        index_begining_mask_hole.append(searchsorted(wavelength,begining_mask_hole[i]))
        index_end_mask_hole.append(searchsorted(wavelength,end_mask_hole[i])-1) # searchsorted gives us the pixel after and we do -1 because we take the pixel before

        freq_step_before_mask_hole.append(wavelength[index_begining_mask_hole[-1]]-wavelength[index_begining_mask_hole[-1]-1])
        freq_step_after_mask_hole.append(wavelength[index_end_mask_hole[-1]+1]-wavelength[index_end_mask_hole[-1]])

    mask = zeros(len(wavelength),'d')
    a = index_begining_mask_hole
    b = index_end_mask_hole
    #fraction of the pixel before and after that is covered by the hole
    fraction_pixel_before_mask_hole = abs(wavelength[a] - begining_mask_hole)/freq_step_before_mask_hole
    fraction_pixel_after_mask_hole  = abs(wavelength[b] - end_mask_hole)/freq_step_after_mask_hole

    for i in arange(len(index_begining_mask_hole)):
        # the contrast_line is 1 minus the flux ratio between the line and the continuous just next to it (so the flux difference between blue and red is already indirectly included)
        mask[a[i]:b[i]] = [contrast_line[i]]*(b[i]-a[i]) #on cree un vecteur de longueur (b[i]-a[i])
        mask[a[i]-1] = contrast_line[i]*fraction_pixel_before_mask_hole[i]
        mask[b[i]] = contrast_line[i]*fraction_pixel_after_mask_hole[i]

    return mask


#---------------------------------------------------------------------------------------------------------------------


def calculate_CCF(vrad,wavelength,spectrum,wavelength_line,contrast_line,verbose=0,extrapolation=200):

    CCF = []

    span_wavelength = max(wavelength) - min(wavelength)
    step_wavelength = 1*median(wavelength[1:]-wavelength[:-1])
    wavelength_before = arange(wavelength[0]-extrapolation,wavelength[0],0.01)
    wavelength_after = arange(wavelength[-1],wavelength[-1]+extrapolation,0.01)
    wavelength_extrapol = concatenate([wavelength_before,wavelength,wavelength_after])

    if wavelength_extrapol[0] - wavelength_line[0] > 0:
        print
        print
        print 'BE CAREFULL:'
        print ' You have to increase the extrapolation factor because'
        print 'the spectrum is thiner than the width of the mask, which should be the opposite'
        print

    for i in arange(len(vrad)):

        if i%100 == 0 and verbose==1 : print i,' over ',len(vrad)

        wavelength_line_shift = wavelength_line*(1+vrad[i]/c) # calculate the position of the spectral lines as a function of the template speed
                                                              # calculates the position of the lines according to the speed of the mask 'spect_type_mask'

        mask_corr = calc_mask(wavelength_extrapol,wavelength_line_shift,contrast_line) # calculation of the mask 'spect_type_mask'
        mask_corr = mask_corr[len(wavelength_before):-len(wavelength_after)]

        CCF.append(sum(spectrum*mask_corr)) # sum of the mask + spectrum = CCF, outside a magnetic region

    return array(CCF)


#---------------------------------------------------------------------------------------------------------------------


def compute_bis(RV,spectrum,err=None):
    if err:
        pass
    else:
        err = ones(len(spectrum),'d')/len(spectrum)
    RV = array(RV,'d')
    spectrum = array(spectrum,'d')
    mod,c,k,v0,fwhm,sig_c,sig_k,sig_v0,sig_fwhm,iter,span,ff,ee,len_depth = fit2.gauss_bis(RV,spectrum,err,ones(10000,'d'))
    model = mod
    continuum = c
    contrast = abs(k)
    vrad = v0
    #ee is the full contrast of the line, 0 being the top and 1 being the bottom, you have to scale this to the real contrast of the line
    depth = 1-ee[:len_depth]*contrast
    bis = ff[:len_depth]
    return model,continuum,contrast,span,vrad,fwhm,depth,bis


#---------------------------------------------------------------------------------------------------------------------
