from numpy import array, zeros, where, arange, searchsorted, r_, concatenate, mean, pi
from pylab import *
import _starspot
from scipy import signal
import functions
RSUN = 695000. # [km]

class Star:
    def __init__(self, vrot, vrot_pole, incl, limba1, limba2, modif_bis_quad, modif_bis_lin, modif_bis_cte, psi, Temp, Temp_diff_spot, rad=1., gamma=0.):
        self.vrot  = vrot    # Rotation velocity (at equator)
        self.vrot_pole = vrot_pole # Rotation velocity at pole
        self.incl  = incl    # Inclination
#               self.limba = limba   # Linear limb-darkenig coeff.
        self.limba1 = limba1   # linear limb-darkenig coeff.
        self.limba2 = limba2   # quadratic limb-darkenig coeff.
        self.modif_bis_quad = modif_bis_quad
        self.modif_bis_lin = modif_bis_lin
        self.modif_bis_cte = modif_bis_cte
        self.psi   = psi     # Phase
        self.Temp  = Temp    # Effective Temp of the star
        self.Temp_diff_spot = Temp_diff_spot # difference in Temp between the Effective Temp and the spot Temp
        self.rad   = rad     # Radius [Msol]
        self.gamma = gamma   # Radial vel. [km/s]
        self.p     = 2.*pi*self.rad*RSUN/self.vrot/86400.

    def calc_maps(self, grid):
        self.Fmap, self.Vmap = _starspot.starmap(self.vrot, self.incl, self.limba1, self.limba2, grid)
        self.Fmap = array(self.Fmap)
        self.Vmap = array(self.Vmap)

class Spot:
    def __init__(self, long, lat, size, I, magn_feature_type, setup_data, check=False):
        self.long = long    # Longitude
        self.lat  = lat     # Latitude
        self.s    = size    # Size
        self.I    = I       # Intensity of active region
        self.magn_feature_type = magn_feature_type # 0 if spot, 1 if plage
        self.setup_data = setup_data # additional info used to uniquely generate spots

        self.check = check

    def calc_maps(self, star, grid, nrho):
        xyz  = _starspot.spot_init(self.s[0], self.long, self.lat, star.incl, nrho)
        xyz2 = _starspot.spot_phase(xyz, star.incl, nrho, star.psi)
        self.visible, self.iminy, self. iminz, self.imaxy, self.imaxz = \
        _starspot.spot_area(xyz2, nrho, grid)
        if self.visible:
            Fmap, Vmap = _starspot.spotmap(star.vrot, star.incl, star.limba1, star.limba2, grid,
                          self.s[0], self.long, star.psi, self.lat,
                          self.iminy, self. iminz,
                          self.imaxy, self.imaxz, self.magn_feature_type, star.Temp, star.Temp_diff_spot)
#                       Fmap, Vmap = _starspot.spotmap(star.vrot, star.incl, star.limba1, star.limba2, grid,
#                                     self.s, self.long, star.psi, self.lat,
#                                     self.I, self.iminy, self. iminz,
#                                     self.imaxy, self.imaxz)
            self.Vmap = array(Vmap)
            self.Fmap = array(Fmap)
        else:
            self.Vmap = zeros((0,0), 'd')
            self.Fmap = zeros((0,0), 'd')

class Ccf:
    def __init__(self, vrad, intensity, width, step, star):
        self.vrad       = vrad
        self.intensity  = intensity
        self.width      = width # the CCF must be defined between -width and + width
        self.step       = step

        self.n          = len(self.vrad)
        # The CCF has a width of 20 km / s. Not against this CCF is for a zero rotation speed (= center of the star)
        # If you are on an edge, or the rotation speed is non-zero, you must shuffle all the CCF according to the speed
        # n_v gives the number of points of the CCF (= sampling) that is necessary to take into account that the CCF will shift according to the speed
        # we put 1.1 * vrot to take the margin
        # we want n_v to be odd, so that the CCF is symmetric, and so have a point of symmetry a v = 0
        if round(((1.1*star.vrot)/self.step)*2)%2 == 0:
            self.n_v        = int(self.n + round(((1.1*star.vrot)/self.step)*2))
        else:
            self.n_v        = int(self.n + round(((1.1*star.vrot)/self.step)*2)+1)
        # v_interval gives the speed interval of the CCF, between -v_interval and + v_interval
        self.v_interval = self.step*(self.n_v-1) / 2. #n_v gives the number of points, (n_v-1) gives the number of intervals. We must use the 2nd value because we multiply by the step (= self.step)

#class Ccf_spot:
#    def __init__(self, prof, sigma, mean, width, step, star):
#       self.prof       = prof
#       self.sigma      = sigma
#       self.mean       = mean
#       self.width      = width
#       self.step       = step
#
#       self.v          = arange(-self.width, self.width+self.step/10., self.step) # largeur de toute la CCF, pas seulement de la Gaussienne
#       self.n_v        = int((self.width+1.1*star.vrot)/self.step+0.55) # nombre de vitesse echantillonnee en fonction de l elargissement du a la rotation width+1.1*vrot (pk 1.1 ici et non pas 1)
#       self.v_interval = self.step*self.n_v
#       self.n_v        = self.n_v * 2

class RVData:
    def __init__(self, jdb=None, vrad=None, svrad=None, bis=None):
        self.jdb       = jdb
        self.vrad      = vrad
        self.svrad     = svrad
        self.bis       = bis

    def computePhases(self, star):
        Rstar = star.rad*RSUN # [km]
        P = (2.*pi*Rstar)/star.vrot # [s]
        P = P/86400. # [day]
        print P
        dt = (self.jdb-self.jdb[0])/P
        phase = dt-dt.astype(int)
        phase = where(phase<0,phase+1,phase)
        return phase

class PhotData:
    def __init__(self, jdb=None, phot=None):
        self.jdb       = jdb
        self.phot      = phot



def AllGraph3(nrho, grid, star, psi, ccf, ccf_spot, choice_magn_region, inst_reso, spots=[]):
#    print 'star.vrot',star.vrot
#    print 'ccf.vrad',ccf.vrad
#    print 'ccf.n_v',ccf.n_v/2.
#    print 'ccf.n_v',ccf.n_v
#    print 'ccf.v_interval',ccf.v_interval

    ##for starspot_old.c
    #fstar, sstar = _starspot.itot(star.vrot, star.incl, star.limba1, star.limba2, star.modif_bis_quad, star.modif_bis_lin, star.modif_bis_cte, grid,ccf.intensity, ccf.v_interval, ccf.n_v, ccf.n)
    ##for starspot_new.c
    #fstar, sstar = _starspot.itot(star.vrot, star.incl, star.limba1, star.limba2, star.modif_bis_quad, star.modif_bis_lin, star.modif_bis_cte, grid, ccf.v_interval, ccf.n_v, ccf.n)
    ##for starspot_fast.c
    fstar, sstar = _starspot.itot(star.vrot, star.incl, star.limba1, star.limba2, star.modif_bis_quad, star.modif_bis_lin, star.modif_bis_cte, grid, ccf.vrad, ccf.intensity, ccf.v_interval, ccf.n_v, ccf.n)
    if len(spots)==0: sstar = array([sstar for i in psi])
    if len(spots)==0: fstar = array([fstar for i in psi])

    print 'spots',spots

    # initialisation of the variables
    fstar_flux=fstar
    fstar_bconv=fstar
    fstar_tot=fstar
    fstar_quiet = fstar

    for spot in spots: #on fait une boucle pour chaque spot
        if not spot.check: continue # spot.check = 1 si l on selection le spot et 0 sinon
        if spot.visible<4: # spot.visible donne le nombre de spot visible sur l etoile
            xyz = _starspot.spot_init(spot.s, spot.long, spot.lat, star.incl, nrho)
            # fspot_flux and all the variable are calculated for each zone of the grid defined (200 zones normally). Each zone fspot_flux[i] corresponds to the spectrum
            # of the star "fstar". len(fspot[[i]) = len(fstar)

            ##for starspot_old.c
#            fspot_flux,fspot_bconv,fspot_tot,sspot = _starspot.spot_scan_npsi(xyz, nrho, psi, len(psi), star.vrot, star.incl,\
#                                                        star.limba1, star.limba2, grid, \
#                                                        ccf.intensity, ccf_spot.intensity, ccf.v_interval, ccf.n_v, ccf.n,\
#                                                        spot.s, spot.long, spot.lat, spot.I)
#                       ##for starspot_new.c
#            fspot_flux,fspot_bconv,fspot_tot,sspot = _starspot.spot_scan_npsi(xyz, nrho, psi, len(psi), star.vrot, star.incl,\
#                                                        star.limba1, star.limba2, grid,\
#                                                        ccf.v_interval, ccf.n_v, ccf.n,\
#                                                        spot.s, spot.long, spot.lat, spot.I)
#                       ##for starspot.c

            fspot_flux,fspot_bconv,fspot_tot,sspot = _starspot.spot_scan_npsi(xyz, nrho, psi, len(psi), star.vrot, star.vrot_pole, star.incl,\
                                                        star.limba1, star.limba2, star.modif_bis_quad, star.modif_bis_lin, star.modif_bis_cte, grid,\
                                                        ccf.vrad, ccf.intensity, ccf_spot.intensity, ccf.v_interval, ccf.n_v, ccf.n,\
                                                        spot.s, spot.long, spot.lat, spot.magn_feature_type, star.Temp, star.Temp_diff_spot)

            sstar=sstar-sspot
            fstar_flux=fstar_flux-fspot_flux
            fstar_bconv=fstar_bconv-fspot_bconv
            fstar_tot=fstar_tot-fspot_tot

        #istart, iend = searchsorted(ccf.vrad,[ccf.rv[0],ccf.rv[-1]])
    istart = (ccf.n_v-len(ccf.vrad))/2 # n_v est impair d apres notre definition, tout comme len(ccf.vrad), donc la difference est paire et on peut diviser par 2
    iend   = istart+len(ccf.vrad)
    fstar_flux = fstar_flux[:,istart:iend] # car fstar_flux va de -20-istart a +20+istart car la CCF a ete elargie pour prendre en compte la rotation
    fstar_flux = array([f/max(f) for f in fstar_flux], dtype='d')
    fstar_bconv = fstar_bconv[:,istart:iend]
    fstar_bconv = array([f/max(f) for f in fstar_bconv], dtype='d')
    fstar_tot = fstar_tot[:,istart:iend]
    fstar_tot = array([f/max(f) for f in fstar_tot], dtype='d')

    fstar_quiet = fstar_quiet[istart:iend]
    fstar_quiet = fstar_quiet/max(fstar_quiet)
    fspot_flux = fspot_flux[:,istart:iend]
    if choice_magn_region == '0':
        fspot_flux = array([f-max(f) for f in fspot_flux], dtype='d')
    elif choice_magn_region == '1':
        fspot_flux = array([f-min(f) for f in fspot_flux], dtype='d')
    fspot_bconv = fspot_bconv[:,istart:iend]
#    fspot_bconv = array([f/where(f[0]==0,1,f[0]) for f in fspot_bconv], dtype='d')


    if inst_reso != 'full':

        c = 299792458. # vitesse de la lumiere en m/s
        HARPS_resolution = inst_reso # resolution R = lambda / Delta(lambda)
        HARPS_inst_profile_FWHM = c/HARPS_resolution/1000. #Resolution = c/Delta_v -> Delta_v = c/R
        HARPS_inst_profile_sigma = HARPS_inst_profile_FWHM/(2*sqrt(2*log(2)))
        Gaussian_low_reso = exp(-ccf.vrad**2/(2*(HARPS_inst_profile_sigma)**2))

        fstar_quiet_tmp = signal.convolve(-fstar_quiet+1,Gaussian_low_reso,'same')
        fstar_quiet = 1-fstar_quiet_tmp*(1-min(fstar_quiet))/max(fstar_quiet_tmp) # normalization

        for i in arange(len(fstar_flux)):
            fstar_flux_tmp = signal.convolve(-fstar_flux[i]+1,Gaussian_low_reso,'same')
            fstar_flux[i] = 1-fstar_flux_tmp*(1-min(fstar_flux[i]))/max(fstar_flux_tmp)
            fstar_bconv_tmp = signal.convolve(-fstar_bconv[i]+1,Gaussian_low_reso,'same')
            fstar_bconv[i] = 1-fstar_bconv_tmp*(1-min(fstar_bconv[i]))/max(fstar_bconv_tmp)
            fstar_tot_tmp = signal.convolve(-fstar_tot[i]+1,Gaussian_low_reso,'same')
            fstar_tot[i] = 1-fstar_tot_tmp*(1-min(fstar_tot[i]))/max(fstar_tot_tmp)

    depth_flux = []; span_flux = []; vrad_flux = []; fwhm_flux = []
    depth_bconv = []; span_bconv = []; vrad_bconv = []; fwhm_bconv = []
    depth_tot = []; span_tot = []; vrad_tot = []; fwhm_tot = []
    ##bis_flux = []; bis_bconv = []; bis_tot = [];
#    figure(2); clf()
    for i in arange(len(fstar_tot)):
        #plot(ccf.vrad,cor-fstar[0])
        #if i%20 == 0 : print i,' over ',len(fstar_tot)
        a_flux = functions.compute_bis(ccf.vrad,fstar_flux[i])
        depth_flux.append(a_flux[0]); span_flux.append(a_flux[2]); vrad_flux.append(a_flux[3]); fwhm_flux.append(abs(a_flux[4]))
        a_bconv = functions.compute_bis(ccf.vrad,fstar_bconv[i])
        depth_bconv.append(a_bconv[0]); span_bconv.append(a_bconv[2]); vrad_bconv.append(a_bconv[3]); fwhm_bconv.append(abs(a_bconv[4]))
        a_tot = functions.compute_bis(ccf.vrad,fstar_tot[i])
        depth_tot.append(a_tot[0]); span_tot.append(a_tot[2]); vrad_tot.append(a_tot[3]); fwhm_tot.append(abs(a_tot[4]))
        ##bis_flux.append(a_flux[1]); bis_bconv.append(a_bconv[1]); bis_tot.append(a_tot[1]);

    depth_flux = array(depth_flux); span_flux = array(span_flux); vrad_flux = array(vrad_flux); fwhm_flux = array(fwhm_flux)
    depth_bconv = array(depth_bconv); span_bconv = array(span_bconv); vrad_bconv = array(vrad_bconv); fwhm_bconv = array(fwhm_bconv)
    depth_tot = array(depth_tot); span_tot = array(span_tot); vrad_tot = array(vrad_tot); fwhm_tot = array(fwhm_tot)
    ##bis_flux = array(bis_flux); bis_bconv = array(bis_bconv); bis_tot = array(bis_tot);
    sstar /= max(sstar)

    return sstar, fstar_quiet,fspot_flux,fspot_bconv, vrad_flux, vrad_bconv, vrad_tot, fstar_flux, fstar_bconv, fstar_tot, span_flux, span_bconv, span_tot, fwhm_flux, fwhm_bconv, fwhm_tot, depth_flux, depth_bconv, depth_tot

def PhotVradSpan(nrho, grid, star, psi, ccf, ccf_spot,choice_magn_region, spots=[], inst_reso='full'):
    results = AllGraph3(nrho, grid, star, psi, ccf, ccf_spot, choice_magn_region, inst_reso, spots)
    return results
