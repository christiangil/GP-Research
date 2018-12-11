"""
Made by Ari Silburt

This file contains all the important spot and phase parameters and functions
needed for soap_run.py. Existing spot/phase parameters have been removed from 
config.cfg.
"""
import numpy as np
from soap2 import ooapi
import spot_functions as sf

#############-PARAMETERS-##############
#######################################
# obs_get_phases_nightly() Star Phase parameters
obs_n_years = 1  # 5/365.  # 3                      # number of years to observe for
obs_per_night = 2  # 2                      # number of obs per night
obs_frac_in_day = 1./3.                     # observing window = 8 hours per day
obs_min_spread_in_day = 1./24.              # min time spread required for obs in a night

# get_spots() Spot Parameters - except N_spots, all variables are either arrays of
# size N_spots or None (None indicates randomly generates values).
spot_density = 109  # 4                            # avg no. of spot initializations per rotation period
spot_start, N_spots = sf.start_times(spot_density, obs_n_years) # starting times of spots (years)
spot_Long = 360*np.random.random(N_spots)   # spot longitudes (degrees)
spot_Lat = sf.spot_latitudes(N_spots)       # spot latitudes (degrees)
# spot_Amp = sf.spot_amplitudes(N_spots)      # spot max amplitudes (units of R_star)
spot_Amp, spot_Types = sf.spot_amplitudes(N_spots)      # spot max amplitudes (units of R_star)
# spot_len = sf.spot_lifetime(N_spots)        # lifetime (length) of spot (units of years)
# print N_spots
spot_gf = [0.1]*N_spots  # [0.33]*N_spots   # Ratio of growth/(growth + decay) of spot. 0-1 range # from Howard 1992, referenced in Borgniet 2015
spot_len = sf.spot_lifetime(spot_Amp, spot_Types, spot_gf)        # lifetime (length) of spot (units of years)
spot_func = 'parabolic'                      # Spot type - 'triangle' (growth/decay) or 'constant', added 'parabolic'

##############-FUNCTIONS-##############
#######################################
def get_spots(PSI, STAR, GRID, NRHO, PROT, seed_id):
    np.random.seed(seed_id)
    spots = []
    msh_covered = np.zeros(len(PSI))
    n_current_spots = np.zeros(len(PSI))
    # print N_spots
    for i in range(N_spots):
        # star spot growth/decay parameters
        longitude = spot_Long[i] if spot_Long[i] else 360*np.random.random()
        latitude = spot_Lat[i] if spot_Lat[i] else np.random.uniform(-30,30)
        start = spot_start[i] if spot_start[i] else np.random.uniform(0,obs_n_years-(PROT/365))
        length = spot_len[i] if spot_len[i] else np.random.uniform(PROT/2,4*PROT)
        amp = spot_Amp[i] if spot_Amp[i] else np.random.uniform(0,0.1)
        gf = spot_gf[i] if spot_gf[i] else np.random.uniform(0.1, 1 / 9.)  # np.random.uniform(0.2,0.4)
        
        # convert time to phase
        start_ph = sf.time_to_phase(PROT, start, 'year')
        length_ph = sf.time_to_phase(PROT, length, 'year')
        size_ph = sf.spot_size_gen(PSI, start_ph, length_ph, amp, gf, spot_func)
        #print size_ph
        if size_ph is not None:
            SPOT = ooapi.Spot(long = longitude,
                              lat  = latitude,
                              size = size_ph,
                              I    = 0.5,
                              magn_feature_type = 0,
                              setup_data = [start, length, amp, gf]) # store in attrs when writing hdf5
                               
            SPOT.calc_maps(STAR, GRID, NRHO)
            spots.append(SPOT)
            for j in range(len(PSI)):
            	if float(size_ph[j]) != 0.:
            		n_current_spots[j] += 1
            msh_covered += sf.r_star2msh(size_ph)
    return spots, msh_covered, n_current_spots

def obs_get_phases_nightly(PROT, seed_id):
    np.random.seed(seed_id)
    obs_frac_in_phase = sf.time_to_phase(PROT, obs_frac_in_day, 'day')
    obs_min_spread_in_phase = sf.time_to_phase(PROT, obs_min_spread_in_day, 'day')
    current_phase = 0
    phases = np.zeros(0)
    for i in range(int(obs_n_years*365)):
        phases_night = np.zeros(0)
        phases_night = np.append(phases_night, np.random.uniform(current_phase, current_phase + obs_frac_in_phase))
        for j in range(1, obs_per_night):
            ph = np.random.uniform(current_phase, current_phase + obs_frac_in_phase)
            while np.any(np.abs(phases_night - ph) < obs_min_spread_in_phase):
                ph = np.random.uniform(current_phase, current_phase + obs_frac_in_phase)
            phases_night = np.append(phases_night, ph)
    
        current_phase += 2*np.pi/PROT   # move current phase ahead by rad_per_day
        phases = np.concatenate((phases, np.sort(phases_night)))
    return phases


############OLD##################
#def get_phases_simple():
#    #phases = np.arange(min_ph, max_ph, interval_ph)
#    phases = np.linspace(min_ph, max_ph, N_obs)
#    offsets = np.random.uniform(-jitter_ph, jitter_ph, len(phases))
#    return phases + offsets

#spot_start = [1,10,20,30]   # starting time of spot growth (units of years)
#spot_len = [15]*N_spots     # lifetime (length) of spot (units of years)
## get_phases() Star Phase parameters
#min_ph = -10
#max_ph = 10
#N_obs = 500         # number of (evenly spaced) observations
##interval_ph = 0.01   # spacing of observations
#jitter_ph = 0.01     # amplitude of random jitter added to observation time
