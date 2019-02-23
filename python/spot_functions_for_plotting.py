import numpy as np
import scipy.stats as stats
from scipy.special import erfinv, erf

def time_to_phase(PROT, time, time_unit='day'):
    """Converts units of time to units of phase based off the 
        equatorial rotation period of the star.
        
        Parameters
        ----------
        PROT : double
            Equatorial rotation period of star
        time : float or array
            time
        time_unit : string
            units of 'time' variable, can be hours, day or year
        
        Returns
        -------
        phase : float or array
            phase
        """
    phase = (2 * np.pi / PROT) * time
    if time_unit == 'day':
        return phase
    elif time_unit == 'year':
        return phase * 365.25
    elif time_unit == 'hour':
        return phase / 24.
    else:
        print("**INCORRECT TIME_UNIT SPECIFIED**")


def phase_to_time(PROT, phase, time_unit='day'):
    """Converts units of phase to time based off the 
        equatorial rotation period of the star.
        
        Parameters
        ----------
        PROT : double
            Equatorial rotation period of star
        phase : float or array
            phase
        time_unit : string
            units of 'time' variable, can be hours, day or year
        
        Returns
        -------
        time : float or array
            time
        """
    time = phase / (2 * np.pi / PROT)
    if time_unit == 'day':
        return phase
    elif time_unit == 'year':
        return phase / 365.25
    elif time_unit == 'hour':
        return phase * 24.
    else:
        print("**INCORRECT TIME_UNIT SPECIFIED**")

def spot_latitudes(N_spots, sigma=7.):
    """Randomly generates spot latitudes from a truncated 
        normal distribusion (prevent > 90 degree latitudes).
        
        Parameters
        ----------
        N_spots : int
            Number of spots to generate latitudes for.
        sigma : float
            Sigma of normal distribution.
        
        Returns
        -------
        latitudes : array (floats)
            Latitudes for the spots
        """
    mu, lower, upper = 16., -90., 90.  # prevent latitudes > 90
    lats = stats.truncnorm((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma).rvs(N_spots)
    sgn = np.sign(0.5 - np.random.random(N_spots))
    latitudes = lats*sgn
    return latitudes

def baumann_draw(types, mu=np.array([46.51, 90.24]), sigma_a=np.array([2.14, 2.49])):
    # drawing a random starspot area (MSH) from weird Baumann 2005 pseudo "log-normal" pdf (https://arxiv.org/pdf/astro-ph/0510516.pdf) eq. 2
    # following this methodology (https://www.comsol.com/blogs/sampling-random-numbers-from-probability-distribution-functions/)
    sigma = np.sqrt(np.log(sigma_a))
    draws = np.exp(np.sqrt(2) * erfinv(2 * np.random.uniform(size=len(types)) - 1))
    for i in range(len(types)):
        ind = types[i]
        # a simple way to prevent spots less than 10 MSH
        while draws[i] < 10:    
            draws[i] = mu[ind] * ((np.exp(sigma[ind]) * draws[i]) ** sigma[ind])
    return draws


def pillet_draw(types, log_mu=np.array([2.619, 3.373]), sigma=np.array([0.806, 0.869])):
    # drawing a random starspot decay rate (MSH/day) from Pillet 1993 log-normal pdf (http://adsabs.harvard.edu/abs/1993A%26A...274..521M) eq. 5
    # following this methodology (https://www.comsol.com/blogs/sampling-random-numbers-from-probability-distribution-functions/)
    mu = np.exp(log_mu)
    draws = np.exp(np.sqrt(2) * erfinv(2 * np.random.uniform(size=len(types)) - 1))
    for i in range(len(types)):
        ind = types[i]
        # a simple way to prevent spots decays less than 3 MSH/day or greater than 200 MSH/day
        while ((draws[i] < 3) or (draws[i] > 200)): 
            draws[i] = mu[ind] * (draws[i] ** sigma[ind])
    return draws


# from SOAP config file...
# active region's size [Rstar]. If you want a size S1 in area of the visible hemisphere, you have to put here sqrt(2*S1), because
# S1 = Area_spot/Area_visible_hemisphere = pi*(size1*Rstar)**2/(2*pi*Rstar**2) = size1**2/2 -> size1 = sqrt(2*S1).
# For 0.1% of the visible hemisphere, put 0.045
def msh2r_star(msh):
    return 1e-3 * np.sqrt(2 * msh)


def r_star2msh(r_star):
    return r_star * r_star * 5e5


# def spot_amplitudes(N_spots, mu=11.8, sigma=2.55):
def spot_amplitudes(N_spots, mu=np.array([46.51, 90.24]), sigma_a=np.array([2.14, 2.49]), types=-1):
    """Randomly generates spot amplitudes from a log-normal
        distribusion. Model from maximum development method of
        "On the size distribution of sunspot groups in the Greenwich
        sunspot record 1874-1976" Baumann and S.K. Solanki (2005).
        See Table 1
        
        Parameters
        ----------
        N_spots : int
            Number of spots to generate latitudes for.

        
        Returns
        -------
        sizes : array (floats)
            Max amplitude for each spot, units of R_star
        """

    # generate a list deciding whether each spot will be an individual spot or a "complex spot group", 0 for individual spot, 1 for group
    if types == -1:
        types = np.round(np.random.uniform(size=N_spots))
        types = types.astype(int)


    # micro_solar_hemispheres = np.exp(np.random.normal(np.log(mu), np.log(sigma), N_spots))  # wrong!
    micro_solar_hemispheres = baumann_draw(types)
    
    # MSH (area) to R_star (length) converter. Conversion from
    # 'On Sunspot and Starspot Lifetimes' - https://arxiv.org/pdf/1409.4337.pdf
    # Bradshaw and Hartigan 2014
    # sizes = np.sqrt(micro_solar_hemispheres*17682025)/695500.0  # old and wrong, 6.04e-3 sqrt(MSH) per Rsun: off by factor of 2.41

    # MSH = 1e-6 1/2 4 pi Rsun^2 = 2e-6 pi Rsun^2  # 1/million of half of star's surface area
    # sizes = np.sqrt(micro_solar_hemispheres * 2e-6 * np.pi)  # 2.51e-3 sqrt(MSH) per Rsun
    sizes = msh2r_star(micro_solar_hemispheres)
    return sizes, types

def start_times(spot_density, obs_n_years):
    """Generates random starting spot times (in years)
        
        Parameters
        ----------
        spot_density : float
            Avg. # of spot initializations per rotation period
        obs_n_years : float
            Number of years of observation
        
        Returns
        -------
        start : array
            starting time (in years) for each spot
        """
    # config = ConfigParser.ConfigParser()
    # config.read("config.cfg")
    # PROT = float(config.get('star','prot' ))    # rotation period of sun (at equator)
    PROT = 25.05
    N_spots = int(365.25 * spot_density * obs_n_years / PROT)
    starts = np.sort(np.random.random(N_spots) * obs_n_years)
    return starts, N_spots

def spot_lifetime(sizes, types, Gf):
    """Generates random lifetimes for each spot (in years)
        
        Parameters
        ----------
        N_spots : int
            Number of spots to generate
        
        Returns
        -------
        lifetimes : array
            Lifetime (in years) for each spot
        """
    # lifetimes = []
    # for i in range(spot_N):
    #     r = np.random.random()
    #     if r < 0.5:
    #         lt = np.random.uniform(1./24., 2.)/365  # 1 hour - 2 days
    #     elif (r >= 0.5) and (r < 0.9):
    #         lt = np.random.uniform(2., 11.)/365     # 2 - 11 days
    #     else:
    #         lt = np.random.uniform(11., 60.)/365    # 11 - 60 days
    #     lifetimes.append(lt)
    
    # add growth time
    time_mod = np.divide(Gf, np.ones(len(Gf)) - Gf)  # 1/(1/Gf - 1)

    lifetimes = np.multiply(time_mod, np.divide((sizes ** 2)/(2e-6 * np.pi), pillet_draw(types)) / 365.25)  # convert sizes to MSH, divide by decay rate, convert to years

    return lifetimes

def spot_size_gen(phases, start_ph, len_ph, A, Gf, spot_func='parabolic'):
    """Generates a spot that can grow/decay in e.g. a triangular fashion
        fashion.
        
        Parameters
        ----------
        phases : array
            Phases over which spot rises and falls
        start_ph : float
            Starting phase that spot starts to appear.
        len_ph : float
            Length (in units of phase) that spot persists for.
        A : float
            Maximum amplitude of spot. If you want a spot with area S1 of 
            the visible hemisphere, set A=sqrt(2*S1), since:
            S1 = Area_spot/Area_visible_hemisphere 
               = pi*(size1*Rstar)**2/(2*pi*Rstar**2) 
               = size1**2/2 -> size1 = sqrt(2*S1)
            For 0.1% of the visible hemisphere, put 0.045
        Gf : float
            Ratio of growth/(growth + decay) of spot. Range between 0-1.
        spot_func : string
            Type of spot to build. Currently triangle and constant (no 
            growth/decay) available.
        
        Returns
        -------
        spot : array
            Size of spot for each phase.
        """
    if spot_func == 'constant':
        return A*np.ones(len(phases))

    # this is broken, assumes that phases grow linearly with `index, and that all spots start and stop at exact observation indices
    elif spot_func == 'triangle':
        if start_ph < phases[0]:
            print("**start phase for spot must be in the range of passed phases, **SKIPPING**")
            return None
        elif start_ph + Gf*len_ph > phases[-2]:
            print("**starspot phases severely outside observing range, **SKIPPING**")
            return None
        
        phases = np.asarray(phases)
        
        # reference points
        fin_ph = start_ph + len_ph
        start = np.argmin(np.abs(phases - start_ph)) # starting index of spot
        middle = np.argmin(np.abs(phases - (Gf*len_ph + start_ph))) # split point for spot
        end = np.argmin(np.abs(phases - fin_ph)) # ending index of spot

        # growth of spot
        Ng = len(phases[start:middle])
        spotg = A*np.linspace(0, 1, Ng)

        # decay of spot
        if start_ph + len_ph >= phases[-1]:  # spot decays to 0 outside phase range
            Nd = len(phases[middle:end+1])
            decay_amp = 1 - np.abs(phases[-1] - phases[middle])/float(np.abs(fin_ph - phases[middle]))
            end_padding = np.zeros(0)
        else:
            Nd = len(phases[middle:end])
            decay_amp = 0
            end_padding = np.zeros(len(phases[end:]))
        spotd = A*np.linspace(1, decay_amp, Nd)

        # sizes over total range
        spot = np.concatenate((np.zeros(len(phases[0:start])),
                               np.concatenate((spotg, spotd)), end_padding))
        return spot

    elif spot_func == 'parabolic':
        if start_ph < phases[0]:
            print("**start phase for spot must be in the range of passed phases, **SKIPPING**")
            return None
        elif start_ph + Gf*len_ph > phases[-2]:
            print("**starspot phases severely outside observing range, **SKIPPING**")
            return None
        
        phases = np.asarray(phases)
        
        # reference points
        fin_ph = start_ph + len_ph
        mid_ph = start_ph + (Gf * len_ph)
        # start = np.argmin(np.abs(phases - start_ph)) # starting index of spot
        # middle = np.argmin(np.abs(phases - (Gf*len_ph + start_ph))) # split point for spot
        # end = np.argmin(np.abs(phases - fin_ph)) # ending index of spot

        # # growth of spot
        # Ng = len(phases[start:middle])
        # spotg = A * np.linspace(0, 1, Ng)

        # # decay of spot
        # if start_ph + len_ph >= phases[-1]:  # spot decays to 0 outside phase range
        #     Nd = len(phases[middle:end+1])
        #     decay_amp = 1 - np.abs(phases[-1] - phases[middle])/float(np.abs(fin_ph - phases[middle]))
        #     end_padding = np.zeros(0)
        # else:
        #     Nd = len(phases[middle:end])
        #     decay_amp = 0
        #     end_padding = np.zeros(len(phases[end:]))
        # spotd = A*np.linspace(1, decay_amp, Nd)

        # # sizes over total range
        # spot = np.concatenate((np.zeros(len(phases[0:start])),
        #                        np.concatenate((spotg, spotd)), end_padding))

        # this is linear decay (and growth because im lazy) in radius space which becomes parabolic decay in area space
        spot = np.zeros(len(phases))
        for i in range(len(phases)):
            phase = phases[i]
            if phase > start_ph:
                if phase < mid_ph:
                    spot[i] = A * (phase - mid_ph) / (Gf * len_ph)
                elif phase < fin_ph:
                    spot[i] = A * (fin_ph - phase) / ((1 - Gf) * len_ph)

        return spot

# testing purposes
if __name__ == '__main__':
    s = start_times(2, 1)
    print(s)
