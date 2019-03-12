import numpy as np
import scipy.stats as stats
# from scipy.special import erfinv, erf

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

def spot_latitudes(N_spots, sigma=7.3):
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
    mu, lower, upper = 15.1, -90., 90.  # prevent latitudes > 90
    lats = stats.truncnorm((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma).rvs(N_spots)
    sgn = np.sign(0.5 - np.random.random(N_spots))
    latitudes = lats*sgn
    return latitudes


# draw from a truncated lognormal distribution
def trunc_lognormal_draw(mu=0, sigma=1, lower=-np.inf, upper=np.inf, size=1):
    zs = stats.truncnorm.rvs((np.log(lower) - mu) / sigma, (np.log(upper) - mu) / sigma, size=size)
    return np.exp(mu + zs * sigma)


def baumann_draw(types, mu=np.log([46.51, 90.24])+np.log([2.14, 2.49]), sigma=np.sqrt(np.log([2.14, 2.49]))):
    # drawing a random starspot area (MSH) from convoluted Baumann 2005 log-normal pdf (https://arxiv.org/pdf/astro-ph/0510516.pdf) eq. 2
    # defaults from Table 1: total area from single spots and total area rows
    # <A> values are multiplied by a factor 1.54 (as in Borgniet et al. (2015))
    # mu = log(<A>) + log(sigma_a); sigma = sqrt(log(sigma_a))
    # exp(mu) ~ [100, 225] MSH; sigma ~ [0.87, 0.96]   
    # mean is exp(mu + sigma^2/2) = <A> sigma_a^(3/2) ~ [146, 355] MSH

    draw_amount = len(types)
    draws = np.zeros(draw_amount)

    bools = (types == 0)
    type_0_amount = sum(bools)
    
    # fill draws with draws from a lognormal for each spot type while preventing spots less than 10 MSH
    draws[bools] = trunc_lognormal_draw(mu=mu[0], sigma=sigma[0], lower=10, size=type_0_amount)
    draws[np.invert(bools)] = trunc_lognormal_draw(mu=mu[1], sigma=sigma[1], lower=10, size=draw_amount-type_0_amount)
    
    return draws


def pillet_draw(types, mu=np.array([2.619, 3.373]), sigma=np.array([0.806, 0.869])):
    # drawing a random starspot decay rate (MSH/day) from Pillet 1993 log-normal pdf (http://adsabs.harvard.edu/abs/1993A%26A...274..521M) eq. 5
    # default mu are the mu_logD from Table 5 from Standard Set: total decay La Laguna types 3 and 2
    # default sigma are the sigma_logD from Table 5 from Standard Set: total decay La Laguna types 3 and 2
    # exp(mu) ~ [13.7, 29.2] MSH/day
    # mean is exp(mu + sigma^2/2) ~ [19.0, 42.5] MSH/day

	draw_amount = len(types)
	draws = np.zeros(draw_amount)

	bools = (types == 0)
	type_0_amount = sum(bools)

	# fill draws with draws from a lognormal for each spot type while preventing decay rates less than 3 MSH/day or higher than 200 MSH/day
	draws[bools] = trunc_lognormal_draw(mu=mu[0], sigma=sigma[0], lower=3, upper=200, size=type_0_amount)
	draws[np.invert(bools)] = trunc_lognormal_draw(mu=mu[1], sigma=sigma[1], lower=3, upper=200, size=draw_amount-type_0_amount)
	
    return draws


# MSH(in units of Rstar^2) = 1e-6 1/2 4 pi Rstar^2 = 2e-6 pi Rstar^2  # 1/million of half of star's surface area
# pi * R_spot(in units of Rstar)^2 = Area(MSH(Rstar^2)) * 2e-6 pi
# R_spot = sqrt(Area(MSH(Rsun)) * 2e-6)

# from SOAP config file...
# active region's size [Rstar]. If you want a size S1 in area of the visible hemisphere, you have to put here sqrt(2*S1), because
# S1 = Area_spot/Area_visible_hemisphere = pi*(size1*Rstar)**2/(2*pi*Rstar**2) = size1**2/2 -> size1 = sqrt(2*S1).
# For 0.1% of the visible hemisphere (1e3 MSH), put 0.045
def r_star2msh(r_star):
    return r_star * r_star * 5e5


def msh2r_star(msh):
    return 1e-3 * np.sqrt(2 * msh)


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
        frac_iso_spots = 0.4  # from Martinez Pillet et al. (1993)
        types = (np.random.uniform(size=N_spots) > frac_iso_spots).astype(int)

    # micro_solar_hemispheres = np.exp(np.random.normal(np.log(mu), np.log(sigma), N_spots))  # wrong!
    micro_solar_hemispheres = baumann_draw(types, mu=mu, sigma_a=sigma_a)
    
    # MSH (area) to R_star (length) converter. Conversion from
    # 'On Sunspot and Starspot Lifetimes' - https://arxiv.org/pdf/1409.4337.pdf
    # Bradshaw and Hartigan 2014
    # sizes = np.sqrt(micro_solar_hemispheres*17682025)/695500.0  # old and wrong, 6.04e-3 sqrt(MSH) per Rsun

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

def spot_lifetime(sizes, types, Gf, mu=np.array([14.8, 30.9]), sigma=np.array([0.806, 0.869])):
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

    lifetimes = np.multiply(time_mod, np.divide((sizes ** 2)/(2e-6 * np.pi), pillet_draw(types, mu=mu, sigma=sigma)) / 365.25)  # convert sizes to MSH, divide by decay rate, convert to years

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