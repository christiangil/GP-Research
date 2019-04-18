import numpy as np
import scipy.stats as stats
# import ConfigParser
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
    phase = time / PROT
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
    time = phase * PROT
    if time_unit == 'day':
        return time
    elif time_unit == 'year':
        return time / 365.25
    elif time_unit == 'hour':
        return time * 24.
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
def trunc_lognormal_draw(mu=0, sigma=1, lower=np.finfo(float).eps, upper=np.inf, size=1):
    zs = stats.truncnorm.rvs((np.log(lower) - mu) / sigma, (np.log(upper) - mu) / sigma, size=size)
    return np.exp(mu + zs * sigma)


def two_lognormal_draw(types, mus, sigmas, lower=np.finfo(float).eps, upper=np.inf):
    # drawing values from two different log normals
    # types is an array of ones and zeros telling which one to draw from
    assert np.max(types) == 1
    assert np.min(types) == 0
    assert len(mus) == 2
    assert len(sigmas) == 2

    draw_amount = len(types)
    draws = np.zeros(draw_amount)

    bools = (types == 0)
    type_0_amount = sum(bools)

    # fill draws with draws from a lognormal for each spot type while preventing spots less than 10 MSH
    draws[bools] = trunc_lognormal_draw(mu=mus[0], sigma=sigmas[0], lower=lower, upper=upper, size=type_0_amount)
    draws[np.invert(bools)] = trunc_lognormal_draw(mu=mus[1], sigma=sigmas[1], lower=lower, upper=upper, size=draw_amount-type_0_amount)

    return draws

# MSH(in units of Rstar^2) = 1e-6 1/2 4 pi Rstar^2 = 2e-6 pi Rstar^2  # 1/million of half of star's surface area
# pi * R_spot(in units of Rstar)^2 = Area(MSH(Rstar^2)) * 2e-6 pi
# R_spot = sqrt(Area(MSH(Rsun)) * 2e-6)
# from SOAP config file...
# active region's size [Rstar]. If you want a size S1 in area of the visible hemisphere, you have to put here sqrt(2*S1), because
# S1 = Area_spot/Area_visible_hemisphere = pi*(size1*Rstar)**2/(2*pi*Rstar**2) = size1**2/2 -> size1 = sqrt(2*S1).
# For 0.1% of the visible hemisphere (1e3 MSH), put 0.045
def r_star2msh(r_star):
    return np.multiply(r_star, r_star) * 5e5


def msh2r_star(msh):
    return 1e-3 * np.sqrt(2 * msh)


# drawing a random starspot area (MSH) from convoluted Baumann 2005 log-normal pdf (https://arxiv.org/pdf/astro-ph/0510516.pdf) eq. 2
# defaults from Table 1: total area from single spots and total area rows
# <A> values are multiplied by a factor 1.54 (as in Borgniet et al. (2015))
# mu = log(<A> * sigma_a); sigma = sqrt(log(sigma_a))
# exp(mu) ~ [100, 225] MSH; sigma ~ [0.87, 0.96]
# mean is exp(mu + sigma^2/2) = <A> sigma_a^(3/2) ~ [146, 355] MSH
def spot_amplitudes(N_spots, mus=np.log([46.51 * 2.14, 90.24 * 2.49]), sigmas=np.sqrt(np.log([2.14, 2.49])), types=-1):

    # generate a list deciding whether each spot will be an individual spot or a "complex spot group", 0 for individual spot, 1 for group
    if types == -1:
        frac_iso_spots = 0.4  # from Martinez Pillet et al. (1993)
        types = (np.random.uniform(size=N_spots) > frac_iso_spots).astype(int)

    micro_solar_hemispheres = two_lognormal_draw(types, mus, sigmas, lower=10)

    # MSH (area) to R_star (length) converter
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


# getting lifetimes by dividing spot sizes by decay rates
# drawing a random starspot decay rate (MSH/day) from Pillet 1993 log-normal pdf (http://adsabs.harvard.edu/abs/1993A%26A...274..521M) eq. 5
# default mu are the mu_logD from Table 5 from Standard Set: total decay La Laguna types 3 and 2
# default sigma are the sigma_logD from Table 5 from Standard Set: total decay La Laguna types 3 and 2
# exp(mu) ~ [13.7, 29.2] MSH/day
# mean is exp(mu + sigma^2/2) ~ [19.0, 42.5] MSH/day
def spot_lifetime(sizes, types, Gf, decay_mus=np.array([2.619, 3.373]), decay_sigmas=np.array([0.806, 0.869])):
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

    # add growth time
    time_mod = np.divide(Gf, np.ones(len(Gf)) - Gf)  # 1/(1/Gf - 1)

    lifetimes = np.multiply(time_mod, np.divide(r_star2msh(sizes), two_lognormal_draw(types, decay_mus, decay_sigmas, lower=3, upper=200)) / 365.25)  # convert sizes to MSH, divide by decay rate, convert to years

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
            Ratio of growth/decay of spot
        spot_func : string
            Type of spot to build. Currently parabolic and constant (no
            growth/decay) available.

        Returns
        -------
        spot : array
            Size of spot for each phase.
        """
    if spot_func == 'constant':
        return A*np.ones(len(phases))

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
