#
#
#
#  July, 20, 2011
#
#   X. Dumusque
#




from pylab import *
import os,sys,string
import glob
import pyrdb


def read_sun_spectrum(filename_high,filename_low):

    data_high = pyrdb.read_rdb(filename_high[:-4]+'_raw.rdb')
    freq_sun_spot_raw = array(data_high['freq'])
    flux_sun_spot_raw = array(data_high['flux'])

    data_low = pyrdb.read_rdb(filename_low[:-4]+'_raw.rdb')
    freq_sun_raw = array(data_low['freq'])
    flux_sun_raw = array(data_low['flux'])

    data_high = pyrdb.read_rdb(filename_high)
    freq_sun_spot = array(data_high['freq'])
    flux_sun_spot = array(data_high['flux'])

    data_low = pyrdb.read_rdb(filename_low)
    freq_sun = array(data_low['freq'])
    flux_sun = array(data_low['flux'])

    test_freq = (abs(freq_sun - freq_sun_spot) < 0.01)
    freq_sun = freq_sun[test_freq]
    freq_sun_spot = freq_sun_spot[test_freq]
    flux_sun = flux_sun[test_freq]
    flux_sun_spot = flux_sun_spot[test_freq]

    #One point is not physical, remove it. This point goes back in frequency
    index_bad_point = where(freq_sun[1:]-freq_sun[:-1] < 0)[0]
    index_bad_point_spot = where(freq_sun_spot[1:]-freq_sun_spot[:-1] < 0)[0]
    freq_sun = delete(freq_sun,index_bad_point)
    freq_sun_spot = delete(freq_sun_spot,index_bad_point_spot)
    flux_sun = delete(flux_sun,index_bad_point)
    flux_sun_spot = delete(flux_sun_spot,index_bad_point_spot)

    if len(freq_sun) == len(freq_sun_spot):
        return freq_sun_spot_raw,flux_sun_spot_raw,freq_sun_raw,flux_sun_raw,freq_sun,flux_sun_spot,flux_sun
    else:
        print sort(freq_sun - freq_sun_spot)
        print 'ATTENTION ERREUR DANS LA SIMILITUDE ENTRE LES VECTEURS DE FREQ'
