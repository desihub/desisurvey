from math import pi, cos, acos, sin, sqrt, log10
from time import strptime
from calendar import timegm

from pyslalib.slalib import sla_rdplan as rdplan, sla_zd as zd, sla_gmst as gmst
from pyslalib.slalib import sla_dmoon as dmoon, sla_evp as evp
from pyslalib.slalib import sla_dsep as dsep

#
# The DES sky brightness model
#   From DES Obstac code base. Author: Eric Neilsen, Fermilab
#   Nov 2015: extracted from ObsTac.
#   Removed  usage of : from python_cookbook import memoize, memoize_lim
#       thus slowing it down, no doubt.
#
# Usage:
#   inputs mjd, ra, dec, filter, tel_lon, tel_lat, tel_ele
#       mjd in days, ra,dec, tel_lon, tel_lat in degrees, tel_ele in meters
#       filter is a DES filter name, e.g. "r"
#
#   print "Moon zenith distance: %f" % moonZd(latitude, longitude, mjd)
#   print "Sun zenith distance: %f" % sunZd(latitude, longitude, mjd)
#   print "Elongation of the moon: %f" % elongation(mjd)
#   print "Moon brightness: %f" % moon_brightness(mjd)
#   print "Pointing angle with moon: %f" % ((180.0/pi)*acos(
#       moon_cosrho(mjd, ra, dec, latitude, longitude)))
#
#    sky_model = GeneralMoonSkyModel(mjd, ra, dec, filter, 
#       tel_lon, tel_lat, tel_ele)
#    lst = gmst(mjd) + sky_model.longitude * pi/180.0
#    ha = lst - ra*pi/180.0
#    z = zd(ha, dec*pi/180.0, sky_model.latitude*pi/180.0)*180.0/pi
#    print "Pointing zenith distance: %f" % z
#    print "Airmass: %f" % airmass(z)
#
#    
#    print "Sky brightness at pointing: %f" % sky_model.skymag
#



slalib_body = {'sun': 0,
               'moon': 3}

def moonZd(latitude, longitude, mjd):
    """Calculate the zenith distance of the moon, in degrees

    Reproduce a value calculated by http://ssd.jpl.nasa.gov/horizons.cgi
    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> mjd = 51778.47
    >>> zd = moonZd(latitude, longitude, mjd)
    >>> print "%3.1f" % zd
    48.0
    """
    ra, decl, diam = rdplan(mjd, 3, longitude*pi/180.0, latitude*pi/180.0)
    lst = gmst(mjd) + longitude * pi/180.0
    ha = lst - ra
    return zd(ha, decl, latitude*pi/180.0)* 180.0/pi

def sunZd(latitude, longitude, mjd):
    """Calculate the sun zenith distance, in degrees

    Reproduce a value calculated by http://ssd.jpl.nasa.gov/horizons.cgi
    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> mjd = 51778.595
    >>> print "%3.1f" % sunZd(latitude, longitude, mjd)
    55.0
    """
    ra, decl, diam = rdplan(mjd, 0, longitude*pi/180.0, latitude*pi/180.0)
    lst = gmst(mjd) + longitude * pi/180.0
    ha = lst - ra
    return zd(ha, decl, latitude*pi/180.0)* 180.0/pi

def elongation(mjd):
    """Calculate the elongation of the moon

    Reproduce a value calculated by http://ssd.jpl.nasa.gov/horizons.cgi
    >>> mjd = 51778.47
    >>> elong = elongation(mjd)
    >>> print "%3.1f" % elong
    94.0
    """
    pv = dmoon(mjd)
    moon_distance = (sum([x**2 for x in pv[:3]]))**0.5
    
    dvb, dpb, dvh, dph = evp(mjd,-1)         
    sun_distance = (sum([x**2 for x in dph[:3]]))**0.5

    a  = acos(
        (-pv[0]*dph[0] - pv[1]*dph[1] - pv[2]*dph[2])/
        (moon_distance*sun_distance)) * 180.0/pi
    return a

def moon_brightness(mjd):
    """The brightness of the moon (relative to full)

    The value here matches about what I expect from the value in 
    Astrophysical Quantities corresponding to the elongation calculated by
    http://ssd.jpl.nasa.gov/horizons.cgi
    >>> mjd = 51778.47
    >>> print "%3.2f" % moon_brightness(mjd)
    0.10
    """
    alpha = 180.0-elongation(mjd)
    return 2.512**(-0.026*abs(alpha) - 4E-9*(alpha**4))

def body_brightness(mjd, body):
    if body=='moon':
        return moon_brightness(mjd)
    else:
        # return 2.512**(26.74-12.74)
        return 2.512**(25.0-12.74)

def body_twilight(latitude, longitude, mjd, body):
    if body == 'moon':
        z = moonZd(latitude, longitude, mjd)
    else:
        z = sunZd(latitude, longitude, mjd)
    if z<90:
        return 1.0
    if z>108:
        return 0.0
    logfrac = 137.11-2.52333*z+0.01111*z*z
    return 10**logfrac

def airmass(zd):
    """Calculate the airmass

    Reproduce Bemporad's empirical values (reported in Astrophysical Quantities)
    >>> print "%5.3f" % airmass(0.0)
    1.000
    >>> print "%5.3f" % airmass(45.0)
    1.413
    >>> print "%3.1f" % airmass(80.0)
    5.6
    """
    z = min(zd,90.0) * pi/180.0
    a = 462.46 + 2.8121/(cos(z)**2 + 0.22*cos(z) + 0.01)
    x = sqrt( (a*cos(z))**2 + 2 * a + 1 ) - a * cos(z)
    return x

def cosrho(mjd, ra, decl, latitude, longitude, body):
    body_idx = slalib_body[body]
    body_ra, body_decl, diam = rdplan(mjd, body_idx, 
                                      longitude*pi/180.0, latitude*pi/180.0)
    rho = dsep(ra*pi/180.0, decl*pi/180.0, body_ra, body_decl)
    return cos(rho)

def moon_cosrho(mjd, ra, decl, latitude, longitude):
    """Calculate the cosine of the angular separation between the moon and a point on the sky

    Test with results near and far from the moon position reported by
    http://ssd.jpl.nasa.gov/horizons.cgi
    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> mjd = 51778.47
    >>> print "%4.2f" % moon_cosrho(mjd, 51.15, 15.54, latitude, longitude)
    1.00
    >>> print "%4.2f" % moon_cosrho(mjd, 51.15, 105.54, latitude, longitude)
    0.00
    """
    return cosrho(mjd, ra, decl, latitude, longitude, 'moon')

def sun_cosrho(mjd, ra, decl, latitude, longitude):
    return cosrho(mjd, ra, decl, latitude, longitude, 'sun')

def mjd(datedict):
    """Convert a dictionary wi/ year, month, day, hour minute to MJD

    >>> testd = {'year': 2000, 'month': 8, 'day': 22, 'hour': 11, 'minute': 17}
    >>> print "%7.2f" % mjd(testd)
    51778.47
    """
    tstring = '%(year)04d-%(month)02d-%(day)02dT%(hour)02d:%(minute)02d:00Z' % datedict
    d = strptime(tstring,'%Y-%m-%dT%H:%M:%SZ')
    posixtime = timegm(d)
    mjd = 40587.0+posixtime/86400.0
    return mjd

def magadd(m1, m2):
    """Add the flux corresponding to two magnitudes, and return the corresponding magnitude
    """
    return -2.5*log10( 10**(-0.4*m1) + 10**(-0.4*m2))


def magn(f):
    """Return the AB magnitude corresponding to the given flux in microJanskys
    """
    if f <= 0:
        return 99.9
    return 23.9 - 2.5*log10( f )

def flux(m,m0=23.9):
    return 10**(-0.4*(m-m0))

def airglowshell(mzen, h, ra, decl, mjd, k, latitude, longitude, r0=6375.0):
    """Return the surface brightness from an airglow shell

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>>
    >>> ra = 28.71208
    >>> decl = 0.74225
    >>> mjd = 51808.33
    >>> k = 0.0583989
    >>> 
    >>> mzen = 20.15215
    >>> h = 300.0
    >>> m_inf = 22.30762
    >>> print "%3.1f" % magadd(m_inf, airglowshell(mzen, h, ra, decl, mjd, k, latitude, longitude))
    19.8
    """
    lst = gmst(mjd) + longitude * pi/180.0
    ha = lst - ra*pi/180.0
    z = zd(ha, decl*pi/180.0, latitude*pi/180.0)*180.0/pi
    x = airmass(z)
    mag = mzen + 1.25*log10(1.0-(r0/(h+r0))*(sin(z*pi/180.0))**2) + k*(x-1)
    return mag

def bodyterm2(ra, decl, mjd, k, latitude, longitude, body):
    lst = gmst(mjd) + longitude * pi/180.0
    ha = lst - ra*pi/180.0
    z = zd(ha, decl*pi/180.0, latitude*pi/180.0) * 180.0/pi
    x = airmass(z)
    if body=='moon':
        xm = airmass( moonZd(latitude, longitude, mjd) )
    else:
        xm = airmass( sunZd(latitude, longitude, mjd) )
    term = (10**(-0.4*k*x)-10**(-0.4*k*xm))/(-0.4*k*(x-xm))
    term = term * body_twilight(latitude, longitude, mjd, body)
    return term

def sunterm2(ra, decl, mjd, k, latitude, longitude):
    return bodyterm2(ra, decl, mjd, k, latitude, longitude, 'sun')

def moonterm2(ra, decl, mjd, k, latitude, longitude):
    """The term in the scattered moonlight function common to Mie and Rayleigh scattering

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> 
    >>> ra = 28.71208
    >>> decl = 0.74225
    >>> mjd = 51808.33
    >>> k = 0.0583989
    >>> print "%4.2f" % moonterm2(ra, decl, mjd, k, latitude, longitude)
    2.08
    """
    return bodyterm2(ra, decl, mjd, k, latitude, longitude, 'moon')

def rayleigh_frho(mjd, ra, decl, latitude, longitude, body):
    mu = cosrho(mjd, ra, decl, latitude, longitude, body)
    return 0.75*(1.0+mu**2)

def sun_rayleigh_frho(mjd, ra, decl, latitude, longitude):
    mu = cosrho(mjd, ra, decl, latitude, longitude, 'sun')
    return 0.75*(1.0+mu**2)

def moon_rayleigh_frho(mjd, ra, decl, latitude, longitude):
    """Calculate the Rayleigh scattered moonlight

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> 
    >>> ra = 28.71208
    >>> decl = 0.74225
    >>> mjd = 51808.33
    >>>
    >>> print "%4.3f" % moon_rayleigh_frho(mjd, ra, decl, latitude, longitude)
    0.874
    """
    mu = moon_cosrho(mjd, ra, decl, latitude, longitude)
    return 0.75*(1.0+mu**2)

def rayleigh(m, ra, decl, mjd, k, latitude, longitude, body):
    term1 = flux(m + magn(
            rayleigh_frho(mjd, ra, decl, latitude, longitude, body)))
    term2 = bodyterm2(ra, decl, mjd, k, latitude, longitude, body)
    return magn(term1 * term2 * body_brightness(mjd, body))

def sun_rayleigh(m, ra, decl, mjd, k, latitude, longitude, body):
    return sun_rayleigh(m, ra, decl, mjd, k, latitude, longitude, 'sun')

def moon_rayleigh(m, ra, decl, mjd, k, latitude, longitude):
    """Calculate the Rayleigh scattered moonlight

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> 
    >>> ra = 28.71208
    >>> decl = 0.74225
    >>> mjd = 51808.33
    >>> k = 0.0583989
    >>>
    >>> m = -4.2843
    >>> print "%4.2f" % moon_rayleigh(m, ra, decl, mjd, k, latitude, longitude)    
    21.71
    """
    return rayleigh(m, ra, decl, mjd, k, latitude, longitude, 'moon')

def mie_frho(g, mjd, ra, decl, latitude, longitude, body):
    mu = cosrho(mjd, ra, decl, latitude, longitude, body)
    return 1.5*((1.0-g**2)/(2.0+g**2)) * (1.0 + mu) * (1.0 + g**2 - 2.0*g*mu*mu)**(-1.5)

def moon_mie_frho(g, mjd, ra, decl, latitude, longitude):
    """Calculate the Rayleigh scattered moonlight

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> 
    >>> ra = 28.71208
    >>> decl = 0.74225
    >>> mjd = 51808.33
    >>>
    >>> g = 0.65
    >>> print "%3.2f" % moon_mie_frho(g, mjd, ra, decl, latitude, longitude)
    0.38
    """
    return mie_frho(g, mjd, ra, decl, latitude, longitude, 'moon')

def mie(g, c, ra, decl, mjd, k, latitude, longitude, body):
    term1 = mie_frho(g, mjd, ra, decl, latitude, longitude, body)
    term2 = bodyterm2(ra, decl, mjd, k, latitude, longitude, body)
    return magn(c * term1 * term2 * body_brightness(mjd, body))

def moon_mie(g, c, ra, decl, mjd, k, latitude, longitude):
    """Calculate the Mie scattered moonlight

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> 
    >>> ra = 28.71208
    >>> decl = 0.74225
    >>> mjd = 51808.33
    >>> k = 0.0583989
    >>>
    >>> g = 0.59
    >>> c = 26.449
    >>> print "%4.2f" % moon_mie(g, c, ra, decl, mjd, k, latitude, longitude)    
    23.10
    """
    return mie(g, c, ra, decl, mjd, k, latitude, longitude, 'moon')

def skymag(m_inf, m_zen, h, g, mie_c, rayl_m, ra, decl, mjd, k, latitude, longitude, offset=0.0):
    """Calculate the total surface brightness of the sky

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> 
    >>> ra = 28.71208
    >>> decl = 0.74225
    >>> mjd = 51808.33
    >>> k = 0.0583989
    >>>
    >>> m_zen = 20.15215
    >>> h = 300.0
    >>> m_inf = 22.30762
    >>> rayl_m = -4.2843
    >>> g = 0.59
    >>> mie_c = 26.449
    >>> print "%4.2f" % skymag(m_inf, m_zen, h, g, mie_c, rayl_m, ra, decl, mjd, k, latitude, longitude)
    19.61
    """

    if moonZd(latitude, longitude, mjd) > 107.8:
        mags = [m_inf,
                airglowshell(m_zen, h, ra, decl, mjd, k, latitude, longitude)]
    else:
        mags = [m_inf,
                airglowshell(m_zen, h, ra, decl, mjd, k, latitude, longitude),
                moon_rayleigh(rayl_m, ra, decl, mjd, k, latitude, longitude),
                moon_mie(g, mie_c, ra, decl, mjd, k, latitude, longitude)]

    if sunZd(latitude, longitude, mjd) < 107.8:
        mags += [rayleigh(rayl_m, ra, decl, mjd, k, latitude, longitude,'sun'),
                 mie(g, mie_c, ra, decl, mjd, k, latitude, longitude,'sun')]

    m = reduce(magadd, mags)
    m = m + offset
    return m

class GeneralMoonSkyModel(object):
    def __init__(self, mjd, ra, decl, filter_name, tel_lon, tel_lat, tel_ele) :

        self.mjd = mjd
        self.ra = ra
        self.decl = decl
        self.filter_name = filter_name

        # parameters from the obstac.conf file
        filters = "g      r      i     z    Y"
        k       = "0.19   0.11  0.08  0.10  0.08"
        m_inf   = "30.00  30.00 30.00 30.00 30.00"
        m_zen   = "22.23  21.40 20.18 19.04 18.12"
        h       = "90     90    90    90    90"
        rayl_m  = "-4.78 -4.05  -2.87 30.00 30.00"
        g       = "0.50   0.51  0.57  0.59  0.997523"
        mie_c   = "40.82  49.38 64.08 75.67 9914.435922"

        i = filters.split().index(filter_name)
        self.k = float(k.split()[i])
        self.m_inf = float(m_inf.split()[i])
        self.m_zen = float(m_zen.split()[i])
        self.h = float(h.split()[i])
        self.rayl_m = float(rayl_m.split()[i])
        self.g = float(g.split()[i])
        self.mie_c = float(mie_c.split()[i])
        self.offset = 0.0

        self.longitude = tel_lon
        self.latitude  = tel_lat
        self.elevation = tel_ele

    @property
    def skymag(self):
        """Calculate the total surface brightness of the sky
        """
        try:
            m = skymag(self.m_inf, self.m_zen, self.h, 
                       self.g, self.mie_c, self.rayl_m, 
                       self.ra, self.decl, self.mjd, 
                       self.k, self.latitude, self.longitude, 
                       self.offset)
        except:
            with open('badskymag','w') as f:
                print >>f, self.mjd, self.ra, self.decl, self.filter_name
            m=99
        return m

    @property
    def dark_skymag(self):
        dsm = magadd(self.m_inf,self.m_zen)
        return dsm

    @property
    def dark_skymag_diff(self):
        delta_skymag = self.skymag-self.dark_skymag
        return delta_skymag

    @property
    def down(self):
        """Return true iff the both sun and moon are down
        """
        return moonZd(self.latitude, self.longitude, self.mjd) > 108.0 and \
            sunZd(self.latitude, self.longitude, self.mjd) > 108.0

