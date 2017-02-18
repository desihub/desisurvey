import numpy as np
from astropy.time import Time
from desisurvey.kpno import mayall

def earthOrientation(MJD):
    """
    This is an approximate formula because the ser7.dat file's range
    is not long enough for the duration of the survey.
    All formulae are from the Naval Observatory.

    Args:
        MJD: float

    Returns:
        x: float (arcseconds)
        y: float (arcseconds)
        UT1-UTC: float (seconds)
    """

    T = 2000.0 + (MJD - 51544.03) / 365.2422
    UT2_UT1 = 0.022*np.sin(2.0*np.pi*T) - 0.012*np.cos(2.0*np.pi*T) \
            - 0.006*np.sin(4.0*np.pi*T) + 0.007*np.cos(4.0*np.pi*T)
    A = 2.0*np.pi*(MJD-57681.0)/365.25
    C = 2.0*np.pi*(MJD-57681.0)/435.0
    x =  0.1042 + 0.0809*np.cos(A) - 0.0636*np.sin(A) + 0.0229*np.cos(C) - 0.0156*np.sin(C) 
    y =  0.3713 - 0.0593*np.cos(A) - 0.0798*np.sin(A) - 0.0156*np.cos(C) - 0.0229*np.sin(C) 
    UT1_UTC = -0.3259 - 0.00138*(MJD - 57689.0) - (UT2_UT1)
    return x, y, UT1_UTC

def mjd2lst(mjd):
    """
    Converts decimal MJD to LST in decimal degrees

    Args:
        mjd: float

    Returns:
        lst: float (degrees)
    """

    lon = str(mayall.west_lon_deg) + 'd'
    lat = str(mayall.lat_deg) + 'd'
    
    t = Time(mjd, format = 'mjd', location=(lon, lat))
    lst_tmp = t.copy()
    
    #try:
    #    lst_str = str(lst_tmp.sidereal_time('apparent'))
    #except IndexError:
    #    lst_tmp.delta_ut1_utc = -0.1225
    #    lst_str = str(lst_tmp.sidereal_time('apparent'))

    x, y, dut = earthOrientation(mjd)
    lst_tmp.delta_ut1_utc = dut
    lst_str = str(lst_tmp.sidereal_time('apparent'))
    # 23h09m35.9586s
    # 01234567890123
    if lst_str[2] == 'h':
        lst_hr = float(lst_str[0:2])
        lst_mn = float(lst_str[3:5])
        lst_sc = float(lst_str[6:-1])
    else:
        lst_hr = float(lst_str[0:1])
        lst_mn = float(lst_str[2:4])
        lst_sc = float(lst_str[5:-1])
    lst = lst_hr + lst_mn/60.0 + lst_sc/3600.0
    lst *= 15.0 # Convert from hours to degrees
    return lst

def radec2altaz(ra, dec, lst):
    """
    Converts from ecliptic to horizontal coordinate systems.

    Args:
        ra: float, observed right ascension (degrees)
        dec: float, observed declination (degrees)
        lst: float, local sidereal time (degrees)

    Returns:
        alt: float, altitude i.e. elevation (degrees)
        az: float, azimuth (degrees)
    """
    h = np.radians(lst - ra)
    if h < 0.0:
        h += 2.0*np.pi
    d = np.radians(dec)
    phi = np.radians(mayall.lat_deg)
    
    sinAz = np.sin(h) / (np.cos(h)*np.sin(phi) - np.tan(d)*np.cos(phi))
    sinAlt = np.sin(phi)*np.sin(d) + np.cos(phi)*np.cos(d)*np.cos(h)

    if sinAlt > 1.0:
        sinAlt = 1.0
    if sinAlt < -1.0:
        sinAlt = -1.0
    if sinAz > 1.0:
        sinAz = 1.0
    if sinAz < -1.0:
        sinAz = -1.0

    return np.degrees(np.arcsin(sinAlt)), np.degrees(np.arcsin(sinAz))

def angsep(ra1, dec1, ra2, dec2):
    """
    Calculates the angular separation between two objects.

    Args:
        ra1: float (degrees)
        dec1: float (degrees)
        ra2: float (degrees)
        dec2: float (degrees)

    Returns:
        delta: float (degrees)
    """

    deltaRA = np.radians(ra1-ra2)
    DEC1 = np.radians(dec1)
    DEC2 = np.radians(dec2)
    cosDelta = np.sin(DEC1)*np.sin(DEC2) + np.cos(DEC1)*np.cos(DEC2)*np.cos(deltaRA)
    return np.degrees(np.arccos(cosDelta))

def equ2gal_J2000(ra_deg, dec_deg):

    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    x = np.array([0.0, 0.0, 0.0])
    x[0] = np.cos(ra) * np.cos(dec)
    x[1] = np.cos(ra) * np.sin(dec)
    x[2] = np.sin(ra)

    M = np.array([ [-0.054876, -0.873437, -0.483835],
                   [ 0.494109, -0.444830,  0.746982],
                   [-0.867666, -0.198076,  0.455984] ])

    y = M.dot(x)
    b = np.arcsin(y[2])
    l = np.arctan2(y[1], y[0])

    l_deg = np.degrees(l)
    b_deg = np.degrees(b)

    return l_deg, b_deg

