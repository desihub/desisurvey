from __future__ import print_function, division
from astropy.coordinates import solar_system_ephemeris, EarthLocation, get_body
from astropy.time import Time
from numpy import pi as PI
from numpy import arccos, cos, sin, sqrt
from astropy.coordinates import AltAz

MIN_VENUS_SEP = 2.0 * PI / 180.0
MIN_MARS_SEP = 2.0 * PI / 180.0
MIN_JUPITER_SEP = 2.0 * PI / 180.0
MIN_SATURN_SEP = 2.0 * PI / 180.0
MIN_NEPTUNE_SEP = 2.0 * PI / 180.0
MIN_URANUS_SEP = 2.0 * PI / 180.0
MIN_CERES_SEP = 2.0 * PI / 180.0

def angdist(ra1,dec1,ra2,dec2):
    """
    Computes the angular distance between two objects

    Args:
        ra1: float (R.A for one of the objects, degrees)
        ra2: float (R.A. for the other object, degrees)
        dec1: float (DEC for one of the objects, degrees)
        dec2: float (DEC for the other object, degrees)

    Returns:
        float, Angular distance in radians between the objects
    """
    angdist = arccos(sin(dec1*180/PI)*sin(dec2*180/PI)+ \
                     cos(dec1*180/PI)*cos(dec2*180/PI)*cos((ra1-ra2)*180/PI))
    return angdist


def avoidObject(datetime, ra0, dec0):
    """
    Checks whether all the objects on the list are far enough away from
    the input coordinates.
    The current list has: Venus, Mars, Jupiter, Saturn, Neptune, Uranus;
    the Moon is treated separately.

    Args:
        datetime: astropy Time; should have timezone info
        ra0: float (apparent or observed, degrees)
        dec0: float (apparent or observed, degrees)

    Returns:
        bool, True if all objects on the list are far enough away
    """

    #KPNO's location (slightly different than in kpno.py)
    location = EarthLocation.of_site('Kitt Peak National Observatory')
    with solar_system_ephemeris.set('de432s'):
        venus = get_body('venus', datetime, location)
        mars = get_body('mars', datetime, location)
        jupiter = get_body('jupiter', datetime, location)
        saturn = get_body('saturn', datetime, location)
        uranus = get_body('uranus', datetime, location)
        neptune = get_body('neptune', datetime, location)

    dist_venus = angdist(venus.ra.deg,venus.dec.deg,ra0,dec0)
    if dist_venus < MIN_VENUS_SEP:
        return False
    dist_mars = angdist(mars.ra.deg,mars.dec.deg,ra0,dec0)
    if dist_mars < MIN_MARS_SEP:
        return False
    dist_jupiter = angdist(jupiter.ra.deg,jupiter.dec.deg,ra0,dec0)
    if dist_jupiter < MIN_JUPITER_SEP:
        return False
    dist_saturn = angdist(saturn.ra.deg,saturn.dec.deg,ra0,dec0)
    if dist_saturn < MIN_SATURN_SEP:
        return False
    dist_neptune = angdist(neptune.ra.deg,neptune.dec.deg,ra0,dec0)
    if dist_neptune < MIN_NEPTUNE_SEP:
        return False
    dist_uranus = angdist(uranus.ra.deg,uranus.dec.deg,ra0,dec0)
    if dist_uranus < MIN_URANUS_SEP:
        return False

    # If still here, return True
    return True

def moonLoc (datetime, ra0, dec0):
    """
    Returns the distance to the Moon if RA and DEC as well as alt, az.

    Args:
        datetime: astropy Time; should have timezone info
        ra0: float (apparent or observed, degrees)
        dec0: float (apparent or observed, degrees)

    Returns:
        float, distance from the Moon (degrees)
        float, Moon altitude (degrees)
        float, Moon azimuth (degrees)
    """

    location = EarthLocation.of_site('Kitt Peak National Observatory')
    aa = AltAz(location=location, obstime=datetime)
    with solar_system_ephemeris.set('de432s'):
        moon = get_body('moon', datetime, location)
    moondist = angdist(moon.ra.deg,moon.dec.deg,ra0,dec0)
    moon = moon.transform_to(aa)
    return moondist*180.0/PI, moon.alt.deg, moon.az.deg
