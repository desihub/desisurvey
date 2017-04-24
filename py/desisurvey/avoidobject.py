from __future__ import print_function, division
import ephem
from datetime import datetime
import numpy as np
import astropy.units as u
import desisurvey.config


MIN_VENUS_SEP = np.radians(2.0)
MIN_MARS_SEP = np.radians(2.0)
MIN_JUPITER_SEP = np.radians(2.0)
MIN_SATURN_SEP = np.radians(2.0)
MIN_NEPTUNE_SEP = np.radians(2.0)
MIN_URANUS_SEP = np.radians(2.0)
MIN_CERES_SEP = np.radians(2.0)

def avoidObject(datetime, ra0, dec0):
    """
    Checks whether all the objects on the list are far enough away from
    the input coordinates.
    The current list has: Venus, Mars, Jupiter, Saturn, Neptune, Uranus;
    the Moon is treated separately.

    Args:
        datetime: datetime object; should have timezone info
        ra0: float (apparent or observed, degrees)
        dec0: float (apparent or observed, degrees)

    Returns:
        bool, True if all objects on the list are far enough away
    """

    ra = np.radians(ra0)
    dec = np.radians(dec0)

    dt = ephem.Date(datetime.datetime)
    gatech = ephem.Observer()
    config = desisurvey.config.Configuration()
    gatech.lat = config.location.latitude().to(u.rad).value
    gatech.lon = config.location.longitude().to(u.rad).value
    gatech.elevation = config.location.elevation().to(u.m).value
    gatech.date = dt
    gatech.epoch = dt

    venus = ephem.Venus()
    venus.compute(gatech)
    if ephem.separation(venus, (ra, dec)) < MIN_VENUS_SEP:
        return False
    mars = ephem.Mars()
    mars.compute(gatech)
    if ephem.separation(mars, (ra, dec)) < MIN_MARS_SEP:
        return False
    #ceres = ephem.Ceres()
    #ceres.compute(gatech)
    #if ephem.separation(ceres, (ra, dec)) < MIN_CERES_SEP:
    #    return False
    jupiter = ephem.Jupiter()
    jupiter.compute(gatech)
    if ephem.separation(jupiter, (ra, dec)) < MIN_JUPITER_SEP:
        return False
    saturn = ephem.Saturn()
    saturn.compute(gatech)
    if ephem.separation(saturn, (ra, dec)) < MIN_SATURN_SEP:
        return False
    neptune = ephem.Neptune()
    neptune.compute(gatech)
    if ephem.separation(neptune, (ra, dec)) < MIN_NEPTUNE_SEP:
        return False
    uranus = ephem.Uranus()
    uranus.compute(gatech)
    if ephem.separation(uranus, (ra, dec)) < MIN_URANUS_SEP:
        return False

    # If still here, return True
    return True

def moonLoc (datetime, ra0, dec0):
    """
    Returns the distance to the Moon if RA and DEC as well as alt, az.

    Args:
        datetime: datetime object; should have timezone info
        ra0: float (apparent or observed, degrees)
        dec0: float (apparent or observed, degrees)

    Returns:
        float, distance from the Moon (degrees)
        float, Moon altitude (degrees)
        float, Moon azimuth (degrees)
    """

    dt = ephem.Date(datetime.datetime)
    gatech = ephem.Observer()
    config = desisurvey.config.Configuration()
    gatech.lat = config.location.latitude().to(u.rad).value
    gatech.lon = config.location.longitude().to(u.rad).value
    gatech.elevation = config.location.elevation().to(u.m).value
    gatech.date = dt
    gatech.epoch = dt

    moon = ephem.Moon()
    moon.compute(gatech)
    ra = np.radians(ra0)
    dec = np.radians(dec0)
    moondist = ephem.separation(moon, (ra, dec))

    return np.degrees(moondist), np.degrees((moon.alt)), np.degrees((moon.az))
