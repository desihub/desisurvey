"""Check distances of a proposed pointing to the moon and planets.
"""
from __future__ import print_function, division

import numpy as np

import astropy.units as u

import ephem

import desisurvey.config


def avoidObject(when, ra0, dec0):
    """
    Check whether solar system objects specified in our config are far enough
    away from the input coordinates. The moon is treated separately.

    Parameters
    ----------
    when : astropy.time.Time
        Time when the check should be performed.
    ra0 : float
        Apparent RA of the pointing to check, in degrees.
    dec0 : float
        Apparent DEC of the pointing to check, in degrees.

    Returns
    -------
    bool
        True if all objects in our config are far enough away.
    """
    ra = np.radians(ra0)
    dec = np.radians(dec0)

    # Initialize the observer for pyephem calculations.
    dt = ephem.Date(when.datetime)
    obs = ephem.Observer()
    config = desisurvey.config.Configuration()
    obs.lat = config.location.latitude().to(u.rad).value
    obs.lon = config.location.longitude().to(u.rad).value
    obs.elevation = config.location.elevation().to(u.m).value
    obs.date = dt
    obs.epoch = dt

    # Loop over bodies named in our configuration.
    config = desisurvey.config.Configuration()
    for name in config.avoid_bodies.keys:
        # Lookup the minimum sparation from this object and convert to radians.
        min_separation = getattr(config.avoid_bodies, name)().to(u.rad).value
        # Initialize and compute the model for this body.
        model = getattr(ephem, name.capitalize())()
        # Calculate the body's separation from the (ra,dec) pointing.
        model.compute(obs)
        if ephem.separation(model, (ra, dec)) < min_separation:
            return False

    # If still here, return True
    return True


def moonLoc(when, ra0, dec0):
    """
    Return the distance to the Moon of RA and DEC as well as alt, az.

    Parameters
    ----------
    when : astropy.time.Time
        Time when the check should be performed.
    ra0 : float
        Apparent RA of the pointing to check, in degrees.
    dec0 : float
        Apparent DEC of the pointing to check, in degrees.

    Returns
    -------
    tuple
        Tuple (moondist, moonalt, moonaz) of the separation angle, altitude
        and azimuth of the moon in degrees.
    """
    dt = ephem.Date(when.datetime)
    obs = ephem.Observer()
    config = desisurvey.config.Configuration()
    obs.lat = config.location.latitude().to(u.rad).value
    obs.lon = config.location.longitude().to(u.rad).value
    obs.elevation = config.location.elevation().to(u.m).value
    obs.date = dt
    obs.epoch = dt

    moon = ephem.Moon()
    moon.compute(obs)
    ra = np.radians(ra0)
    dec = np.radians(dec0)
    moondist = ephem.separation(moon, (ra, dec))

    return np.degrees(moondist), np.degrees((moon.alt)), np.degrees((moon.az))
