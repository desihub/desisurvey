"""Utility functions for survey planning and scheduling.
"""
from __future__ import print_function, division

import datetime
import os
import warnings

import numpy as np

import pytz

import astropy.time
from astropy.coordinates import EarthLocation
import astropy.units as u

import desiutil.log
import desiutil.iers

import desimodel.weather

from .config import Configuration



_telescope_location = None
#
# This global variable appears to be unused.
#
_dome_closed_fractions = None

# Temporary assignment for backward compatibility
freeze_iers = desiutil.iers.freeze_iers


def get_location():
    """Return the telescope's earth location.

    The location object is cached after the first call, so there is no need
    to cache this function's return value externally.

    Returns
    -------
    astropy.coordinates.EarthLocation
    """
    global _telescope_location
    if _telescope_location is None:
        config = Configuration()
        _telescope_location = EarthLocation.from_geodetic(
            lat=config.location.latitude(),
            lon=config.location.longitude(),
            height=config.location.elevation())
    return _telescope_location


def get_observer(when, alt=None, az=None):
    """Return the AltAz frame for the telescope at the specified time(s).

    Refraction corrections are not applied (for now).

    The returned object is automatically broadcast over input arrays.

    Parameters
    ----------
    when : astropy.time.Time
        One or more times when the AltAz transformations should be calculated.
    alt : astropy.units.Quantity or None
        Local altitude angle(s)
    az : astropy.units.Quantity or None
        Local azimuth angle(s)

    Returns
    -------
    astropy.coordinates.AltAz
        AltAz frame object suitable for transforming to/from local horizon
        (alt, az) coordinates.
    """
    if alt is not None and az is not None:
        kwargs = dict(alt=alt, az=az)
    elif alt is not None or az is not None:
        raise ValueError('Must specify both alt and az.')
    else:
        kwargs = {}
    return astropy.coordinates.AltAz(
        location=get_location(), obstime=when, pressure=0, **kwargs)


def cos_zenith_to_airmass(cosZ):
    """Convert a zenith angle to an airmass.

    Uses the Rozenberg 1966 interpolation formula, which gives reasonable
    results for high zenith angles, with a horizon air mass of 40.
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Interpolative_formulas
    Rozenberg, G. V. 1966. "Twilight: A Study in Atmospheric Optics."
    New York: Plenum Press, 160.

    The value of cosZ is clipped to [0,1], so observations below the horizon
    return the horizon value (~40).

    Parameters
    ----------
    cosZ : float or array
        Cosine of angle(s) to convert.

    Returns
    -------
    float or array
        Airmass value(s) >= 1.
    """
    cosZ = np.clip(np.asarray(cosZ), 0., 1.)
    return np.clip(1. / (cosZ + 0.025 * np.exp(-11 * cosZ)), 1., None)


def get_airmass(when, ra, dec):
    """Return the airmass of (ra,dec) at the specified observing time.

    Uses :func:`cos_zenith_to_airmass`.

    Parameters
    ----------
    when : astropy.time.Time
        Observation time, which specifies the local zenith.
    ra : astropy.units.Quantity
        Target RA angle(s)
    dec : astropy.units.Quantity
        Target DEC angle(s)
    Returns
    -------
    array or float
        Value of the airmass for each input (ra,dec).
    """
    target = astropy.coordinates.ICRS(ra=ra, dec=dec)
    zenith = get_observer(when, alt=90 * u.deg, az=0 * u.deg
                          ).transform_to(astropy.coordinates.ICRS)
    # Calculate zenith angle in degrees.
    zenith_angle = target.separation(zenith)
    # Convert to airmass.
    return cos_zenith_to_airmass(np.cos(zenith_angle))


def cos_zenith(ha, dec, latitude=None):
    """Calculate cos(zenith) for specified hour angle, DEC and latitude.

    Combine with :func:`cos_zenith_to_airmass` to calculate airmass.

    Parameters
    ----------
    ha : astropy.units.Quantity
        Hour angle(s) to use, with units convertible to angle.
    dec : astropy.units.Quantity
        Declination angle(s) to use, with units convertible to angle.
    latitude : astropy.units.Quantity or None
        Latitude angle to use, with units convertible to angle.
        Defaults to the latitude of :func:`get_location` if None.

    Returns
    -------
    numpy array
        cosine of zenith angle(s) corresponding to the inputs.
    """
    if latitude is None:
        # Use the observatory latitude by default.
        latitude = Configuration().location.latitude()
    # Calculate sin(altitude) = cos(zenith).
    cosZ = (np.sin(dec) * np.sin(latitude) +
            np.cos(dec) * np.cos(latitude) * np.cos(ha))
    # Return a plain array (instead of a unitless Quantity).
    return cosZ.value


def is_monsoon(night):
    """Test if this night's observing falls in the monsoon shutdown.

    Uses the monsoon date ranges defined in the
    :class:`desisurvey.config.Configuration`.

    Parameters
    ----------
    night : date
        Converted to a date using :func:`desisurvey.utils.get_date`.

    Returns
    -------
    bool
        True if this night's observing falls during the monsoon shutdown.
    """
    date = get_date(night)
    # Fetch our configuration.
    config = Configuration()
    # Test if date falls within any of the shutdowns.
    for key in config.monsoon.keys:
        node = getattr(config.monsoon, key)
        if date >= node.start() and date < node.stop():
            return True
    # If we get here, date does not fall in any of the shutdowns.
    return False


def local_noon_on_date(day):
    """Convert a date to an astropy time at local noon.

    Local noon is used as the separator between observing nights. The purpose
    of this function is to standardize the boundary between observing nights
    and the mapping of dates to times.

    Generates astropy ErfaWarnings for times in the future.

    Parameters
    ----------
    day : datetime.date
        The day to use for generating a time object.

    Returns
    -------
    astropy.time.Time
        A Time object with the input date and a time corresponding to
        local noon at the telescope.
    """
    # Fetch our configuration.
    config = Configuration()

    # Build a datetime object at local noon.
    tz = pytz.timezone(config.location.timezone())
    local_noon = tz.localize(
        datetime.datetime.combine(day, datetime.time(hour=12)))

    # Convert to UTC.
    utc_noon = local_noon.astimezone(pytz.utc)

    # Return a corresponding astropy Time.
    return astropy.time.Time(utc_noon)


def get_current_date():
    """Give current date following get_date convention (date changes at noon).

    Returns
    -------
    datetime.date object for current night, following get_date convention
    """
    date = datetime.datetime.now().astimezone()
    return get_date(date)


def get_date(date):
    """Convert different date specifications into a datetime.date object.

    We use strptime() to convert an input string, so leading zeros are not
    required for strings in the format YYYY-MM-DD, e.g. 2019-8-3 is considered
    valid.

    Instead of testing the input type, we try different conversion methods:
    ``.datetime.date()`` for an astropy time and ``datetime.date()`` for a
    datetime.

    Date specifications that include a time of day (datetime, astropy time, MJD)
    are rounded down to the previous local noon before converting to a date.
    This ensures that all times during a local observing night are mapped to
    the same date, when the night started.  A "naive" (un-localized) datetime
    is assumed to refer to UTC.

    Generates astropy ERFA warnings for future dates.

    Parameters
    ----------
    date : astropy.time.Time, datetime.date, datetime.datetime, string or number
        Specification of the date to return.  A string must have the format
        YYYY-MM-DD (but leading zeros on MM and DD are optional).  A number
        will be interpreted as a UTC MJD value.

    Returns
    -------
    datetime.date
    """
    input_date = date
    # valid types: string, number, Time, datetime, date
    try:
        # Convert bytes to str.
        date = date.decode()
    except AttributeError:
        pass
    try:
        # Convert a string of the form YYYY-MM-DD into a date.
        # This will raise a ValueError for a badly formatted string
        # or invalid date such as 2019-13-01.
        try:
            date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            try:
                date = datetime.datetime.strptime(date, '%Y%m%d').date()
            except ValueError:
                raise
    except TypeError:
        pass
    # valid types: number, Time, datetime, date
    try:
        # Convert a number to an astropy time, assuming it is a UTC MJD value.
        date = astropy.time.Time(date, format='mjd')
    except ValueError:
        pass
    # valid types: Time, datetime, date
    try:
        # Convert an astropy time into a datetime
        date = date.datetime
    except AttributeError:
        pass
    # valid types: datetime, date
    try:
        # Localize a naive datetime assuming it refers to UTC.
        date = pytz.utc.localize(date)
    except (AttributeError, ValueError):
        pass
    # valid types: localized datetime, date
    try:
        # Convert a localized datetime into the date of the previous noon.
        local_tz = pytz.timezone(
            Configuration().location.timezone())
        local_time = date.astimezone(local_tz)
        date = local_time.date()
        if local_time.hour < 12:
            date -= datetime.timedelta(days=1)
    except AttributeError:
        pass
    # valid types: date
    if not isinstance(date, datetime.date):
        raise ValueError('Invalid date specification: {0}.'.format(input_date))
    return date


def night_to_str(date):
    """Return DESI string format (YYYYMMDD) of datetime night.

    Parameters
    ----------
    date : datetime.date object, as from get_date()

    Returns
    -------
    str
        YYYMMDD formatted date string
    """
    return date.isoformat().replace('-', '')


def day_number(date):
    """Return the number of elapsed days since the start of the survey.

    Does not perform any range check that the date is within the nominal
    survey schedule.

    Parameters
    ----------
    date : astropy.time.Time, datetime.date, datetime.datetime, string or number
        Converted to a date using :func:`get_date`.

    Returns
    -------
    int
        Number of elapsed days since the start of the survey.
    """
    config = Configuration()
    return (get_date(date) - config.first_day()).days


def slewtime(ra1, dec1, ra2, dec2, freeslewtime=10,
             ignore_positive_ra=False):
    """Estimate slew times.

    Uses slew model from DESI-3687.  Assumes that 10 s of slew time are
    "free"---i.e., they can be overlapped with other overheads.

    Parameters
    ----------
    ra1 : float
        right ascension (deg)
    dec1 : float
        declination (deg)
    ra2 : float
        right ascension (deg)
    dec2 : float
        declination (deg)
    freeslewtime : float
        amount of time during which one can slew "for free" (s)
    ignore_positive_ra : float
        if True, slew time in the positive RA direction doesn't count.
        Intended to provide no penalty for slews that are just keeping up
        with the sky.

    Returns
    -------
    Estimated slew time in s needed to reach target.
    """
    slewconstants = dict(ra=[0.45, 0.05, 8],
                         dec=[0.45, 0.05, 8])
    if ((ra1.shape != ra2.shape) or (ra1.shape != dec1.shape) or
        (ra1.shape != dec2.shape)):
        raise ValueError('ra1, ra2, dec1, dec2 must have same shape.')
    isscalar = np.ndim(ra1) == 0
    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)
    def slewtimefun(dx, slewtype):
        v, a, t = slewconstants[slewtype]
        dx = ((dx + 180) % 360)-180
        if slewtype == 'ra' and ignore_positive_ra:
            m = dx > 0
            dx[m] = 0
        dx = np.abs(dx)
        m = dx < v**2/a
        tt = np.zeros(len(dx), dtype='f4')
        tt[m] = 2*np.sqrt(dx[m]/a)+t
        tt[~m] = dx[~m]/v+t+v/a
        return np.clip(tt-freeslewtime, 0, np.inf)
    tra = slewtimefun(ra1-ra2, 'ra')
    tdec = slewtimefun(dec1-dec2, 'dec')
    tra = np.atleast_1d(tra)
    tdec = np.atleast_1d(tdec)
    tslew = np.max([tra, tdec], axis=0)
    if isscalar:
        tslew = tslew[0]
    return tslew


def separation_matrix(ra1, dec1, ra2, dec2, max_separation=None):
    """Build a matrix of pair-wise separation between (ra,dec) pointings.

    The ra1 and dec1 arrays must have the same shape. The ra2 and dec2 arrays
    must also have the same shape, but it can be different from the (ra1,dec1)
    shape, resulting in a non-square return matrix.

    Uses the Haversine formula for better accuracy at low separations. See
    https://en.wikipedia.org/wiki/Haversine_formula for details.

    Equivalent to using the separations() method of astropy.coordinates.ICRS,
    but faster since it bypasses any units.

    Parameters
    ----------
    ra1 : array
        1D array of n1 RA coordinates in degrees (without units attached).
    dec1 : array
        1D array of n1 DEC coordinates in degrees (without units attached).
    ra2 : array
        1D array of n2 RA coordinates in degrees (without units attached).
    dec2 : array
        1D array of n2 DEC coordinates in degrees (without units attached).
    max_separation : float or None
        When present, the matrix elements are replaced with booleans given
        by (value <= max_separation), which saves some computation.

    Returns
    -------
    array
        Array with shape (n1,n2) with element [i1,i2] giving the 3D separation
        angle between (ra1[i1],dec1[i1]) and (ra2[i2],dec2[i2]) in degrees
        or, if max_separation is not None, booleans (value <= max_separation).
    """
    ra1, ra2 = np.deg2rad(ra1), np.deg2rad(ra2)
    dec1, dec2 = np.deg2rad(dec1), np.deg2rad(dec2)
    if ra1.shape != dec1.shape:
        raise ValueError('Arrays ra1, dec1 must have the same shape.')
    if len(ra1.shape) != 1:
        raise ValueError('Arrays ra1, dec1 must be 1D.')
    if ra2.shape != dec2.shape:
        raise ValueError('Arrays ra2, dec2 must have the same shape.')
    if len(ra2.shape) != 1:
        raise ValueError('Arrays ra2, dec2 must be 1D.')
    havRA12 = 0.5 * (1 - np.cos(ra2 - ra1[:, np.newaxis]))
    havDEC12 = 0.5 * (1 - np.cos(dec2 - dec1[:, np.newaxis]))
    havPHI = havDEC12 + np.cos(dec1)[:, np.newaxis] * np.cos(dec2) * havRA12
    if max_separation is not None:
        # Replace n1 x n2 arccos calls with a single sin call.
        threshold = np.sin(0.5 * np.deg2rad(max_separation)) ** 2
        return havPHI <= threshold
    else:
        return np.rad2deg(np.arccos(np.clip(1 - 2 * havPHI, -1, +1)))


def match(a, b):
    """Find matching elements of b in unique array a by index.

    Returns indices ma, mb such that a[ma] == b[mb]

    Parameters
    ----------
    a : unique array
    b : array

    Returns
    -------
    ma, mb : indices such that a[ma] == b[mb]
    """
    sa = np.argsort(a)
    _, ua = np.unique(a[sa], return_index=True)
    if len(ua) != len(a):
        raise ValueError('All keys in a must be unique.')
    ind = np.searchsorted(a[sa], b)
    m = (ind >= 0) & (ind < len(a))
    matches = a[sa[ind[m]]] == b[m]
    m[m] &= matches
    return sa[ind[m]], np.flatnonzero(m)
