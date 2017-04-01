"""Tabulate sun and moon ephemerides on each night of the survey calendar.
"""
from __future__ import print_function, division
from astropy.time import Time
import astropy.table
from datetime import datetime, timedelta
import numpy as np
import scipy.interpolate
import ephem
import desisurvey.kpno as kpno
import warnings
import math
import os.path
import desiutil.log


def getCalAll(startdate, enddate, num_moon_steps=32, use_cache=True):
    """Computes the nightly sun and moon ephemerides for the date range given.

    Args:
        startdate: astropy Time for survey start
        enddate: astropy Time for survey end
        num_moon_steps: number of steps for calculating moon altitude
            during the night, when it is up.
        use_cache: use a previously cached table if available.
    Returns:
        Astropy table of sun and moon ephemerides.
    """
    log = desiutil.log.get_logger()

    # Build filename for saving the ephemerides.
    mjd_start = int(math.floor(startdate.mjd))
    mjd_stop = int(math.ceil(enddate.mjd))
    filename = 'ephem_{0}_{1}.fits'.format(mjd_start, mjd_stop)
    if use_cache and os.path.exists(filename):
        log.info('Loading cached ephemerides from {0}'.format(filename))
        return astropy.table.Table.read(filename)

    # Allocate space for the data we will calculate.
    num_days = (enddate.datetime - startdate.datetime).days + 1
    data = np.empty(num_days, dtype=[
        ('MJDsunset', float), ('MJDsunrise', float), ('MJDetwi', float),
        ('MJDmtwi', float), ('MJDe13twi', float), ('MJDm13twi', float),
        ('MJDmoonrise', float), ('MJDmoonset', float), ('MoonFrac', float),
        ('MoonNightStart', float), ('MoonNightStop', float),
        ('MJD_bright_start', float), ('MJD_bright_end', float),
        ('MoonAlt', float, num_moon_steps),
        ('MoonAz', float, num_moon_steps), ('dirName', 'S8')])

    # Initialize the observer.
    mayall = ephem.Observer()
    mayall.lat = np.radians(kpno.mayall.lat_deg)
    mayall.lon = np.radians(kpno.mayall.west_lon_deg)
    # Disable atmospheric refraction corrections.
    mayall.pressure = 0.0
    # This throws a warning because of the early year, but it is harmless.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=astropy.utils.exceptions.AstropyUserWarning)
        mjd0 = Time(datetime(1900,1,1,12,0,0)).mjd

    # Loop over days.
    for day_offset in range(num_days):
        day = startdate.datetime + timedelta(days=day_offset)
        mayall.date = day
        row = data[day_offset]
        # Calculate sun rise/set with different horizons.
        mayall.horizon = '-0:34' # the value that the USNO uses.
        row['MJDsunset'] = Time(mayall.next_setting(ephem.Sun()).datetime()).mjd
        row['MJDsunrise'] = Time(mayall.next_rising(ephem.Sun()).datetime()).mjd
        # 13 deg twilight, adequate (?) for BGS sample.
        mayall.horizon = '-13'
        row['MJDe13twi'] = Time(mayall.next_setting(ephem.Sun(), use_center=True).datetime()).mjd
        row['MJDm13twi'] = Time(mayall.next_rising(ephem.Sun(), use_center=True).datetime()).mjd
        # 15 deg twilight, start of dark time if the moon is down.
        mayall.horizon = '-15'
        row['MJDetwi'] = Time(mayall.next_setting(ephem.Sun(), use_center=True).datetime()).mjd
        row['MJDmtwi'] = Time(mayall.next_rising(ephem.Sun(), use_center=True).datetime()).mjd
        # Moon.
        mayall.horizon = '-0:34' # the value that the USNO uses.
        row['MJDmoonrise'] = Time(mayall.next_rising(ephem.Moon()).datetime()).mjd
        if row['MJDmoonrise'] > row['MJDsunrise']:
            row['MJDmoonrise'] = Time(mayall.previous_rising(ephem.Moon()).datetime()).mjd
        mayall.date = Time(row['MJDmoonrise'], format='mjd').to_datetime()
        row['MJDmoonset'] = Time(mayall.next_setting(ephem.Moon()).datetime()).mjd
        # Calculate moon phase at the midpoint between sunset and sunrise.
        m0 = ephem.Moon()
        m0.compute(0.5 * (row['MJDsunset'] + row['MJDsunrise']) - mjd0)
        # Calculate the fraction of the moon's surface that is illuminated.
        row['MoonFrac'] = m0.moon_phase
        # Determine when the moon is up while the sun is down during this
        # night, if at all.
        row['MoonNightStart'] = max(row['MJDmoonrise'], row['MJDsunset'])
        row['MoonNightStop'] = min(max(row['MJDmoonset'], row['MJDsunset']),
                                   row['MJDsunrise'])
        # Tabulate the moon altitude at num_moon_steps equally spaced times
        # covering this interval.
        t_moon = np.linspace(row['MoonNightStart'], row['MoonNightStop'],
                             num_moon_steps)
        moon_alt, moon_az = row['MoonAlt'], row['MoonAz']
        for i, t in enumerate(t_moon):
            mayall.date = ephem.Date(t - mjd0)
            m0.compute(mayall)
            moon_alt[i] = math.degrees(float(m0.alt))
            moon_az[i] = math.degrees(float(m0.az))
        # Build the night's directory name.
        row['dirName'] = ('{y:04d}{m:02d}{d:02d}'
                          .format(y=day.year, m=day.month, d=day.day))

        # Calculate the start/stop of any bright-time during this night.
        t_moon, bright = get_bright(row)
        if np.any(bright):
            row['MJD_bright_start'] = np.min(t_moon[bright])
            row['MJD_bright_end'] = np.max(t_moon[bright])
        else:
            row['MJD_bright_start'] = row['MJD_bright_end'] = 0.5 * (
                row['MJDsunset'] + row['MJDsunrise'])

    t = astropy.table.Table(data, meta=dict(name='Survey Ephemerides'))
    log.info('Saving ephemerides to {0}'.format(filename))
    t.write(filename, overwrite=True)
    return t


def get_moon_interpolator(row):
    """Build a linear interpolator for the moon (alt, az) on one night.

    The returned interpolator is only valid for the night specified, so will
    not return valid moon positions for the previous and next nights.

    The values (alt, az) = (-1, 0) are returned for all times when the moon
    is below the horizon.

    Parameters
    ----------
    row : astropy.table.Row
        A single row from the ephemerides astropy Table corresponding to the
        night in question.

    Returns
    -------
    callable
        A callable object that takes a single MJD value or an array of MJD
        values and returns corresponding (alt, az) values in degrees.
    """
    # Calculate the grid of time steps where the moon position is
    # already tabulated in the ephemerides table.
    n_moon = len(row['MoonAlt'])
    t_grid = np.linspace(row['MoonNightStart'], row['MoonNightStop'], n_moon)
    if t_grid[0] == t_grid[-1]:
        return lambda mjd: np.array([-1., 0.])

    # Copy the (alt, az) for this night into 2d array.
    # Is there any performance advantage to (2, n_moon) vs (n_moon, 2)?
    moon_pos = np.empty((2, n_moon))
    moon_pos[0, :] = row['MoonAlt']
    moon_pos[1, :] = row['MoonAz']

    # Return a cubic interpolator in (alt, az) during this interval.
    # Return (-1, 0) outside the interval.
    return scipy.interpolate.interp1d(
        t_grid, moon_pos, axis=1, kind='linear', copy=False,
        bounds_error=False, fill_value=np.array([-1, 0]), assume_sorted=True)


def get_bright(row, interval_mins=1.):
    """Identify bright-time for a single night, if any.

    The bright-time defintion used here is::

        (sun altitude < -13) and
        ((moon fraction > 0.6) or (moon_fraction * moon_altitude > 30 deg))

    Note that this does definition not include times when the sun altitude
    is between -13 and -15 deg that would otherwise be considered gray or dark.

    Parameters
    ----------
    row : astropy.table.Row
        A single row from the ephemerides astropy Table corresponding to the
        night in question.
    interval_mins : float
        Grid spacing for tabulating program changes in minutes.

    Returns
    -------
    tuple
        Tuple (t_moon, bright) where t_moon is an equally spaced MJD grid with
        the requested interval and bright is a boolean array of the same length
        that indicates which grid times are in the BRIGHT program.  All other
        grid times are in the GRAY program.  Returns empty arrays if the
        any bright time would last less than the specified interval.
    """
    # Calculate the grid of time steps where the program should be tabulated.
    interval_days = interval_mins / (24. * 60.)
    t_start = int(math.ceil(max(row['MoonNightStart'], row['MJDe13twi']) /
                            interval_days))
    t_stop = int(math.floor(min(row['MoonNightStop'], row['MJDm13twi']) /
                            interval_days))
    t_out = np.arange(t_start, t_stop + 1) * interval_days

    # Calculate the grid of time steps where the moon altitude is tabulated.
    t_in = np.linspace(row['MoonNightStart'], row['MoonNightStop'],
                       len(row['MoonAlt']))

    # Use linear interpolation of the moon altitude.
    alt_out = np.interp(t_out, t_in, row['MoonAlt'])

    # Set the program at each time step: 0=DARK, 1=GRAY, 2=BRIGHT.
    moon_frac = row['MoonFrac']
    bright = (moon_frac >= 60) or (alt_out * moon_frac >= 30.)

    return t_out, bright
