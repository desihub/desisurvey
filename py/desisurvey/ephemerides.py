"""Tabulate sun and moon ephemerides on each night of the survey calendar.
"""
from __future__ import print_function, division

import warnings
import math
import os.path
import datetime

import numpy as np
import scipy.interpolate

import astropy.time
import astropy.table
import astropy.units as u

import ephem

import desiutil.log
import desisurvey.config


class Ephemerides(object):
    """Computes the nightly sun and moon ephemerides for the date range given.

    Args:
        start_date: astropy Time for local noon of first day to compute.
        stop_date: astropy Time for local noon of last day to compute.
        num_moon_steps: number of steps for tabulating moon (alt, az)
            during each 24-hour period from local noon to local noon.
            Ignored when a cached file is loaded.
        use_cache: use a previously cached table if available.
    """
    def __init__(self, start_date, stop_date, num_moon_steps=49, use_cache=True):
        self.log = desiutil.log.get_logger()
        config = desisurvey.config.Configuration()

        # Build filename for saving the ephemerides.
        mjd_start = int(math.floor(start_date.mjd))
        mjd_stop = int(math.floor(stop_date.mjd))
        filename = config.get_path(
            'ephem_{0}_{1}.fits'.format(mjd_start, mjd_stop))
        if use_cache and os.path.exists(filename):
            self._table = astropy.table.Table.read(filename)
            self.start_date = astropy.time.Time(
                self._table.meta['START'], format='isot')
            self.stop_date = astropy.time.Time(
                self._table.meta['STOP'], format='isot')
            self.log.info('Loaded ephemerides from {0} for {1} to {2}'
                          .format(filename, start_date, stop_date))
            return

        # Allocate space for the data we will calculate.
        num_days = (stop_date.datetime - start_date.datetime).days + 1
        data = np.empty(num_days, dtype=[
            ('MJDstart', float),
            ('MJDsunset', float), ('MJDsunrise', float), ('MJDetwi', float),
            ('MJDmtwi', float), ('MJDe13twi', float), ('MJDm13twi', float),
            ('MJDmoonrise', float), ('MJDmoonset', float), ('MoonFrac', float),
            ('MoonNightStart', float), ('MoonNightStop', float),
            ('MJD_bright_start', float), ('MJD_bright_end', float),
            ('MoonAlt', float, num_moon_steps),
            ('MoonAz', float, num_moon_steps)])

        # Initialize the observer.
        mayall = ephem.Observer()
        mayall.lat = config.location.latitude().to(u.rad).value
        mayall.lon = config.location.longitude().to(u.rad).value
        # Disable atmospheric refraction corrections.
        mayall.pressure = 0.0
        # Calculate the MJD corresponding to date=0. in ephem.
        # This throws a warning because of the early year, but it is harmless.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=astropy.utils.exceptions.AstropyUserWarning)
            mjd0 = astropy.time.Time(
                datetime.datetime(1899, 12, 31, 12, 0, 0)).mjd

        # Loop over days.
        for day_offset in range(num_days):
            day = start_date + day_offset * u.day
            mayall.date = day.datetime
            row = data[day_offset]
            # Store local noon for this day.
            row['MJDstart'] = day.mjd
            # Calculate sun rise/set with different horizons.
            mayall.horizon = '-0:34' # the value that the USNO uses.
            row['MJDsunset'] = mayall.next_setting(ephem.Sun()) + mjd0
            row['MJDsunrise'] = mayall.next_rising(ephem.Sun()) + mjd0
            # 13 deg twilight, adequate (?) for BGS sample.
            mayall.horizon = (
                config.programs.BRIGHT.max_sun_altitude().to(u.rad).value)
            row['MJDe13twi'] = mayall.next_setting(
                ephem.Sun(), use_center=True) + mjd0
            row['MJDm13twi'] = mayall.next_rising(
                ephem.Sun(), use_center=True) + mjd0
            # 15 deg twilight, start of dark time if the moon is down.
            mayall.horizon = (
                config.programs.DARK.max_sun_altitude().to(u.rad).value)
            row['MJDetwi'] = mayall.next_setting(
                ephem.Sun(), use_center=True) + mjd0
            row['MJDmtwi'] = mayall.next_rising(
                ephem.Sun(), use_center=True) + mjd0
            # Moon.
            m0 = ephem.Moon()
            mayall.horizon = '-0:34' # the value that the USNO uses.
            row['MJDmoonrise'] = mayall.next_rising(m0) + mjd0
            if row['MJDmoonrise'] > row['MJDsunrise']:
                # Any moon visible tonight is from the previous moon rise.
                row['MJDmoonrise'] = mayall.previous_rising(m0) + mjd0
            mayall.date = row['MJDmoonrise'] - mjd0
            row['MJDmoonset'] = mayall.next_setting(ephem.Moon()) + mjd0
            # Calculate the fraction of the moon's surface that is illuminated
            # at the midpoint between sunset and sunrise.
            m0.compute(0.5 * (row['MJDsunset'] + row['MJDsunrise']) - mjd0)
            row['MoonFrac'] = m0.moon_phase
            # Determine when the moon is up while the sun is down during this
            # night, if at all.
            row['MoonNightStart'] = max(row['MJDmoonrise'], row['MJDsunset'])
            row['MoonNightStop'] = min(max(row['MJDmoonset'], row['MJDsunset']),
                                       row['MJDsunrise'])
            # Tabulate the moon altitude at num_moon_steps equally spaced times
            # covering this interval.
            t_moon = row['MJDstart'] + np.linspace(0., 1., num_moon_steps)
            moon_alt, moon_az = row['MoonAlt'], row['MoonAz']
            for i, t in enumerate(t_moon):
                mayall.date = t - mjd0
                m0.compute(mayall)
                moon_alt[i] = math.degrees(float(m0.alt))
                moon_az[i] = math.degrees(float(m0.az))

            # Calculate the start/stop of any bright-time during this night.
            t_moon, bright = get_bright(row)
            if np.any(bright):
                row['MJD_bright_start'] = np.min(t_moon[bright])
                row['MJD_bright_end'] = np.max(t_moon[bright])
            else:
                row['MJD_bright_start'] = row['MJD_bright_end'] = 0.5 * (
                    row['MJDsunset'] + row['MJDsunrise'])

        t = astropy.table.Table(
            data, meta=dict(NAME='Survey Ephemerides',
                            START=start_date.isot, STOP=stop_date.isot))
        self.log.info('Saving ephemerides to {0}'.format(filename))
        t.write(filename, overwrite=True)

        self.start_date = start_date
        self.stop_date = stop_date
        self._table = t


    def get(self, time):
        """Return the row for the 24-hour period including the specified time.

        Parameters
        ----------
        time : astropy.time.Time
            Time during the day requested.

        Returns
        -------
        astropy.table.Row
            Row of ephemeris data for the requested 24-hour period.
        """
        # The extra 1e-6 is to avoid roundoff error bumping us down a day.
        day_index = int(np.floor(time.mjd - self.start_date.mjd + 1e-6))
        if day_index < 0 or day_index >= len(self._table):
            raise ValueError('Requested time outside ephemerides: {0}'
                             .format(time.datetime))
        return self._table[day_index]


    def get_program(self, mjd):
        """Tabulate the program during one night.

        Parameters
        ----------
        mjd : float or array
            MJD values during a single night where the program should be
            tabulated.

        Returns
        -------
        tuple
            Tuple (dark, gray, bright) of boolean arrays that tabulates the
            program at each input MJD.
        """
        # Get the night of the earliest time.
        mjd = np.asarray(mjd)
        night = self.get(astropy.time.Time(np.min(mjd), format='mjd'))

        # Check that all input MJDs are valid for this night.
        mjd0 = night['MJDstart']
        if np.any((mjd < mjd0) | (mjd >= mjd0 + 1)):
            raise ValueError('MJD values span more than one night.')

        # Calculate the moon altitude angle in degrees at each grid time.
        moon_alt, _ = get_moon_interpolator(night)(mjd)

        # Lookup the moon illuminated fraction for this night.
        moon_frac = night['MoonFrac']

        # Identify times between 13 and 15 degree twilight.
        twilight13 = (mjd >= night['MJDe13twi']) & (mjd <= night['MJDm13twi'])
        twilight15 = (mjd >= night['MJDetwi']) & (mjd <= night['MJDmtwi'])

        # Identify program during each MJD.
        GRAY = desisurvey.config.Configuration().programs.GRAY
        gray = twilight15 & (moon_alt >= 0) & (
            (moon_frac <= GRAY.max_moon_illumination()) &
            (moon_frac * moon_alt <=
             GRAY.max_moon_illumination_altitude_product().to(u.deg).value))
        dark = twilight15 & (moon_alt < 0)
        bright = twilight13 & ~(dark | gray)

        assert not np.any(dark & gray | dark & bright | gray & bright)

        return dark, gray, bright


def get_moon_interpolator(row):
    """Build an interpolator for the moon (alt, az) during one night.

    The values (alt, az) = (-1, 0) are returned for all times when the moon
    is below the horizon or outside of the 24-hour period that this
    interpolator covers.

    Parameters
    ----------
    row : astropy.table.Row
        A single row from the ephemerides astropy Table corresponding to the
        night in question.

    Returns
    -------
    callable
        A callable object that takes a single MJD value or an array of MJD
        values and returns corresponding (alt, az) values in degrees, with
        -90 <= alt <= +90 and 0 <= az < 360.
    """
    # Calculate the grid of time steps where the moon position is
    # already tabulated in the ephemerides table.
    n_moon = len(row['MoonAlt'])
    t_grid = row['MJDstart'] + np.linspace(0., 1., n_moon)

    # Construct a 3D array of (alt, cos(az), sin(az)) values for this night.
    # Use cos(az), sin(az) instead of az directly to avoid wrap-around
    # discontinuities.
    moon_pos = np.empty((3, n_moon))
    moon_pos[0, :] = row['MoonAlt']
    az = np.radians(row['MoonAz'])
    moon_pos[1, :] = np.cos(az)
    moon_pos[2, :] = np.sin(az)

    # Build a cubic interpolator in (alt, az) during this interval.
    # Return (0, 0, 0) outside the interval.
    interpolator = scipy.interpolate.interp1d(
        t_grid, moon_pos, axis=1, kind='cubic', copy=False,
        bounds_error=False, fill_value=0., assume_sorted=True)

    # Wrap the interpolator to convert (cos(az), sin(az)) to az in degrees.
    def wrapper(mjd):
        alt, cos_az, sin_az = interpolator(mjd)
        # Map arctan2 range [-180, +180] into [0, 360] with fmod().
        az = np.fmod(360 + np.degrees(np.arctan2(sin_az, cos_az)), 360)
        return alt, az
    return wrapper


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

    # Calculate the moon altitude at each grid time.
    f_moon = get_moon_interpolator(row)
    alt_out, _ = f_moon(t_out)

    # Identify grid times falling in the BRIGHT program.
    moon_frac = row['MoonFrac']
    bright = (moon_frac >= 0.6) | (alt_out * moon_frac >= 30.)

    return t_out, bright
