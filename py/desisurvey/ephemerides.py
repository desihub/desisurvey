"""Tabulate sun and moon ephemerides during the survey.
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
import desisurvey.utils


class Ephemerides(object):
    """Tabulate sun and moon ephemerides during the survey.

    Parameters
    ----------
    start_date : datetime.date or None
        Survey starts on the evening of this date. Use the ``first_day``
        config parameter if None (the default).
    stop_date : datetime.date or None
        Survey stops on the morning of this date. Use the ``last_day``
        config parameter if None (the default).
    num_moon_steps : int
        Number of steps for tabulating moon (alt, az) during each 24-hour
        period from local noon to local noon. Ignored when a cached file
        is loaded.
    use_cache : bool
        When True, use a previously saved table if available.
    write_cache : bool
        When True, write a generated table so it is available for
        future invocations.
    """
    def __init__(self, start_date=None, stop_date=None, num_moon_steps=49,
                 use_cache=True, write_cache=True):
        self.log = desiutil.log.get_logger()
        config = desisurvey.config.Configuration()

        # Use our config to set any unspecified dates.
        if start_date is None:
            start_date = config.first_day()
        if stop_date is None:
            stop_date = config.last_day()

        # Validate date range.
        num_days = (stop_date - start_date).days + 1
        if num_days <= 0:
            raise ValueError('Expected start_date < stop_date.')
        self.num_days = num_days

        # Convert to astropy times at local noon.
        self.start = desisurvey.utils.local_noon_on_date(start_date)
        self.stop = desisurvey.utils.local_noon_on_date(stop_date)

        # Build filename for saving the ephemerides.
        filename = config.get_path(
            'ephem_{0}_{1}.fits'.format(start_date, stop_date))
        if use_cache and os.path.exists(filename):
            self._table = astropy.table.Table.read(filename)
            assert self._table.meta['START'] == str(start_date)
            assert self._table.meta['STOP'] == str(stop_date)
            assert len(self._table) == num_days
            self.log.info('Loaded ephemerides from {0} for {1} to {2}'
                          .format(filename, start_date, stop_date))
            return

        # Allocate space for the data we will calculate.
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
            day = self.start + day_offset * u.day
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
            # at local midnight.
            m0.compute(row['MJDstart'] + 0.5 - mjd0)
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
                            START=str(start_date), STOP=str(stop_date)))
        if write_cache:
            self.log.info('Saving ephemerides to {0}'.format(filename))
            t.write(filename, overwrite=True)
        self._table = t


    def get_night(self, which):
        """Return the row of ephemerides for a single night.

        Parameters
        ----------
        which : int or datetime.date or astropy.time.Time
            Which night to return.  An integer specifies a row index.
            A date specifies the evening of the night to return. A time
            is rounded down to the previous local noon and specifies
            the subsequent night.

        Returns
        -------
        astropy.table.Row
            Row of ephemeris data for the requested 24-hour period.
        """
        if isinstance(which, astropy.time.Time):
            # The extra 1e-6 is to avoid roundoff error bumping us down a day.
            day_index = int(np.floor(which.mjd - self.start.mjd + 1e-6))
        elif isinstance(which, datetime.date):
            day_index = (which - self.start.datetime.date()).days
        else:
            day_index = which
        if day_index < 0 or day_index >= len(self._table):
            raise ValueError('Requested night outside ephemerides: {0}'
                             .format(which))
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
        night = self.get_night(astropy.time.Time(np.min(mjd), format='mjd'))

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


    def is_full_moon(self, night, num_nights=None):
        """Test if a night occurs during a full-moon break.

        The full moon break is defined as the ``num_nights`` nights where
        the moon is most fully illuminated at local midnight.  This method
        should normally be called with ``num_nights`` equal to None, in which
        case the value is taken from our
        :class:`desisurvey.config.Configuration``. Any partial break at the
        begining or end of the period where ephemerides are calculated
        is ignored.

        Parameters
        ----------
        night : datetime.date or astropy.time.Time or int
            Specifies the night in question using :meth:`get_night`.
        num_nights : int or None
            Number of nights reserved for each full-moon break.

        Returns
        -------
        bool
            True if the specified night falls during a full-moon break.
        """
        # Check the requested length of the full moon break.
        if num_nights is None:
            num_nights = desisurvey.config.Configuration().full_moon_nights()
        half_cycle = 12
        if num_nights < 1 or num_nights > 2 * half_cycle:
            raise ValueError('Full moon break must be 1-24 nights.')
        # Look up the offset of this night in our table.
        night = self.get_night(night)
        offset = int(round(night['MJDstart'] - self._table[0]['MJDstart']))
        # Ignore any partial breaks at the ends of our date range.
        if offset < num_nights or offset + num_nights >= len(self._table):
            return False
        # Fetch a single lunar cycle of illumination data centered
        # on this night (unless we are close to one end of the table).
        lo = max(0, offset - half_cycle)
        hi = min(self.num_days, offset + half_cycle + 1)
        cycle = self._table['MoonFrac'][lo:hi]
        # Sort the illumination fractions in this cycle.
        sort_order = np.argsort(cycle)
        # Return True if tonight's illumination is in the top num_nights.
        return offset - lo in sort_order[-num_nights:]


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
