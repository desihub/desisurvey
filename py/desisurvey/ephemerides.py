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
import astropy.utils.exceptions
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
    num_obj_steps : int
        Number of steps for tabulating object (ra, dec) during each 24-hour
        period from local noon to local noon. Ignored when a cached file
        is loaded.
    use_cache : bool
        When True, use a previously saved table if available.
    write_cache : bool
        When True, write a generated table so it is available for
        future invocations.

    Attributes
    ----------
    start : astropy.time.Time
        Local noon before the first night for which ephemerides are calculated.
    stop : astropy.time.Time
        Local noon after the last night for which ephemerides are calculated.
    num_nights : int
        Number of consecutive nights for which ephemerides are calculated.
    """
    def __init__(self, start_date=None, stop_date=None, num_obj_steps=25,
                 use_cache=True, write_cache=True):
        self.log = desiutil.log.get_logger()
        config = desisurvey.config.Configuration()

        # Freeze IERS table for consistent results.
        desisurvey.utils.freeze_iers()

        # Use our config to set any unspecified dates.
        if start_date is None:
            start_date = config.first_day()
        if stop_date is None:
            stop_date = config.last_day()

        # Validate date range.
        num_nights = (stop_date - start_date).days
        if num_nights <= 0:
            raise ValueError('Expected start_date < stop_date.')
        self.num_nights = num_nights

        # Convert to astropy times at local noon.
        self.start = desisurvey.utils.local_noon_on_date(start_date)
        self.stop = desisurvey.utils.local_noon_on_date(stop_date)

        # Moon illumination fraction interpolator will be initialized the
        # first time it is used.
        self._moon_illum_frac_interpolator = None

        # Build filename for saving the ephemerides.
        filename = config.get_path(
            'ephem_{0}_{1}.fits'.format(start_date, stop_date))

        # Use cached ephemerides if requested and available.
        if use_cache and os.path.exists(filename):
            self._table = astropy.table.Table.read(filename)
            assert self._table.meta['START'] == str(start_date)
            assert self._table.meta['STOP'] == str(stop_date)
            assert len(self._table) == num_nights
            self.log.info('Loaded ephemerides from {0} for {1} to {2}'
                          .format(filename, start_date, stop_date))
            return

        # Initialize an empty table to fill.
        meta = dict(NAME='Survey Ephemerides',
                    START=str(start_date), STOP=str(stop_date))
        self._table = astropy.table.Table(meta=meta)
        mjd_format = '%.5f'
        self._table['noon'] = astropy.table.Column(
            length=num_nights, format=mjd_format,
            description='MJD of local noon before night')
        self._table['dusk'] = astropy.table.Column(
            length=num_nights, format=mjd_format,
            description='MJD of dark/gray sunset')
        self._table['dawn'] = astropy.table.Column(
            length=num_nights, format=mjd_format,
            description='MJD of dark/gray sunrise')
        self._table['brightdusk'] = astropy.table.Column(
            length=num_nights, format=mjd_format,
            description='MJD of bright sunset')
        self._table['brightdawn'] = astropy.table.Column(
            length=num_nights, format=mjd_format,
            description='MJD of bright sunrise')
        self._table['moonrise'] = astropy.table.Column(
            length=num_nights, format=mjd_format,
            description='MJD of moonrise before/during night')
        self._table['moonset'] = astropy.table.Column(
            length=num_nights, format=mjd_format,
            description='MJD of moonset after/during night')
        self._table['moon_illum_frac'] = astropy.table.Column(
            length=num_nights, format='%.3f',
            description='Illuminated fraction of moon surface')
        self._table['brightstart'] = astropy.table.Column(
            length=num_nights, format=mjd_format,
            description='MJD when any bright time starts after twilight')
        self._table['brightstop'] = astropy.table.Column(
            length=num_nights, format=mjd_format,
            description='MJD when any bright time stops after twilight')

        # Add (ra,dec) arrays for each object that we need to avoid and
        # check that ephem has a model for it.
        models = {}
        for name in config.avoid_bodies.keys:
            models[name] = getattr(ephem, name.capitalize())()
            self._table[name + '_ra'] = astropy.table.Column(
                length=num_nights, shape=(num_obj_steps,), format='%.2f',
                description='RA of {0} during night in degrees'.format(name))
            self._table[name + '_dec'] = astropy.table.Column(
                length=num_nights, shape=(num_obj_steps,), format='%.2f',
                description='DEC of {0} during night in degrees'.format(name))

        # The moon is required.
        if 'moon' not in models:
            raise ValueError('Missing required avoid_bodies entry for "moon".')

        # Initialize the observer.
        mayall = ephem.Observer()
        mayall.lat = config.location.latitude().to(u.rad).value
        mayall.lon = config.location.longitude().to(u.rad).value
        mayall.elevation = config.location.elevation().to(u.m).value
        # Configure atmospheric refraction model for rise/set calculations.
        mayall.pressure = 1e3 * config.location.pressure().to(u.bar).value
        mayall.temp = config.location.temperature().to(u.C).value
        # Do not use atmospheric refraction corrections for other calculations.
        mayall_no_ar = mayall.copy()
        mayall_no_ar.pressure = 0.
        # Calculate the MJD corresponding to date=0. in ephem.
        # This throws a warning because of the early year, but it is harmless.
        with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore', astropy.utils.exceptions.AstropyUserWarning)
            mjd0 = astropy.time.Time(
                datetime.datetime(1899, 12, 31, 12, 0, 0)).mjd

        # Initialize a grid covering each 24-hour period for
        # tabulating the (ra,dec) of objects to avoid.
        t_obj = np.linspace(0., 1., num_obj_steps)

        # Calculate ephmerides for each night.
        for day_offset in range(num_nights):
            day = self.start + day_offset * u.day
            mayall.date = day.datetime
            row = self._table[day_offset]
            # Store local noon for this day.
            row['noon'] = day.mjd
            # Calculate bright twilight.
            mayall.horizon = (
                config.programs.BRIGHT.max_sun_altitude().to(u.rad).value)
            row['brightdusk'] = mayall.next_setting(
                ephem.Sun(), use_center=True) + mjd0
            row['brightdawn'] = mayall.next_rising(
                ephem.Sun(), use_center=True) + mjd0
            # Calculate dark / gray twilight.
            mayall.horizon = (
                config.programs.DARK.max_sun_altitude().to(u.rad).value)
            row['dusk'] = mayall.next_setting(
                ephem.Sun(), use_center=True) + mjd0
            row['dawn'] = mayall.next_rising(
                ephem.Sun(), use_center=True) + mjd0
            # Calculate the moonrise/set for any moon visible tonight.
            m0 = ephem.Moon()
            mayall.horizon = '-0:34' # the value that the USNO uses.
            row['moonrise'] = mayall.next_rising(m0) + mjd0
            if row['moonrise'] > row['brightdawn']:
                # Any moon visible tonight is from the previous moon rise.
                row['moonrise'] = mayall.previous_rising(m0) + mjd0
            mayall.date = row['moonrise'] - mjd0
            row['moonset'] = mayall.next_setting(ephem.Moon()) + mjd0
            # Calculate the fraction of the moon's surface that is illuminated
            # at local midnight.
            m0.compute(row['noon'] + 0.5 - mjd0)
            row['moon_illum_frac'] = m0.moon_phase
            # Loop over objects to avoid.
            for i, t in enumerate(t_obj):
                # Set the date of the no-refraction model.
                mayall_no_ar.date = row['noon'] + t - mjd0
                for name, model in models.items():
                    model.compute(mayall_no_ar)
                    row[name + '_ra'][i] = math.degrees(float(model.ra))
                    row[name + '_dec'][i] = math.degrees(float(model.dec))

        # Initialize a grid covering each night with 15sec resolution
        # for tabulating the night program.
        t_grid = get_grid(step_size=15 * u.s)

        # Tabulate the observing program for each night to initialize the
        # brightstart/stop values needed by the afternoonplan module.
        for day_offset in range(num_nights):
            t_night = row['noon'] + 0.5 + t_grid
            dark, gray, bright = self.get_program(t_night)
            # Select BRIGHT times during dark night, i.e., excluding the
            # bright twilight component of bright time.
            bright_night = (
                bright & (t_night > row['dusk']) & (t_night < row['dawn']))
            if np.any(bright_night):
                # Record the first/last BRIGHT time during dark night.
                row['brightstart'] = np.min(t_night[bright_night])
                row['brightstop'] = np.max(t_night[bright_night])
            else:
                # Set brightstart/stop equal to local midnight.
                row['brightstart'] = row['brightstop'] = row['noon'] + 0.5

        if write_cache:
            self.log.info('Saving ephemerides to {0}'.format(filename))
            self._table.write(filename, overwrite=True)

    def get_row(self, row_index):
        """Return the specified row of our table.

        Parameters
        ----------
        row_index : int
            Index starting from zero of the requested row.  Negative values
            are allowed and specify offsets from the end of the table in
            the usual way.

        Returns
        astropy.table.Row or int
            Row of ephemeris data for the requested night.
        """
        if row_index < -self.num_nights or row_index >= self.num_nights:
            raise ValueError('Requested row index outside table: {0}'
                             .format(row_index))
        return self._table[row_index]

    def get_night(self, night, as_index=False):
        """Return the row of ephemerides for a single night.

        Parameters
        ----------
        night : date
            Converted to a date using :func:`desisurvey.utils.get_date`.
        as_index : bool
            Return the row index of the specified night in our per-night table
            if True.  Otherwise return the row itself.

        Returns
        -------
        astropy.table.Row or int
            Row of ephemeris data for the requested night or the index
            of this row (selected via ``as_index``).
        """
        date = desisurvey.utils.get_date(night)
        row_index = (date - self.start.datetime.date()).days
        if row_index < 0 or row_index >= self.num_nights:
            raise ValueError('Requested night outside ephemerides: {0}'
                             .format(night))
        return row_index if as_index else self._table[row_index]

    def get_moon_illuminated_fraction(self, mjd):
        """Return the illuminated fraction of the moon.

        Uses linear interpolation on the tabulated fractions at midnight and
        should be accurate to about 0.01.  For reference, the fraction changes
        by up to 0.004 per hour.

        Parameters
        ----------
        mjd : float or array
            MJD values during a single night where the program should be
            tabulated.

        Returns
        -------
        float or array
            Illuminated fraction at each input time.
        """
        mjd = np.asarray(mjd)
        if (np.min(mjd) < self._table['noon'][0] or
            np.max(mjd) >= self._table['noon'][-1] + 1):
            raise ValueError('Requested MJD is outside ephemerides range.')
        if self._moon_illum_frac_interpolator is None:
            # Lazy initialization of a cubic interpolator.
            midnight = self._table['noon'] + 0.5
            self._moon_illum_frac_interpolator = scipy.interpolate.interp1d(
                midnight, self._table['moon_illum_frac'], copy=True,
                kind='linear', fill_value='extrapolate', assume_sorted=True)
        return self._moon_illum_frac_interpolator(mjd)

    def get_program(self, mjd):
        """Tabulate the program during one night.

        The program definitions are taken from
        :class:`desisurvey.config.Configuration` and depend only on
        sun and moon ephemerides for the night.

        Parameters
        ----------
        mjd : float or array
            MJD values during a single night where the program should be
            tabulated.

        Returns
        -------
        tuple
            Tuple (dark, gray, bright) of boolean arrays that tabulates the
            program at each input MJD.  Each output array has the same
            shape as the input ``mjd`` array.
        """
        # Get the night of the earliest time.
        mjd = np.asarray(mjd)
        night = self.get_night(astropy.time.Time(np.min(mjd), format='mjd'))

        # Check that all input MJDs are valid for this night.
        mjd0 = night['noon']
        if np.any((mjd < mjd0) | (mjd >= mjd0 + 1)):
            raise ValueError('MJD values span more than one night.')

        # Calculate the moon (ra, dec) in degrees at each grid time.
        interpolator = get_object_interpolator(night, 'moon', altaz=True)
        moon_alt, _ = interpolator(mjd)

        # Lookup the moon illuminated fraction at each time.
        moon_frac = self.get_moon_illuminated_fraction(mjd)

        # Select bright and dark night conditions.
        bright_night = (
            mjd >= night['brightdusk']) & (mjd <= night['brightdawn'])
        dark_night = (mjd >= night['dusk']) & (mjd <= night['dawn'])

        # Identify program during each MJD.
        GRAY = desisurvey.config.Configuration().programs.GRAY
        max_prod = GRAY.max_moon_illumination_altitude_product().to(u.deg).value
        gray = dark_night & (moon_alt >= 0) & (
            (moon_frac <= GRAY.max_moon_illumination()) &
            (moon_frac * moon_alt <= max_prod))
        dark = dark_night & (moon_alt < 0)
        bright = bright_night & ~(dark | gray)

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
        night : date
            Converted to a date using :func:`desisurvey.utils.get_date`.
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
        # Look up the index of this night in our table.
        index = self.get_night(night, as_index=True)
        # Ignore any partial breaks at the ends of our date range.
        if index < num_nights or index + num_nights >= len(self._table):
            return False
        # Fetch a single lunar cycle of illumination data centered
        # on this night (unless we are close to one end of the table).
        lo = max(0, index - half_cycle)
        hi = min(self.num_nights, index + half_cycle + 1)
        cycle = self._table['moon_illum_frac'][lo:hi]
        # Sort the illumination fractions in this cycle.
        sort_order = np.argsort(cycle)
        # Return True if tonight's illumination is in the top num_nights.
        return index - lo in sort_order[-num_nights:]


def get_object_interpolator(row, object_name, altaz=False):
    """Build an interpolator for object location during one night.

    Wrap around in RA is handled correctly and we assume that the object never
    wraps around in DEC.  The interpolated unit vectors should be within
    0.3 degrees of the true unit vectors in both (dec,ra) and (alt,az).

    Parameters
    ----------
    row : astropy.table.Row
        A single row from the ephemerides astropy Table corresponding to the
        night in question.
    object_name : string
        Name of the object to build an interpolator for.  Must be listed under
        avoid_objects in :class:`our configuration
        <desisurvey.config.Configuration>`.
    altaz : bool
        Interpolate in (alt,az) if True, else interpolate in (dec,ra).

    Returns
    -------
    callable
        A callable object that takes a single MJD value or an array of MJD
        values and returns the corresponding (dec,ra) or (alt,az) values in
        degrees, with -90 <= dec,alt <= +90 and 0 <= ra,az < 360.
    """
    # Find the tabulated (ra, dec) values for the requested object.
    try:
        ra = row[object_name + '_ra']
        dec = row[object_name + '_dec']
    except AttributeError:
        raise ValueError('Invalid object_name {0}.'.format(object_name))

    # Calculate the grid of MJD time steps where (ra,dec) are tabulated.
    t_obj = row['noon'] + np.linspace(0., 1., len(ra))

    # Interpolate in (theta,phi) = (dec,ra) or (alt,az)?
    if altaz:
        # Convert each (ra,dec) to (alt,az) at the appropriate time.
        times = astropy.time.Time(t_obj, format='mjd')
        frame = desisurvey.utils.get_observer(times)
        sky = astropy.coordinates.ICRS(ra=ra * u.deg, dec=dec * u.deg)
        altaz = sky.transform_to(frame)
        theta = altaz.alt.to(u.deg).value
        phi = altaz.az.to(u.deg).value
    else:
        theta = dec
        phi = ra

    # Construct arrays of (theta, cos(phi), sin(phi)) values for this night.
    # Use cos(phi), sin(phi) instead of phi directly to avoid wrap-around
    # discontinuities.  Leave theta in degrees.
    data = np.empty((3, len(ra)))
    data[0] = theta
    phi = np.radians(phi)
    data[1] = np.cos(phi)
    data[2] = np.sin(phi)

    # Build a cubic interpolator in (alt, az) during this interval.
    # Return (0, 0, 0) outside the interval.
    interpolator = scipy.interpolate.interp1d(
        t_obj, data, axis=1, kind='cubic', copy=True,
        bounds_error=False, fill_value=0., assume_sorted=True)

    # Wrap the interpolator to convert (cos(phi), sin(phi)) back to an angle
    # in degrees.
    def wrapper(mjd):
        theta, cos_phi, sin_phi = interpolator(mjd)
        # Map arctan2 range [-180, +180] into [0, 360] with fmod().
        phi = np.fmod(360 + np.degrees(np.arctan2(sin_phi, cos_phi)), 360)
        return theta, phi
    return wrapper


def get_grid(step_size = 1 * u.min,
             night_start = -6 * u.hour, night_stop = 7 * u.hour):
    """Calculate a grid of equally spaced times covering one night.

    In case the requested step size does not evenly divide the requested
    range, the last grid point will be rounded up.

    The default range covers all possible observing times at KPNO.

    Parameters
    ----------
    step_size : astropy.units.Quantity
        Size of each grid step with time units.
    night_start : astropy.units.Quantity
        First grid point relative to local midnight with time units.
    night_stop : astropy.units.Quantity
        Last grid point relative to local midnight with time units.

    Returns
    -------
    array
        Numpy array of dimensionless offsets relative to local midnight
        in units of days.
    """
    num_points = int(round(((night_stop - night_start) / step_size).to(1).value))
    night_stop = night_start + num_points * step_size
    return (night_start.to(u.day).value +
            step_size.to(u.day).value * np.arange(num_points + 1))
