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
import desisurvey.tiles


# Date range 2019-2025 for tabulated ephemerides.
# This range is chosen large enough to cover commissioning,
# survey validation and the 5-year main survey, so should
# not normally need to be changed, except for testing.
START_DATE = datetime.date(2019, 1, 1)
STOP_DATE  = datetime.date(2025, 12, 31)

_ephem = None

def get_ephem(use_cache=True, write_cache=True):
    """Return tabulated ephemerides for (START_DATE,STOP_DATE).

    The pyephem module must be installed to calculate ephemerides,
    but is not necessary when a FITS file of precalcuated data is
    available.

    Parameters
    ----------
    use_cache : bool
        Use cached ephemerides from memory or disk if possible
        when True.  Otherwise, always calculate from scratch.
    write_cache : bool
        When True, write a generated table so it is available for
        future invocations. Writing only takes place when a
        cached object is not available or ``use_cache`` is False.

    Returns
    -------
    Ephemerides
        Object with tabulated ephemerides for (START_DATE,STOP_DATE).
    """
    global _ephem

    # Freeze IERS table for consistent results.
    desisurvey.utils.freeze_iers()

    # Use standardized string representation of dates.
    start_iso = START_DATE.isoformat()
    stop_iso = STOP_DATE.isoformat()
    range_iso = '({},{})'.format(start_iso, stop_iso)

    log = desiutil.log.get_logger()
    # First check for a cached object in memory.
    if use_cache and _ephem is not None:
        if _ephem.start_date != START_DATE or _ephem.stop_date != STOP_DATE:
            raise RuntimeError('START_DATE, STOP_DATE have changed.')
        log.debug('Returning cached ephemerides for {}.'.format(range_iso))
        return _ephem
    # Next check for a FITS file on disk.
    config = desisurvey.config.Configuration()
    filename = config.get_path('ephem_{}_{}.fits'.format(start_iso, stop_iso))
    if use_cache and os.path.exists(filename):
        # Save restored object in memory.
        _ephem = Ephemerides(START_DATE, STOP_DATE, restore=filename)
        log.info('Restored ephemerides for {} from {}.'
                 .format(range_iso, filename))
        return _ephem
    # Finally, create new ephemerides and save in the memory cache.
    log.info('Building ephemerides for {}...'.format(range_iso))
    _ephem = Ephemerides(START_DATE, STOP_DATE)
    if write_cache:
        # Save the tabulated ephemerides to disk.
        _ephem._table.write(filename, overwrite=True)
        log.info('Saved ephemerides for {} to {}'.format(range_iso, filename))
    return _ephem


class Ephemerides(object):
    """Tabulate ephemerides.

    :func:`get_ephem` should normally be used rather than calling this
    constructor directly.

    Parameters
    ----------
    start_date : datetime.date
        Calculated ephemerides start on the evening of this date.
    stop_date : datetime.date
        Calculated ephemerides stop on the morning of this date.
    num_obj_steps : int
        Number of steps for tabulating object (ra, dec) during each 24-hour
        period from local noon to local noon. Ignored when restore is set.
    restore : str or None
        Name of a file to restore ephemerides from.  Construct ephemerides
        from scratch when None. A restored file must have start and stop
        dates that match our args.

    Attributes
    ----------
    start : astropy.time.Time
        Local noon before the first night for which ephemerides are calculated.
    stop : astropy.time.Time
        Local noon after the last night for which ephemerides are calculated.
    num_nights : int
        Number of consecutive nights for which ephemerides are calculated.
    """
    def __init__(self, start_date, stop_date, num_obj_steps=25, restore=None):
        self.log = desiutil.log.get_logger()
        config = desisurvey.config.Configuration()

        # Validate date range.
        num_nights = (stop_date - start_date).days
        if num_nights <= 0:
            raise ValueError('Expected start_date < stop_date.')
        self.num_nights = num_nights
        self.start_date = start_date
        self.stop_date = stop_date

        # Convert to astropy times at local noon.
        self.start = desisurvey.utils.local_noon_on_date(start_date)
        self.stop = desisurvey.utils.local_noon_on_date(stop_date)

        # Moon illumination fraction interpolator will be initialized the
        # first time it is used.
        self._moon_illum_frac_interpolator = None

        # Restore ephemerides from a FITS table if requested.
        if restore is not None:
            self._table = astropy.table.Table.read(restore)            
            assert self._table.meta['START'] == str(start_date)
            assert self._table.meta['STOP'] == str(stop_date)
            assert len(self._table) == num_nights
            return

        # Initialize an empty table to fill.
        meta = dict(NAME='Survey Ephemerides', EXTNAME='EPHEM',
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
        self._table['brightdusk_LST'] = astropy.table.Column(
            length=num_nights, format='%.5f',
            description='Apparent LST at brightdawn in degrees')
        self._table['brightdawn_LST'] = astropy.table.Column(
            length=num_nights, format='%.5f',
            description='Apparent LST at brightdusk in degrees')
        self._table['moonrise'] = astropy.table.Column(
            length=num_nights, format=mjd_format,
            description='MJD of moonrise before/during night')
        self._table['moonset'] = astropy.table.Column(
            length=num_nights, format=mjd_format,
            description='MJD of moonset after/during night')
        self._table['moon_illum_frac'] = astropy.table.Column(
            length=num_nights, format='%.3f',
            description='Illuminated fraction of moon surface')
        self._table['nearest_full_moon'] = astropy.table.Column(
            length=num_nights, format='%.5f',
            description='Nearest full moon - local midnight in days')
        self._table['programs'] = astropy.table.Column(
            length=num_nights, shape=(4,), dtype=np.int16,
            description='Program sequence between dusk and dawn')
        self._table['changes'] = astropy.table.Column(
            length=num_nights, shape=(3,),
            description='MJD of program changes between dusk and dawn')

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
            # Use the USNO standard for defining moonrise/set, which means that
            # it will not exactly correspond to DARK <-> ? program transitions
            # at an altitude of 0deg.
            mayall.horizon = '-0:34'
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

        # Build a 1s grid covering the night.
        step_size_sec = 1
        step_size_day = step_size_sec / 86400.
        dmjd_grid = desisurvey.ephem.get_grid(step_size=step_size_sec * u.s)
        # Loop over nights to calculate the program sequence.
        self._table['programs'][:] = -1
        self._table['changes'][:] = 0.
        for row in self._table:
            mjd_grid = dmjd_grid + row['noon'] + 0.5
            pindex = self.tabulate_program(
                mjd_grid, include_twilight=False, as_tuple=False)
            assert pindex[0] == -1 and pindex[-1] == -1
            # Calculate index-1 where new programs starts (-1 because of np.diff)
            changes = np.where(np.diff(pindex) != 0)[0]
            # Must have at least DAY -> NIGHT -> DAY changes.
            assert len(changes) >= 2 and pindex[changes[0]] == -1 and pindex[changes[-1] + 1] == -1
            # Max possible changes is 5.
            assert len(changes) <= 6
            # Check that first change is at dusk.
            assert np.abs(mjd_grid[changes[0]] + 0.5 * step_size_day - row['dusk']) <= step_size_day
            # Check that the last change is at dusk.
            assert np.abs(mjd_grid[changes[-1]] + 0.5 * step_size_day - row['dawn']) <= step_size_day
            row['programs'][0] = pindex[changes[0] + 1]
            for k, idx in enumerate(changes[1:-1]):
                row['programs'][k + 1] = pindex[idx + 1]
                row['changes'][k] = mjd_grid[idx] + 0.5 * step_size_day

        # Tabulate all full moons covering (start, stop) with a 30-day pad.
        full_moons = []
        lo, hi = self._table[0]['noon'] - 30 - mjd0, self._table[-1]['noon'] + 30 - mjd0
        when = lo
        while when < hi:
            when = ephem.next_full_moon(when)
            full_moons.append(when)
        full_moons = np.array(full_moons) + mjd0
        # Find the first full moon after each midnight.
        midnight = self._table['noon'] + 0.5
        idx = np.searchsorted(full_moons, midnight, side='left')
        assert np.all(midnight <= full_moons[idx])
        assert np.all(midnight > full_moons[idx - 1])
        # Calculate time until next full moon and after previous full moon.
        next_full_moon = full_moons[idx] - midnight
        prev_full_moon = midnight - full_moons[idx - 1]
        # Record the nearest full moon to each midnight.
        next_is_nearest = next_full_moon <= prev_full_moon
        self._table['nearest_full_moon'][next_is_nearest] = next_full_moon[next_is_nearest]
        self._table['nearest_full_moon'][~next_is_nearest] = -prev_full_moon[~next_is_nearest]

        # Calculate apparent LST at each brightdusk/dawn in degrees.
        dusk_t = astropy.time.Time(self._table['brightdusk'].data, format='mjd')
        dawn_t = astropy.time.Time(self._table['brightdawn'].data, format='mjd')
        dusk_t.location = desisurvey.utils.get_location()
        dawn_t.location = desisurvey.utils.get_location()
        self._table['brightdusk_LST'] = dusk_t.sidereal_time('apparent').to(u.deg).value
        self._table['brightdawn_LST'] = dawn_t.sidereal_time('apparent').to(u.deg).value
        # Subtract 360 deg if LST wraps around during this night, so that the
        # [dusk, dawn] values can be used for linear interpolation.
        wrap = self._table['brightdusk_LST'] > self._table['brightdawn_LST']
        self._table['brightdusk_LST'][wrap] -= 360
        assert np.all(self._table['brightdawn_LST'] > self._table['brightdusk_LST'])

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

    @property
    def table(self):
        """Read-only access to our internal table."""
        return self._table

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
        row_index = (date - self.start_date).days
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

    def get_night_program(self, night, include_twilight=False, program_as_int=False):
        """Return the program sequence for one night.

        The program definitions are taken from
        :class:`desisurvey.config.Configuration` and depend only on
        sun and moon ephemerides for the night.

        Parameters
        ----------
        night : date
            Converted to a date using :func:`desisurvey.utils.get_date`.
        include_twilight : bool
            Include twilight time at the start and end of each night in
            the BRIGHT program.
        program_as_int : bool
            Return program encoded as a small integer instead of a string
            when True.

        Returns
        -------
        tuple
            Tuple (programs, changes) where programs is a list of N program
            names and changes is a 1D numpy array of N+1 MJD values that
            bracket each program during the night.
        """
        night_ephem = self.get_night(night)
        programs = night_ephem['programs']
        changes = night_ephem['changes']
        # Unused slots are -1.
        num_programs = np.count_nonzero(programs >= 0)
        programs = programs[:num_programs]
        changes = changes[:num_programs - 1]
        if include_twilight:
            start = night_ephem['brightdusk']
            stop = night_ephem['brightdawn']
            BRIGHT = desisurvey.tiles.Tiles.PROGRAM_INDEX['BRIGHT']
            if programs[0] != BRIGHT:
                # Twilight adds a BRIGHT program at the start of the night.
                programs = np.insert(programs, 0, BRIGHT)
                changes = np.insert(changes, 0, night_ephem['dusk'])
            if programs[-1] != BRIGHT:
                # Twilight adds a BRIGHT program at the end of the night.
                programs = np.append(programs, BRIGHT)
                changes = np.append(changes, night_ephem['dawn'])
        else:
            start = night_ephem['dusk']
            stop = night_ephem['dawn']
        # Add start, stop to the change times.
        changes = np.concatenate(([start], changes, [stop]))
        if not program_as_int:
            # Replace program indices with names.
            programs = [desisurvey.tiles.Tiles.PROGRAMS[pidx] for pidx in programs]
        return programs, changes

    def get_program_hours(self, start_date=None, stop_date=None,
                        include_monsoon=False, include_full_moon=False,
                        include_twilight=True):
        """Tabulate hours in each program during each night of the survey.

        Use :func:`desisurvey.plots.plot_program` to visualize program hours.

        This method calculates scheduled hours with no correction for weather.
        Use 1 - :func:`desimodel.weather.dome_closed_fractions` to lookup
        nightly corrections based on historical weather data.

        Parameters
        ----------
        ephem : :class:`desisurvey.ephem.Ephemerides`
            Tabulated ephemerides data to use for determining the program.
        start_date : date or None
            First night to include or use the first date of the survey. Must
            be convertible to a date using :func:`desisurvey.utils.get_date`.
        stop_date : date or None
            First night to include or use the last date of the survey. Must
            be convertible to a date using :func:`desisurvey.utils.get_date`.
        include_monsoon : bool
            Include nights during the annual monsoon shutdowns.
        include_fullmoon : bool
            Include nights during the monthly full-moon breaks.
        include_twilight : bool
            Include twilight time at the start and end of each night in
            the BRIGHT program.

        Returns
        -------
        array
            Numpy array of shape (3, num_nights) containing the number of
            hours in each program (0=DARK, 1=GRAY, 2=BRIGHT) during each
            night.
        """
        # Determine date range to use.
        config = desisurvey.config.Configuration()
        if start_date is None:
            start_date = config.first_day()
        else:
            start_date = desisurvey.utils.get_date(start_date)
        if stop_date is None:
            stop_date = config.last_day()
        else:
            stop_date = desisurvey.utils.get_date(stop_date)
        if start_date >= stop_date:
            raise ValueError('Expected start_date < stop_date.')

        num_nights = (stop_date - start_date).days
        hours = np.zeros((3, num_nights))
        for i in range(num_nights):
            tonight = start_date + datetime.timedelta(days=i)
            if not include_monsoon and desisurvey.utils.is_monsoon(tonight):
                continue
            if not include_full_moon and self.is_full_moon(tonight):
                continue
            programs, changes = self.get_night_program(
                tonight, include_twilight=include_twilight, program_as_int=True)
            for p, dt in zip(programs, np.diff(changes)):
                hours[p, i] += dt
        hours *= 24

        return hours

    def get_available_lst(self, start_date=None, stop_date=None, nbins=192, origin=-60,
                          weather=None, include_monsoon=False, include_full_moon=False,
                          include_twilight=False):
        """Calculate histograms of available LST for each program.

        Parameters
        ----------
        start_date : date or None
            First night to include or use the first date of the survey. Must
            be convertible to a date using :func:`desisurvey.utils.get_date`.
        stop_date : date or None
            First night to include or use the last date of the survey. Must
            be convertible to a date using :func:`desisurvey.utils.get_date`.
        nbins : int
            Number of LST bins to use.
        origin : float
            Rotate DEC values in plots so that the left edge is at this value
            in degrees.
        weather : array or None
            1D array of nightly weather factors (0-1) to use, or None to calculate
            available LST assuming perfect weather.  Length must equal the number
            of nights between start and stop. Values are fraction of the night
            with the dome open (0=never, 1=always). Use
            1 - :func:`desimodel.weather.dome_closed_fractions` to lookup
            suitable corrections based on historical weather data.
        include_monsoon : bool
            Include nights during the annual monsoon shutdowns.
        include_fullmoon : bool
            Include nights during the monthly full-moon breaks.
        include_twilight : bool
            Include twilight in the BRIGHT program when True.

        Returns
        -------
        tuple
            Tuple (lst_hist, lst_bins) with lst_hist having shape (3,nbins) and
            lst_bins having shape (nbins+1,).
        """
        config = desisurvey.config.Configuration()
        if start_date is None:
            start_date = config.first_day()
        else:
            start_date = desisurvey.utils.get_date(start_date)
        if stop_date is None:
            stop_date = config.last_day()
        else:
            stop_date = desisurvey.utils.get_date(stop_date)
        num_nights = (stop_date - start_date).days
        if num_nights <= 0:
            raise ValueError('Expected start_date < stop_date.')
        if weather is not None:
            weather = np.asarray(weather)
            if len(weather) != num_nights:
                raise ValueError('Expected weather array of length {}.'.format(num_nights))
        # Initialize LST histograms for each program.
        lst_bins = np.linspace(origin, origin + 360, nbins + 1)
        lst_hist = np.zeros((len(desisurvey.tiles.Tiles.PROGRAMS), nbins))
        dlst = 360. / nbins
        # Loop over nights.
        for n in range(num_nights):
            night = start_date + datetime.timedelta(n)
            if not include_monsoon and desisurvey.utils.is_monsoon(night):
                continue
            if not include_full_moon and self.is_full_moon(night):
                continue
            # Look up the program changes during this night.
            programs, changes = self.get_night_program(
                night, include_twilight, program_as_int=True)
            # Convert each change MJD to a corresponding LST in degrees.
            night_ephem = self.get_night(night)
            MJD0, MJD1 = night_ephem['brightdusk'], night_ephem['brightdawn']
            LST0, LST1 = [night_ephem['brightdusk_LST'], night_ephem['brightdawn_LST']]
            lst_changes = LST0 + (changes - MJD0) * (LST1 - LST0) / (MJD1 - MJD0)
            assert np.all(np.diff(lst_changes) > 0)
            lst_bin = (lst_changes - origin) / 360 * nbins
            # Loop over programs during the night.
            for i, prog_index in enumerate(programs):
                phist = lst_hist[prog_index]
                lo, hi = lst_bin[i:i + 2]
                # Ensure that 0 <= lo < nbins
                left_edge = np.floor(lo / nbins) * nbins
                lo -= left_edge
                hi -= left_edge
                assert 0 <= lo and lo < nbins
                ilo = int(np.ceil(lo))
                assert ilo > 0
                # Calculate the weight of this night in sidereal hours.
                wgt = 24 / nbins
                if weather is not None:
                    wgt *= weather[n]
                # Divide this program's LST window among the LST bins.
                if hi < nbins:
                    # [lo,hi) falls completely within [0,nbins)
                    ihi = int(np.floor(hi))
                    if ilo == ihi + 1:
                        # LST window is contained within a single LST bin.
                        phist[ihi] += (hi - lo) * wgt
                    else:
                        # Accumulate to bins that fall completely within the window.
                        phist[ilo:ihi] += wgt
                        # Accumulate to partial bins at each end of the program window.
                        phist[ilo - 1] += (ilo - lo) * wgt
                        phist[ihi] += (hi - ihi) * wgt
                else:
                    # [lo,hi) wraps around on the right edge.
                    hi -= nbins
                    assert hi >= 0 and hi < nbins
                    ihi = int(np.floor(hi))
                    # Accumulate to bins that fall completely within the window.
                    phist[ilo:nbins] += wgt
                    phist[0:ihi] += wgt
                    # Accumulate partial bins at each end of the program window.
                    phist[ilo - 1] += (ilo - lo) * wgt
                    phist[ihi] += (hi - ihi) * wgt
        return lst_hist, lst_bins

    def tabulate_program(self, mjd, include_twilight=False, as_tuple=True):
        """Tabulate the program during one night.

        The program definitions are taken from
        :class:`desisurvey.config.Configuration` and depend only on
        sun and moon ephemerides for the night.

        Parameters
        ----------
        mjd : float or array
            MJD values during a single night where the program should be
            tabulated.
        include_twilight : bool
            Include twilight time at the start and end of each night in
            the BRIGHT program.
        as_tuple : bool
            Return a tuple (dark, gray, bright) or else a vector of int16
            values.

        Returns
        -------
        tuple or array
            Tuple (dark, gray, bright) of boolean arrays that tabulates the
            program at each input MJD or an array of small integer indices
            into :attr:`desisurvey.tiles.Tiles.PROGRAMS`, with the special value
            -1 indicating DAYTIME. All output arrays have the same shape as
            the input ``mjd`` array.
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

        # Calculate the moon illuminated fraction at each time.
        moon_frac = self.get_moon_illuminated_fraction(mjd)

        # Select bright and dark night conditions.
        dark_night = (mjd >= night['dusk']) & (mjd <= night['dawn'])
        if include_twilight:
            bright_night = (
                mjd >= night['brightdusk']) & (mjd <= night['brightdawn'])
        else:
            bright_night = dark_night

        # Identify program during each MJD.
        GRAY = desisurvey.config.Configuration().programs.GRAY
        max_prod = GRAY.max_moon_illumination_altitude_product().to(u.deg).value
        max_frac = GRAY.max_moon_illumination()
        gray = dark_night & (moon_alt >= 0) & (
            (moon_frac <= max_frac) &
            (moon_frac * moon_alt <= max_prod))
        dark = dark_night & (moon_alt < 0)
        bright = bright_night & ~(dark | gray)

        assert not np.any(dark & gray | dark & bright | gray & bright)

        if as_tuple:
            return dark, gray, bright
        else:
            # Default value -1=DAYTIME.
            program = np.full(mjd.shape, -1, np.int16)
            program[dark] = desisurvey.tiles.Tiles.PROGRAM_INDEX['DARK']
            program[gray] = desisurvey.tiles.Tiles.PROGRAM_INDEX['GRAY']
            program[bright] = desisurvey.tiles.Tiles.PROGRAM_INDEX['BRIGHT']
            return program

    def is_full_moon(self, night, num_nights=None):
        """Test if a night occurs during a full-moon break.

        The full moon break is defined as the ``num_nights`` nights where
        the moon is most fully illuminated at local midnight.  This method
        should normally be called with ``num_nights`` equal to None, in which
        case the value is taken from our
        :class:`desisurvey.config.Configuration``.

        Parameters
        ----------
        night : date
            Converted to a date using :func:`desisurvey.utils.get_date`.
        num_nights : int or None
            Number of nights to block out around each full-moon.

        Returns
        -------
        bool
            True if the specified night falls during a full-moon break.
        """
        # Check the requested length of the full moon break.
        if num_nights is None:
            num_nights = desisurvey.config.Configuration().full_moon_nights()
        # Look up the index of this night in our table.
        index = self.get_night(night, as_index=True)
        # When is the nearest full moon?
        nearest = self._table['nearest_full_moon'][index]
        if np.abs(nearest) < 0.5 * num_nights:
            return True
        elif nearest == 0.5 * num_nights:
            # Tie breaker if two nights are equally close.
            return True
        else:
            return False

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


def get_grid(step_size=1, night_start=-6, night_stop=7):
    """Calculate a grid of equally spaced times covering one night.

    In case the requested step size does not evenly divide the requested
    range, the last grid point will be rounded up.

    The default range covers all possible observing times at KPNO.

    Parameters
    ----------
    step_size : :class:`astropy.units.Quantity`, optional
        Size of each grid step with time units, default 1 min.
    night_start : :class:`astropy.units.Quantity`, optional
        First grid point relative to local midnight with time units, default -6 h.
    night_stop : :class:`astropy.units.Quantity`, optional
        Last grid point relative to local midnight with time units, default 7 h.

    Returns
    -------
    array
        Numpy array of dimensionless offsets relative to local midnight
        in units of days.
    """
    if not isinstance(step_size, u.Quantity):
        step_size = step_size * u.min
    if not isinstance(night_start, u.Quantity):
        night_start = night_start * u.hour
    if not isinstance(night_stop, u.Quantity):
        night_stop = night_stop * u.hour
    num_points = int(round(((night_stop - night_start) / step_size).to(1).value))
    night_stop = night_start + num_points * step_size
    return (night_start.to(u.day).value +
            step_size.to(u.day).value * np.arange(num_points + 1))
