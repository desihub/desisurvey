"""Plan future DESI observations.
"""
from __future__ import print_function, division

import datetime

import numpy as np

import astropy.table
import astropy.io.fits
import astropy.units as u

import desiutil.log

import desimodel.io

import desisurvey.utils


def create(hourangles, priorities):
    """Create a new plan for the start of the survey.

    Parameters
    ----------
    hourangles : array
        1D array of floats with design hour angles in degrees to use for
        each tile.
    priorities : array
        1D array of non-negative floats with initial priorities to use for each
        tile. Priority normalization is arbitrary, but higher values correspond
        to higher priority observing.
    """
    config = desisurvey.config.Configuration()
    tiles = astropy.table.Table(
        desimodel.io.load_tiles(onlydesi=True, extra=False,
                                tilesfile=config.tiles_file()))
    ntiles = len(tiles)

    hourangles = np.asarray(hourangles)
    if len(hourangles.shape) != 1 or len(hourangles) != ntiles:
        raise ValueError('Invalid hourangles parameter.')

    priorities = np.asarray(priorities)
    if len(priorities.shape) != 1 or len(priorities) != ntiles:
        raise ValueError('Invalid priorities parameter.')

    plan = astropy.table.Table()
    plan['tileid'] = tiles['TILEID']
    plan['ra'] = tiles['RA']
    plan['dec'] = tiles['DEC']
    plan['pass'] = tiles['PASS']

    plan['priority'] = priorities
    plan['hourangle'] = hourangles

    # Record day-number (relative to config.first_day) when a tile is first
    # covered by any tiles in passes that must be completed before
    # fiber assignment can be run.
    config = desisurvey.config.Configuration()
    num_nights = desisurvey.utils.day_number(config.last_day())
    plan['covered'] = np.full(ntiles, num_nights, int)

    # Any passes with no fiber-assignment dependency are initially available.
    dependent = config.fiber_assignment_order.keys
    plan['available'] = np.zeros(ntiles, bool)
    for passnum in range(8):
        if 'P' + str(passnum) not in dependent:
            sel = plan['pass'] == passnum
            plan['covered'][sel] = -1
            plan['available'][sel] = True

    return plan


def update_available(plan, progress, night, ephem, fa_delay, fa_delay_type):
    """Update list of available tiles.

    A tile becomes available when all overlapping tiles in the previous pass
    of the same program are complete. A newly available tile is ready for fiber
    assignment. Overlap is defined as center_separation < 2 * tile_radius.

    Parameters
    ----------
    plan : astropy.table.Table
        A table created and updated using functions in this package.
    progress : desisurvey.progress.Progress
        A record of observing progress so far.
    night : datetime.date
        Date when planning is being performed, used to interpret the
        next parameter.
    ephem : desisurvey.ephem.Ephemerides
        Tabulated ephemerides data to use.
    fa_delay : int
        Delay between when a tile is covered and then subsequently made
        available for observing by having fibers assigned. Units are
        specified by ``fa_delay_type``.
    fa_delay_type : 'd' or 'm' or 'q'
        Interpret ``fa_delay`` as a delay in days ('d'), full moons ('m')
        or quarters ('q').  A value of zero will assign fibers at the next
        afternoon / full-moon / 3rd full moon.

    Returns
    -------
    plan
        The input plan with the 'covered' and 'available' columns updated.
    """
    assert (fa_delay >= 0) and (fa_delay_type in ('d', 'm', 'q'))
    log = desiutil.log.get_logger()
    # Look up the nominal tile radius for determining overlaps.
    config = desisurvey.config.Configuration()
    tile_radius = config.tile_radius().to(u.deg).value
    # Look up the current night number.
    day_number = desisurvey.utils.day_number(night)
    # Find complete tiles.
    complete = (progress._table['status'] == 2)
    # Average length of synodic month in days.
    synodic = 29.53
    # Run monthly / quarterly fiber assignment?
    month_number = int(np.floor(day_number / synodic))
    full_moon = ephem.is_full_moon(night, num_nights=1)
    do_monthly = do_quarterly = False
    if full_moon:
        if fa_delay_type == 'm':
            log.info('Will run monthly fiber assignment.')
            do_monthly = True
        elif fa_delay_type == 'q' and ((month_number + 1) % 3 == 0):
            log.info('Will run quarterly fiber assignment.')
            do_quarterly = True
    # Loop over passes.
    ra = plan['ra']
    dec = plan['dec']
    for passnum in range(8):
        under = (plan['pass'] == passnum)
        over = np.zeros_like(under)
        overattr = 'P'+str(passnum)
        if not hasattr(config.fiber_assignment_order, overattr):
            # These tiles should be available from the start of the survey.
            if not np.all(plan['available'][under]):
                raise RuntimeError('Expected all tiles available in pass {0}.'
                                   .format(passnum))
        else:
            overpasses = getattr(config.fiber_assignment_order, overattr)()
            for overpass in overpasses.split('+'):
                if not len(overpass) == 2 and overpass[0] == 'P':
                    raise RuntimeError(
                        'Invalid pass in fiber_assignment_order: {0}.'
                        .format(overpass))
                over |= (plan['pass'] == int(overpass[1]))
            overlapping = desisurvey.utils.separation_matrix(
                ra[under], dec[under], ra[over], dec[over], 2 * tile_radius)
            covered = np.all(~overlapping | complete[over], axis=1)
            new_covered = covered & (plan['covered'][under] > day_number)
            if np.any(new_covered):
                new_tiles = plan['tileid'][under][new_covered]
                log.info(
                    'New tiles covered in pass {0}: {1}.'
                    .format(passnum, ','.join([str(tid) for tid in new_tiles])))
                # Record the night number when these tiles were first covered.
                new = under.copy()
                new[under] = new_covered
                plan['covered'][new] = day_number
            # Check if any tiles are newly available now.
            if fa_delay_type == 'd':
                avail = plan['available'][under] | (
                    plan['covered'][under] + fa_delay <= day_number)
            elif do_monthly or do_quarterly:
                # Calculate delay since each tile was covered in units of
                # lunar cycles ("months") or 3 lunar cycles ("quarters").
                period = 3 * synodic if do_quarterly else synodic
                delay = np.floor(
                    (day_number - plan['covered'][under]) / period)
                avail = plan['available'][under] | (delay >= fa_delay)
            else:
                # No new available tiles.
                avail = plan['available'][under]
            # Are there any new tiles in this pass available now?
            new_avail = avail & ~(plan['available'][under])
            if np.any(new_avail):
                new_tiles = plan['tileid'][under][new_avail]
                log.info(
                    'New tiles available in pass {0}: {1}.'
                    .format(passnum, ','.join([str(tid) for tid in new_tiles])))
                # Record the night number when these tiles were first covered.
                plan['available'][under] |= new_avail
    return plan


def load_design_hourangle(name='surveyinit.fits'):
    """Load design hour-angle assignments from disk.

    Reads column 'HA' from binary table HDU 'DESIGN'. This is the format
    saved by the ``surveyinit`` script, but any FITS file following the
    same convention can be used.

    Parameters
    ----------
    name : str
        Name of the FITS file to read. A relative path is assumed to
        refer to the output path specified in the configuration.
    
    Returns
    -------
    array
        1D array of design hour angles in degrees, with indexing
        that matches :class:`desisurvey.tiles.Tiles`.
    """
    config = desisurvey.config.Configuration()
    fullname = config.get_path(name)
    with astropy.io.fits.open(fullname, memmap=False) as hdus:
        HA = hdus['DESIGN'].data['HA'].copy()
    tiles = desisurvey.tiles.get_tiles()
    if HA.shape != (tiles.ntiles,):
        raise ValueError('Read unexpected HA shape.')
    return HA


def load_weather(start_date=None, stop_date=None, name='surveyinit.fits'):
    """Load dome-open fraction expected during each night of the survey.
    
    Reads Image HDU 'WEATHER'. This is the format saved by the
    ``surveyinit`` script, but any FITS file following the same
    convention can be used.

    Parameters
    ----------
    name : str
        Name of the FITS file to read. A relative path is assumed to
        refer to the output path specified in the configuration.
    start_date : date or None
        First night to include or use the first date of the survey. Must
        be convertible to a date using :func:`desisurvey.utils.get_date`.
    stop_date : date or None
        First night to include or use the last date of the survey. Must
        be convertible to a date using :func:`desisurvey.utils.get_date`.

    Returns
    -------
    array
        1D array of length equal to the span between stop_date and
        start_date. Values are between 0 (dome closed all night) and
        1 (dome open all night).
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
    if stop_date <= start_date:
        raise ValueError('Expected start_date < stop_date.')
    with astropy.io.fits.open(config.get_path(name), memmap=False) as hdus:
        weather = hdus['WEATHER'].data
        num_nights = len(weather)
        first = desisurvey.utils.get_date(hdus['WEATHER'].header['FIRST'])
        last = first + datetime.timedelta(num_nights)
    if start_date < first:
        raise ValueError('Weather not available before {}.'.format(first.isoformat()))
    num_nights = (stop_date - start_date).days
    if last < stop_date:
        raise ValueError('Weather not available after {}.',format(last.isoformat()))
    ilo, ihi = (start_date - first).days, (stop_date - first).days
    return weather[ilo:ihi]


class Planner(object):
    """Coordinate afternoon planning activities.
    """
    def __init__(self, rules=None, fiberassign_cadence='monthly', tiles_file=None):
        self.rules = rules
        if fiberassign_cadence not in ('daily', 'monthly'):
            raise ValueError('Invalid fiberassign_cadence: "{}".'.format(fiberassign_cadence))
        self.fiberassign_cadence = fiberassign_cadence
        config = desisurvey.config.Configuration()
        self.tiles = desisurvey.tiles.get_tiles(tiles_file)
        self.ephem = desisurvey.ephem.get_ephem()
        # Initialize per-tile arrays.
        self.tile_covered = np.full(self.tiles.ntiles, -1)
        self.tile_countdown = np.full(self.tiles.ntiles, 1)
        self.tile_available = np.zeros(self.tiles.ntiles, bool)
        # Precompute the tile overlaps between passes needed to update fiber assignments.
        self.tile_over = {}
        self.overlapping = {}
        fiberassign_order = config.fiber_assignment_order
        tile_diameter = 2 * config.tile_radius().to(u.deg).value
        for passnum in self.tiles.passes:
            under = self.tiles.passnum == passnum
            over = np.zeros_like(under)
            key = 'P{}'.format(passnum)
            if key not in fiberassign_order.keys:
                # Mark tiles in this pass as initially available.
                self.tile_covered[under] = 0
                self.tile_available[under] = True
                print('Pass {} available for initial observing.'.format(passnum))
            else:
                overpasses = getattr(fiberassign_order, key)()
                for overpass in overpasses.split('+'):
                    if not len(overpass) == 2 and overpass[0] == 'P':
                        raise RuntimeError(
                            'Invalid pass in fiber_assignment_order: {}.'
                            .format(overpass))
                    over |= (self.tiles.passnum == int(overpass[1]))
                self.overlapping[passnum] = desisurvey.utils.separation_matrix(
                    self.tiles.tileRA[under], self.tiles.tileDEC[under],
                    self.tiles.tileRA[over], self.tiles.tileDEC[over], tile_diameter)
            self.tile_over[passnum] = over

    def initialize(self, night):
        # Remember the first night of the survey.
        self.initial_night = night
        # Initialize priorities.
        if self.rules is not None:
            none_completed = np.zeros(self.tiles.ntiles, bool)
            self.tile_priority = self.rules.apply(none_completed)
            if not np.any(self.tile_priority > 0):
                raise RuntimeError('Initial tile priorities are all <= 0.')
        else:
            self.tile_priority = np.ones(self.tiles.ntiles, float)
        return self.tile_available, self.tile_priority

    def fiberassign(self, night, completed):
        # Calculate the number of elapsed nights in the survey.
        day_number = (night - self.initial_night).days
        print('Running fiber assignment on {} (day number {}) with {} tiles completed.'
              .format(night, day_number, np.count_nonzero(completed)))
        for passnum in self.tiles.passes:
            under = self.tiles.passnum == passnum
            over = self.tile_over[passnum]
            if not np.any(over):
                continue
            overlapping = self.overlapping[passnum]
            # Identify all tiles in this pass whose covering tiles are completed.
            covered = np.all(~overlapping | completed[over], axis=1)
            # Which tiles have been newly covered since the last call to fiberassign?
            new_covered = covered & (self.tile_covered[under] == -1)
            if np.any(new_covered):
                new_tiles = self.tiles.tileID[under][new_covered]
               # Record the day number when these tiles were first covered.
                new = under.copy()
                new[under] = new_covered
                self.tile_covered[new] = day_number
        # Identify tiles that have been covered but not yet had fiber assignment run.
        ready = ~self.tile_available & (self.tile_covered >= 0)
        # Run fiber assignment on ready tiles that have completed their countdown.
        run_now = ready & (self.tile_countdown == 0)
        self.tile_available[run_now] = True
        # Update delay countdown for the remaining ready tiles.
        delayed = ready & (self.tile_countdown > 0)
        self.tile_countdown[delayed] -= 1
        print('fiber assigned {} tiles, with {} delayed.'
              .format(np.count_nonzero(run_now), np.count_nonzero(delayed)))

    def afternoon_plan(self, night, completed):
        # Update fiber assignments this afternoon?
        if self.fiberassign_cadence == 'monthly':
            # Run fiber assignment on the afternoon before the full moon.
            dt = self.ephem.get_night(night)['nearest_full_moon']
            run_fiberassign = (dt > -0.5) and (dt <= 0.5)
            assert run_fiberassign == self.ephem.is_full_moon(night, num_nights=1)
        else:
            run_fiberassign = True
        if run_fiberassign:
            self.fiberassign(night, completed)
        # Update tile priorities.
        if self.rules is not None:
            self.tile_priority = self.rules.apply(completed)

        return self.tile_available, self.tile_priority
