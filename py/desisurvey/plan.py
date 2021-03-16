"""Plan future DESI observations.
"""
from __future__ import print_function, division

import datetime
import os.path

import numpy as np

import astropy.table
import astropy.io.fits

import desiutil.log

import desisurvey.utils
import desisurvey.tiles
import desisurvey.ephem


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
    condition : str
        DARK/GRAY/BRIGHT.  For SV tiles that may be observed in multple
        conditions, load the appropriate design HA for this conditions.

    Returns
    -------
    array
        1D array of design hour angles in degrees, with indexing
        that matches :class:`desisurvey.tiles.Tiles`.
    """
    config = desisurvey.config.Configuration()
    fullname = config.get_path(name)
    tiles = desisurvey.tiles.get_tiles()
    design = astropy.io.fits.getdata(fullname, 'DESIGN')
    ind, mask = tiles.index(design['tileID'], return_mask=True)
    HA = np.zeros(tiles.ntiles, dtype='f4')
    HA[ind[mask]] = design['HA'][mask]
    if not np.all(mask) or len(design) != tiles.ntiles:
        log = desiutil.log.get_logger()
        log.warning('The tile file and HA optimizations do not match.')
    return HA


def get_fiber_assign_dir(fiber_assign_dir):
    if fiber_assign_dir is None:
        fiber_assign_dir = os.environ.get('FIBER_ASSIGN_DIR', None)
    if fiber_assign_dir is None:
        config = desisurvey.config.Configuration()
        fiber_assign_dir = getattr(config, 'fiber_assign_dir', None)
        if fiber_assign_dir is not None:
            fiber_assign_dir = fiber_assign_dir()
    return fiber_assign_dir


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
        raise ValueError('Weather not available before {}.'.format(
            first.isoformat()))
    num_nights = (stop_date - start_date).days
    if last < stop_date:
        raise ValueError('Weather not available after {}.'.format(
            last.isoformat()))
    ilo, ihi = (start_date - first).days, (stop_date - first).days
    return weather[ilo:ihi]


class Planner(object):
    """Coordinate afternoon planning activities.

    Parameters
    ----------
    rules : object or None
        Object with an ``apply`` method that is used to implement survey
        strategy by updating tile priorities each afternoon.  When None, all
        tiles have equal priority.
    restore : str or None
        Restore internal state from the snapshot saved to this filename,
        or initialize a new planner when None. Use :meth:`save` to
        save a snapshot to be restored later. Filename is relative to
        the configured output path unless an absolute path is
        provided. Raise a RuntimeError if the saved tile IDs do not
        match the current tiles_file values.
    """
    def __init__(self, rules=None, restore=None, simulate=False):
        self.log = desiutil.log.get_logger()
        self.rules = rules
        config = desisurvey.config.Configuration()
        self.min_snr2_fraction = config.min_snr2_fraction()
        self.simulate = simulate
        if self.simulate:
            self.fiberassign_cadence = config.fiber_assignment_cadence()
            if ((self.fiberassign_cadence not in ('daily', 'monthly')) and
                    (not isinstance(self.fiberassign_cadence, int))):
                raise ValueError('Invalid fiberassign_cadence: "{}".'.format(
                    self.fiberassign_cadence))

        nogray = getattr(config, 'tiles_nogray', False)
        if not isinstance(nogray, bool):
            nogray = nogray()
        self.nogray = nogray

        self.tiles = desisurvey.tiles.get_tiles()
        self.ephem = desisurvey.ephem.get_ephem()
        if restore is not None:
            # Restore the plan for a survey in progress.
            fullname = config.get_path(restore)
            if not os.path.exists(fullname):
                raise RuntimeError('Cannot restore planner from non-existent '
                                   '"{}".'.format(fullname))
            t = astropy.table.Table.read(fullname, hdu='STATUS')
            ind, mask = self.tiles.index(t['TILEID'], return_mask=True)
            if np.any(~mask):
                self.log.warning('Ignoring {} tiles not in tile file.'.format(
                    sum(mask)))
            ind = ind[mask]
            # in operations, we don't really need CADENCE
            if self.simulate:
                self.first_night = desisurvey.utils.get_date(t.meta['FIRST'])
                self.last_night = desisurvey.utils.get_date(t.meta['LAST'])
                self.tile_countdown = np.zeros(self.tiles.ntiles, dtype='i4')
                self.tile_countdown[ind] = t['COUNTDOWN'].data[mask].copy()
                if t.meta['CADENCE'] != self.fiberassign_cadence:
                    raise ValueError('Fiberassign cadence mismatch.')
            self.tile_status = np.zeros(self.tiles.ntiles, dtype='U20')
            self.tile_status[:] = 'unobs'
            self.tile_available = np.zeros(self.tiles.ntiles, dtype='bool')
            self.tile_available[ind] = t['AVAILABLE'].data[mask].copy()
            self.tile_status[ind] = t['STATUS'].data[mask].copy()

            self.tile_priority = np.zeros(self.tiles.ntiles, 'f4')
            self.donefrac = np.zeros(self.tiles.ntiles, 'f4')
            self.tile_priority[ind] = t['PRIORITY'].data[mask].copy()
            self.donefrac[ind] = t['DONEFRAC'].data[mask].copy()

            self.designha = np.zeros(self.tiles.ntiles, 'f4')
            self.designha[ind] = t['DESIGNHA'].data[mask].copy()
            self.log.debug(('Restored plan with {} unobserved, {} pending, '
                            'and {} completed tiles from {}.').format(
                                np.sum(self.donefrac <= 0),
                                np.sum((self.donefrac > 0) &
                                       (self.tile_status != 'done')),
                                np.sum(self.tile_status == 'done'),
                                fullname))
        else:
            # Initialize the plan for a a new survey.
            self.tile_available = np.zeros(self.tiles.ntiles, dtype='bool')
            self.donefrac = np.zeros(self.tiles.ntiles, 'f4')
            self.designha = load_design_hourangle()

            self.first_night = self.last_night = None
            # Initialize per-tile arrays.
            self.tile_status = np.zeros(self.tiles.ntiles, dtype='U80')
            self.tile_status[:] = 'unobs'
            if self.simulate:
                # Initailize the delay countdown for each tile.
                self.tile_countdown = self.tiles.fiberassign_delay.copy()
            # Initialize priorities.
            if self.rules is not None:
                none_started = np.zeros(self.tiles.ntiles, 'f4')
                self.tile_priority = self.rules.apply(none_started)
            else:
                self.tile_priority = np.ones(self.tiles.ntiles, float)
        if not np.any(self.tile_priority > 0):
            raise RuntimeError('All tile priorities are all <= 0.')

    def add_pending_tile(self, tileid):
        """Add a newly observed, now-pending tile to the pending tile list.

        Updates tile availability so that this tile's neighbors will not
        be observed until this tile is completed.

        Parameters
        ----------
        tileid : int
        """
        idx = self.tiles.index(tileid)
        overlapping = self.tiles.overlapping[idx]
        # if this tile's LyA decisions have already been made,
        # that's a problem!
        assert not self.tile_status[idx] == 'done'
        self.tile_available[overlapping] = 0

    def set_donefrac(self, tileid, donefrac):
        """Update planner with new tile donefrac.

        Parameters
        ----------
        tileid : array
            1D array of integer tileIDs to update

        donefrac : array
            1D array of completion fractions for tiles, matching tileid
        """
        tileind, mask = self.tiles.index(tileid, return_mask=True)
        if np.any(~mask):
            self.log.warning('Some tiles with unknown IDs; ignoring')
            tileind = tileind[mask]
            donefrac = donefrac[mask]
        self.donefrac[tileind] = donefrac
        for tileid0, donefrac0 in zip(np.array(tileid)[mask], donefrac):
            if donefrac0 > 0:
                self.add_pending_tile(tileid0)

    def obsend(self):
        return ((self.tile_status == 'obsend') |
                (self.tile_status == 'done') |
                (self.donefrac > self.min_snr2_fraction))

    def obsend_by_program(self):
        tiledone = self.obsend()
        tiledone_by_program = np.zeros(len(self.tiles.programs), 'i4')
        for program in self.tiles.programs:
            progidx = self.tiles.program_index[program]
            m = self.tiles.program_mask[program]
            tiledone_by_program[progidx] = np.sum(tiledone[m])
        return tiledone_by_program

    def survey_completed(self):
        """Test if all tiles have been completed.
        """
        return np.sum(self.obsend()) == self.tiles.ntiles

    def save(self, name):
        """Save a snapshot of our current state that can be restored.

        The output file has a binary table (extname PLAN) with columns
        TILEID, STARTED, OBSERVED, COMPLETED, COUNTDOWN, and PRIORITY and
        header keywords CADENCE, FIRST, LAST. The saved file size is about
        400Kb.

        Parameters
        ----------
        name : str
            Name of FITS file where the snapshot will be saved. The file will
            be saved under our configuration's output path unless name is
            already an absolute path.  Pass the same name to the constructor's
            ``restore`` argument to restore this snapshot.
        """
        config = desisurvey.config.Configuration()
        fullname = config.get_path(name)
        meta = dict(EXTNAME='STATUS')
        if self.simulate:
            meta['CADENCE'] = self.fiberassign_cadence
            meta['FIRST'] = self.first_night.isoformat()
            meta['LAST'] = self.last_night.isoformat()
        t = astropy.table.Table(meta=meta)
        t['TILEID'] = self.tiles.tileID
        t['RA'] = self.tiles.tileRA
        t['DEC'] = self.tiles.tileDEC
        t['DONEFRAC'] = self.donefrac
        t['AVAILABLE'] = self.tile_available
        t['STATUS'] = self.tile_status
        t['PRIORITY'] = self.tile_priority
        t['DESIGNHA'] = self.designha
        if self.simulate:
            t['COUNTDOWN'] = self.tile_countdown
        t.write(fullname+'.tmp', overwrite=True, format='fits')
        os.rename(fullname+'.tmp', fullname)
        self.log.debug(
            ('Saved plan with {} unobserved, {} pending, and {} '
             'completed tiles from {}.').format(
                 np.sum(self.tile_status == 'unobs'),
                 np.sum(self.tile_status == 'obsstart'),
                 np.sum(self.tile_status == 'done'),
                 fullname))

    def fiberassign_simulate(self, night):
        """Update fiber assignments.
        """
        # Calculate the number of elapsed nights in the survey.
        pending = self.tile_status == 'obsend'
        run_now = pending & (self.tile_countdown <= 0)
        self.tile_status[run_now] = 'done'
        delayed = pending & (self.tile_countdown > 0)
        self.tile_countdown[delayed] -= 1
        self.tile_available = np.ones(len(self.tile_status), dtype='bool')
        self.log.info('Completed {} tiles with {} delayed on {}.'
                      .format(np.count_nonzero(run_now),
                              np.count_nonzero(delayed), night))

    def fiberassign(self, dirname):
        r"""Update list of tiles available for spectroscopy.

        Scans given directory looking for fiberassign file and populates Plan
        object accordingly.

        Parameters
        ----------
        dirname : str
            file name of directory where fiberassign files are to be found
            This directory is recursively scanned for all files with names
            matching tile-(\d+)\.fits.  TILEIDs are populated according to
            the name of the fiberassign file, and any header information is
            ignored.
        """
        import glob
        import re
        files = glob.glob(os.path.join(dirname, '**/*.fits*'), recursive=True)
        rgx = re.compile('.*fiberassign-(\d+)\.fits(\.gz)?')
        available_tileids = []
        for fn in files:
            match = rgx.match(fn)
            if match:
                available_tileids.append(int(match.group(1)))
        available = np.zeros(self.tiles.ntiles, dtype='bool')
        ind, mask = self.tiles.index(available_tileids, return_mask=True)
        available[ind[mask]] = True
        self.tile_available = available.copy()
        self.log.info('Fiber assignment files found for {} tiles.'.format(
            np.count_nonzero(available)))
        if np.count_nonzero(available) == 0:
            self.log.error('No fiberassign files available for scheduling!')
        if np.any(~mask):
            self.log.warning(
                'Ignoring {} tiles that were assigned, '.format(sum(~mask)) +
                'but not found in the tile file.')

    def afternoon_plan(self, night, fiber_assign_dir=None):
        """Update plan for a given night.  Update tile availability and priority.

        Parameters
        ----------
        night : str
            night string, YYYY-MM-DD, to be planned.
            This argument has no effect for when Plan.simulate = False; in this
            case, tile availability and priority is based entirely on what
            files are currently present in the fiberassign directory and what
            the planner believes the current tile completions are.

        Returns
        -------
        new_observed (array), new_completed (array)
            boolean arrays indicating newly observed and newly completed tiles.
        """
        if self.first_night is None:
            # Remember the first night of the survey.
            self.first_night = night
        day_number = (night - self.first_night).days

        oldstatus = self.tile_status.copy()
        newlystarted = ((self.donefrac > 0) & (self.tile_status == 'unobs'))
        self.tile_status[newlystarted] = 'obsstart'
        newlyobserved = ((self.donefrac >= self.min_snr2_fraction) &
                         ((self.tile_status == 'obsstart') |
                          (self.tile_status == 'unobs')))
        self.tile_status[newlyobserved] = 'obsend'
        self.log.debug(('Starting afternoon planning for {} with {} / {} '
                        'tiles completed.').format(
                            night,
                            np.sum(self.tile_status == 'done'),
                            self.tiles.ntiles))
        if self.simulate:
            # Update fiber assignments this afternoon?
            if self.fiberassign_cadence == 'monthly':
                # Run fiber assignment on the afternoon before the full moon.
                dt = self.ephem.get_night(night)['nearest_full_moon']
                run_fiberassign = (dt > -0.5) and (dt <= 0.5)
                assert run_fiberassign == self.ephem.is_full_moon(
                    night, num_nights=1)
            elif isinstance(self.fiberassign_cadence, int):
                run_fiberassign = (day_number % self.fiberassign_cadence) == 0
            else:
                run_fiberassign = True
            if run_fiberassign:
                self.fiberassign_simulate(night)
            self.last_night = night
        else:
            self.log.error('we need a mechanism to mark completed tiles.')
            fiber_assign_dir = get_fiber_assign_dir(fiber_assign_dir)
            if fiber_assign_dir is None:
                raise ValueError(
                    'fiber_assign_dir must be set either in '
                    'config.yaml or in FIBER_ASSIGN_DIR; failing!')
            self.fiberassign(fiber_assign_dir)
        # Update tile priorities.
        if self.rules is not None:
            self.tile_priority = self.rules.apply(self.donefrac)
        self.last_night = night
        pending = ((self.tile_status == 'obsstart') |
                   (self.tile_status == 'obsend'))
        for tileid in self.tiles.tileID[pending]:
            self.add_pending_tile(tileid)
        newlycompleted = ((self.tile_status == 'done') &
                          (oldstatus != 'done'))
        return newlystarted, newlycompleted
