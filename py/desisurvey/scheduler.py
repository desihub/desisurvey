"""Schedule observations during an observing night.

This module supercedes desisurvey.old.schedule.
"""
from __future__ import print_function, division

import os.path

import numpy as np

import astropy.table
import astropy.io.fits
import astropy.units as u

import desiutil.log

import desimodel.io

import desisurvey.config
import desisurvey.utils
import desisurvey.etc
import desisurvey.tiles
import desisurvey.ephem


class Scheduler(object):
    """Create a new Scheduler.

    Reads global configuration, tiles, design hour angles and ephemerides.

    The tiles, programs and passes to observe are specified by the tiles file.
    Program names are predefined in our config, but not all programs need
    to be represented.  Pass numbers are arbitrary integers and do not
    need to be consecutive or dense.

    Design hour angles can be read from the output of ``surveyinit`` using
    :func:`desisurvey.plan.load_design_hourangle`.

    Parameters
    ----------
    planner : :class:`desisurvey.plan.Planner`
        Planner object to use for initialzing tiles to schedule.
    design_hourangles : array or None
        1D array of design hour angles to use in degrees, or use
        :func:`desisurvey.plan.load_design_hourangle` when None.
    restore : str or None
        Restore internal state from the snapshot saved to this filename,
        or initialize a new scheduler when None. Use :meth:`save` to
        save a snapshot to be restored later.
    snr2frac : array or None
        Array of fractional SNR**2 values accumulated so far per tile.
        Initialized to zero when None. This is the only internal state
        required to restore a scheduler object.
    tiles_file : str or None
        Use this file containing the tile definitions, or the default
        specified in the configuration when None.
    """
    def __init__(self, design_hourangle=None, restore=None, tiles_file=None):
        self.log = desiutil.log.get_logger()
        # Load our configuration.
        config = desisurvey.config.Configuration()
        self.min_snr2frac = config.min_snr2_fraction()
        GRAY = desisurvey.config.Configuration().programs.GRAY
        self.max_prod = GRAY.max_moon_illumination_altitude_product().to(u.deg).value
        self.max_frac = GRAY.max_moon_illumination()
        self.threshold_alt = self.max_prod / self.max_frac
        self.max_airmass = desisurvey.utils.cos_zenith_to_airmass(np.sin(config.min_altitude()))
        # Load static tile info.
        self.tiles = desisurvey.tiles.get_tiles(tiles_file)
        ntiles = self.tiles.ntiles
        # Check hourangles.
        if design_hourangle is None:
            self.design_hourangle = desisurvey.plan.load_design_hourangle()
        else:
            self.design_hourangle = np.asarray(design_hourangle)
        if self.design_hourangle.shape != (self.tiles.ntiles,):
            raise ValueError('Array design_hourangle has wrong shape.')
        # Initialize snr2frac, which is our only internal state.
        if restore is not None:
            # Restore the snr2frac array for a survey in progress.
            fullname = config.get_path(restore)
            if not os.path.exists(fullname):
                raise RuntimeError('Cannot restore scheduler from non-existent "{}".'.format(fullname))
            with astropy.io.fits.open(fullname, memmap=False) as hdus:
                self.snr2frac = hdus[0].data.copy()
            if self.snr2frac.shape != (ntiles,):
                raise ValueError('Invalid snr2frac array shape.')
            self.log.debug('Restored scheduler snapshot from "{}".'.format(fullname))
        else:
            # Initialize for a new survey.
            self.snr2frac = np.zeros(ntiles, float)
        # Initialize arrays derived from snr2frac.
        # Note that indexing of completed_by_pass uses tiles.pass_index, which is not necessarily
        # the same as range(tiles.npasses).
        self.completed = (self.snr2frac >= self.min_snr2frac)
        self.completed_by_pass = np.zeros(self.tiles.npasses, np.int32)
        for passnum in self.tiles.passes:
            idx = self.tiles.pass_index[passnum]
            self.completed_by_pass[idx] = np.count_nonzero(self.completed[self.tiles.passnum == passnum])
        # Allocate memory for internal arrays.
        self.exposure_factor = np.zeros(ntiles)
        self.hourangle = np.zeros(ntiles)
        self.cosdHA = np.zeros(ntiles)
        self.airmass = np.zeros(ntiles)
        self.in_night_pool = np.zeros(ntiles, bool)
        self.tile_sel = np.zeros(ntiles, bool)
        self.LST = 0.
        self.night = None
        # Initialize tile priority and available arrays.
        self.tile_priority = None
        self.tile_available = None
        # Load the ephemerides to use.
        self.ephem = desisurvey.ephem.get_ephem()
        # Initialize tile availability and priority.
        # No tiles will be scheduled until these are updated using update_tiles().
        self.tile_available = np.zeros(self.tiles.ntiles, bool)
        self.tile_planned = np.zeros(self.tiles.ntiles, bool)
        self.tile_priority = np.zeros(self.tiles.ntiles, float)

    def save(self, name):
        """Save a snapshot of our current state that can be restored.

        The only internal state required to restore a Scheduler is the array
        of snr2frac values per tile.

        Snapshot is saved to $DESISURVEY_OUTPUT/scheduler_YYYYMMDD.fits using the
        date when :meth:`init_night` was last run.
        The snapshot file size is about ??Kb.

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
        hdr = astropy.io.fits.Header()
        # Record the last night this scheduler was initialized for.
        hdr['NIGHT'] = self.night.isoformat() if self.night else ''
        # Record the number of completed tiles.
        hdr['NDONE'] = self.completed_by_pass.sum()
        # Save a copy of our snr2frac array.
        astropy.io.fits.PrimaryHDU(self.snr2frac, header=hdr).writeto(fullname, overwrite=True)
        self.log.debug('Saved scheduler snapshot to "{}".'.format(fullname))

    def update_tiles(self, tile_available, tile_priority):
        """Update tile availability and priority.

        A valid update must have some tiles available with priority > 0.
        Once a tile has been "planned", i.e., assigned priority > 0, it can
        not be later un-planned, i.e., assigned zero priority.

        Parameters
        ----------
        tile_available : array
            1D array of booleans to indicate which tiles have had fibers assigned
            and so are available to schedule.
        tile_priority : array
            1D array of per-tile priority values >= 0 used to implement survey strategy.

        Returns
        -------
        tuple
            Tuple (new_available, new_planned) of 1D arrays of tile indices that
            identify any tiles are newly available or "planned" (assigned priority > 0).
        """
        if np.any(tile_priority < 0):
            raise ValueError('All tile priorities must be >= 0.')
        new_available = tile_available & ~self.tile_available
        new_planned = (tile_priority > 0) & ~self.tile_planned
        new_unplanned = (tile_priority == 0) & self.tile_planned
        if np.any(new_unplanned):
            raise RuntimeError('Some previously planned tiles now have zero priority.')
        self.tile_available[new_available] = True
        self.tile_planned[new_planned] = True
        self.tile_priority[new_planned] = tile_priority[new_planned]
        if not np.any(self.tile_available & self.tile_planned):
            raise ValueError('No available tiles with priority > 0 to schedule.')
        return np.where(new_available)[0], np.where(new_planned)[0]

    def init_night(self, night, use_twilight=False, verbose=False):
        """Initialize scheduling for the specified night.

        Must be called before calls to :meth:`next_tile` and
        :meth:`update_tile` during the night.

        Tile availability and priority is assumed fixed during the night.
        """
        if self.tile_available is None or self.tile_priority is None:
            raise RuntimeError('Must call update_tiles() before init_night().')
        self.night = night
        self.use_twilight = use_twilight
        self.night_ephem = self.ephem.get_night(night)
        # Lookup the program for this night.
        self.night_programs, self.night_changes = self.ephem.get_night_program(
            night, include_twilight=use_twilight)
        if verbose:
            midnight = self.night_ephem['noon'] + 0.5
            self.log.info('Program: {}'.format(self.night_programs))
            self.log.info('Changes: {}'.format(np.round(24 * (self.night_changes - midnight), 3)))
        # Initialize linear interpolation of MJD -> LST in degrees during this night.
        self.MJD0, MJD1 = self.night_ephem['brightdusk'], self.night_ephem['brightdawn']
        self.LST0, LST1 = [
            self.night_ephem['brightdusk_LST'], self.night_ephem['brightdawn_LST']]
        self.dLST = (LST1 - self.LST0) / (MJD1 - self.MJD0)
        # Initialize tracking of the program through the night.
        self.night_index = 0
        # Remember the last tile observed this night.
        self.last_idx = None
        # Initialize the pool of tiles that could be observed this night.
        self.in_night_pool[:] = ~self.completed & self.tile_planned & self.tile_available

    def next_tile(self, mjd_now, ETC, seeing, transp, method='design'):
        """Select the next tile to observe.

        The :meth:`init_night` method must be called before calling this
        method during a night.

        Parameters
        ----------
        mjd_now : float
            Time when the decision is being made.
        ETC : :class:`desisurvey.etc.ExposureTimeCalculator`
            Object with a method ``could_complete()`` that is used to determine
            which tiles could be completed within a specified amount of time
            under current observing conditions. This use of ETC does not
            change its internal state.
        seeing : float
            Estimate of current atmospherid seeing in arcseconds.
        transp : float
            Estimate of current atmospheric transparency in the range 0-1.

        Returns
        -------
        tuple
            Tuple (TILEID,PASSNUM,SNR2FRAC,EXPFAC,AIRMASS,PROGRAM,PROGEND)
            giving the ID and associated properties of the selected tile.
            When TILEID is None, no tile is observable and this method
            should be called again after some delay.
        """
        if self.night is None:
            raise ValueError('Must call init_night() before next_tile().')
        # Which program are we in?
        while mjd_now >= self.night_changes[self.night_index + 1]:
            self.night_index += 1
        program = self.night_programs[self.night_index]
        # How much time remaining in this program?
        mjd_program_end = self.night_changes[self.night_index + 1]
        t_remaining = mjd_program_end - mjd_now
        # Select available tiles in this program.
        self.tile_sel = self.tiles.program_mask[program] & self.in_night_pool
        if not np.any(self.tile_sel):
            return None, None, None, None, None, program, mjd_program_end
        # Calculate the local apparent sidereal time in degrees.
        self.LST = self.LST0 + self.dLST * (mjd_now - self.MJD0)
        # Calculate the hour angle of each available tile in degrees.
        #######################################################
        ### should be offset to estimated exposure midpoint ###
        #######################################################
        self.hourangle[:] = 0.
        self.hourangle[self.tile_sel] = self.LST - self.tiles.tileRA[self.tile_sel]
        # Calculate the airmass of each available tile.
        self.airmass[:] = self.max_airmass
        self.airmass[self.tile_sel] = self.tiles.airmass(
            self.hourangle[self.tile_sel], self.tile_sel)
        self.tile_sel &= self.airmass < self.max_airmass
        if not np.any(self.tile_sel):
            return None, None, None, None, None, program, mjd_program_end
        # Estimate exposure factors for all available tiles.
        self.exposure_factor[:] = 1e8
        self.exposure_factor[self.tile_sel] = self.tiles.dust_factor[self.tile_sel]
        self.exposure_factor[self.tile_sel] *= desisurvey.etc.airmass_exposure_factor(self.airmass[self.tile_sel])
        # Apply global weather factors that are the same for all tiles.
        global_factor = desisurvey.etc.seeing_exposure_factor(seeing) * desisurvey.etc.transparency_exposure_factor(transp)
        self.exposure_factor[self.tile_sel] *= global_factor
        # Restrict to tiles that could be completed in the remaining time.
        self.tile_sel[self.tile_sel] &= ETC.could_complete(
            t_remaining, program, self.snr2frac[self.tile_sel], self.exposure_factor[self.tile_sel])
        if not np.any(self.tile_sel):
            return None, None, None, None, None, program, mjd_program_end
        if method == 'greedy':
            # Pick the tile with the smallest exposure factor.
            idx = np.argmin(self.exposure_factor)
        else:
            # Pick the tile that is closest to its design hour angle.
            self.cosdHA[:] = -1
            self.cosdHA[self.tile_sel] = np.cos(np.radians(
                self.hourangle[self.tile_sel] - self.design_hourangle[self.tile_sel]))
            idx = np.argmax(self.cosdHA)
        return (self.tiles.tileID[idx], self.tiles.passnum[idx],
                self.snr2frac[idx], self.exposure_factor[idx],
                self.airmass[idx], program, mjd_program_end)

    def update_snr(self, tileID, snr2frac):
        """Update SNR for one tile.
        """
        idx = self.tiles.index(tileID)
        self.snr2frac[idx] = snr2frac
        if self.snr2frac[idx] >= self.min_snr2frac:
            self.in_night_pool[idx] = False
            self.completed[idx] = True
            passidx = self.tiles.pass_index[self.tiles.passnum[idx]]
            self.completed_by_pass[passidx] += 1
        # Remember the last tile observed this night.
        self.last_idx = idx

    def survey_completed(self):
        """Test if all tiles have been completed.
        """
        return self.completed_by_pass.sum() == self.tiles.ntiles
