"""Schedule observations during an observing night.

This module supercedes desisurvey.old.schedule.
"""
from __future__ import print_function, division

import os.path

import numpy as np

import astropy.io.fits
import astropy.units as u

import desiutil.log

import desisurvey.config
import desisurvey.utils
import desisurvey.etc
import desisurvey.tiles
import desisurvey.ephem


class Scheduler(object):
    """Create a new next-tile scheduler.

    Design hour angles are read from the output of ``surveyinit`` using
    :func:`desisurvey.plan.load_design_hourangle`, by default.

    The only internal state needed by the scheduler is the list of
    accumulated SNR2 fractions per tile, which can be restored
    from a file created using :meth:`save`.

    A newly created or restored scheduler must be configured with
    calls to :meth:`update_tiles` (to tile availablity and priority)
    and :meth:`init_night` (to precompute data for a night's observing)
    before tiles can be selected.

    Use :meth:`next_tile` to select the next tile to observe during
    a night.  If the tile is observed, the internal state must be
    updated with a call to :meth:`update_snr`.

    Parameters
    ----------
    restore : str or None
        Restore internal state from the snapshot saved to this filename,
        or initialize a new scheduler when None. Use :meth:`save` to
        save a snapshot to be restored later. Filename is relative to
        the configured output path unless an absolute path is
        provided.
    design_hourangles : array or None
        1D array of design hour angles to use in degrees, or use
        :func:`desisurvey.plan.load_design_hourangle` when None,
        when not restoring.
    """
    def __init__(self, restore=None, design_hourangle=None):
        self.log = desiutil.log.get_logger()
        # Load our configuration.
        config = desisurvey.config.Configuration()
        ignore_completed_priority = getattr(config,
                                            'ignore_completed_priority', -1)
        if not isinstance(ignore_completed_priority, int):
            ignore_completed_priority = ignore_completed_priority()
        self.ignore_completed_priority = ignore_completed_priority
        self.min_snr2frac = config.min_snr2_fraction()
        GRAY = desisurvey.config.Configuration().programs.GRAY
        self.max_prod = GRAY.max_moon_illumination_altitude_product().to(u.deg).value
        self.max_frac = GRAY.max_moon_illumination()
        self.threshold_alt = self.max_prod / self.max_frac
        self.max_airmass = desisurvey.utils.cos_zenith_to_airmass(np.sin(config.min_altitude()))
        self.max_ha = config.max_hour_angle().to(u.deg).value
        # Load static tile info.
        self.tiles = desisurvey.tiles.get_tiles()
        ntiles = self.tiles.ntiles
        # Initialize snr2frac, which is our only internal state.
        if restore is not None:
            # Restore the snr2frac array for a survey in progress.
            fullname = config.get_path(restore)
            if not os.path.exists(fullname):
                raise RuntimeError('Cannot restore scheduler from non-existent "{}".'.format(fullname))
            t = astropy.table.Table.read(fullname, hdu='STATUS')
            ind, mask = self.tiles.index(t['TILEID'], return_mask=True)
            ind = ind[mask]
            self.snr2frac = np.zeros(ntiles, 'f4')
            self.lastexpid = np.zeros(ntiles, 'i4')
            self.design_hourangle = np.zeros(ntiles, 'f4')
            self.design_hourangle_cond = np.zeros(ntiles, '3f4')
            self.snr2frac[ind] = t['DONEFRAC'][mask].copy()
            self.lastexpid[ind] = t['LASTEXPID'][mask].copy()
            self.design_hourangle[ind] = t['DESIGNHA'][mask].copy()
            self.design_hourangle_cond[ind] = t['DESIGNHACOND'][mask].copy()
            self.log.debug('Restored scheduler snapshot from "{}".'.format(fullname))
        else:
            # Initialize for a new survey.
            self.snr2frac = np.zeros(ntiles, float)
            self.lastexpid = np.zeros(ntiles, float)
            self.design_hourangle = desisurvey.plan.load_design_hourangle()
            if self.design_hourangle.shape[0] != self.tiles.ntiles:
                raise ValueError('design_hourangle has the wrong shape!')
            self.design_hourangle_cond = np.array([
                desisurvey.plan.load_design_hourangle(condition=cond)
                for cond in ['DARK', 'GRAY', 'BRIGHT']]).T
            if self.design_hourangle.shape[0] != self.tiles.ntiles:
                raise ValueError('design_hourangle has the wrong shape!')

        # Initialize arrays derived from snr2frac.
        # Note that indexing of completed_by_pass uses tiles.pass_index, which is not necessarily
        # the same as range(tiles.npasses).
        self.completed = (self.snr2frac >= self.min_snr2frac)
        self.completed_by_pass = np.zeros(self.tiles.npasses, np.int32)
        for passnum in self.tiles.passes:
            idx = self.tiles.pass_index[passnum]
            self.completed_by_pass[idx] = np.count_nonzero(self.completed[self.tiles.passnum == passnum])

        # Check hourangles.
        if design_hourangle is not None:
            self.design_hourangle = design_hourangle
        if self.design_hourangle.shape != (self.tiles.ntiles,):
            raise ValueError('Array design_hourangle has wrong shape.')


        # Allocate memory for internal arrays.
        self.exposure_factor = np.zeros(ntiles)
        self.hourangle = np.zeros(ntiles)
        self.airmass = np.zeros(ntiles)
        self.in_night_pool = np.zeros(ntiles, bool)
        self.tile_sel = np.zeros(ntiles, bool)
        self.LST = 0.
        self.night = None
        # Load the ephemerides to use.
        self.ephem = desisurvey.ephem.get_ephem()
        # Initialize tile availability and priority.
        # No tiles will be scheduled until these are updated using update_tiles().
        self.tile_available = np.zeros(self.tiles.ntiles, bool)
        self.tile_planned = np.zeros(self.tiles.ntiles, bool)
        self.tile_priority = np.zeros(self.tiles.ntiles, float)
        # Lookup avoidance cone angles.
        self.avoid_bodies = {}
        for body in config.avoid_bodies.keys:
            self.avoid_bodies[body] = getattr(config.avoid_bodies, body)().to(u.deg).value

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
        new_available = tile_available & ~self.tile_available
        new_unavailable = ~tile_available & self.tile_available
        if np.any(new_unavailable):
            raise RuntimeError('Some previously available tiles now unavailable.')
        self.tile_available[new_available] = True

        assert np.all(self.tile_priority >= 0)
        if np.any(tile_priority < 0):
            raise ValueError('All tile priorities must be >= 0.')
        tile_planned = tile_priority > 0
        new_planned = tile_planned & ~self.tile_planned
        new_unplanned = ~tile_planned & self.tile_planned
        if np.any(new_unplanned):
            raise RuntimeError('Some previously planned tiles now have zero priority.')
        self.tile_planned[new_planned] = True
        # Tile priorities can change after they become > 0, so copy all planned priorities,
        # not just the newly planned priorities.
        self.tile_priority[self.tile_planned] = tile_priority[self.tile_planned]
        # Precompute log(priority) since that is what we use for scheduling.
        # Ignore harmless warnings about log(0) = -inf.
        with np.errstate(divide='ignore'):
            self.log_priority = np.log(self.tile_priority)

        if not np.any(self.tile_available & self.tile_planned):
            raise ValueError('No available tiles with priority > 0 to schedule.')
        new_available, new_planned = np.where(new_available)[0], np.where(new_planned)[0]
        self.log.debug('{} new available tiles, {} new planned tiles.'.format(
            len(new_available), len(new_planned)))
        return new_available, new_planned

    def init_night(self, night, use_twilight=False):
        """Initialize scheduling for the specified night.

        Must be called before calls to :meth:`next_tile` and
        :meth:`update_tile` during the night.

        The pool of available tiles during the night consists of those that:

         - Have fibers assigned.
         - Have non-zero priority (aka "planned").
         - Have not already reached their target SNR (aka "completed").
         - Are not too close to a planet during this night.

        Tile availability and priority is assumed fixed during the night.
        When the moon is up, tiles are also vetoed if they are too
        close to the moon. The angles that define "too close" to a
        planet or the moon are specified in config.avoid_bodies.

        Parameters
        ----------
        night : str
            Date on the evening this night starts in the format YYYY-MM-DD.
        use_twilight : bool
            Include twilight when calculating the scheduled program changes
            during this night when True.
        verbose : bool
            Generate verbose logging output when True.
        """
        self.log.debug('Initializing scheduler for {}'.format(night))
        if self.tile_available is None or self.tile_priority is None:
            raise RuntimeError('Must call update_tiles() before init_night().')
        self.night = night
        self.night_ephem = self.ephem.get_night(night)
        midnight = self.night_ephem['noon'] + 0.5
        # Lookup the program for this night.
        self.night_programs, self.night_changes = self.ephem.get_night_program(
            night, include_twilight=use_twilight)
        self.log.debug('  Program: {}'.format(self.night_programs))
        self.log.debug('  Changes: {}'.format(np.round(24 * (self.night_changes - midnight), 3)))
        # Initialize linear interpolation of MJD -> LST in degrees during this night.
        self.MJD0, MJD1 = self.night_ephem['brightdusk'], self.night_ephem['brightdawn']
        self.LST0, LST1 = [
            self.night_ephem['brightdusk_LST'], self.night_ephem['brightdawn_LST']]
        self.dLST = (LST1 - self.LST0) / (MJD1 - self.MJD0)
        # Initialize tracking of the program through the night.
        # Remember the last tile observed this night.
        self.last_idx = None
        # Initialize the pool of tiles that could be observed this night.
        self.in_night_pool[:] = self.tile_planned & self.tile_available
        if self.ignore_completed_priority <= 0:
             self.in_night_pool &= ~self.completed
        # Check if any tiles cannot be observed because they are too close to a planet this night.
        poolRA = self.tiles.tileRA[self.in_night_pool]
        poolDEC = self.tiles.tileDEC[self.in_night_pool]
        avoid_idx = []
        for body in self.avoid_bodies:
            if body == 'moon':
                continue
            # Get body (RA,DEC) at midnight.
            bodyDEC, bodyRA = desisurvey.ephem.get_object_interpolator(
                self.night_ephem, body, altaz=False)(midnight)
            too_close = desisurvey.utils.separation_matrix(
                [bodyRA], [bodyDEC], poolRA, poolDEC, self.avoid_bodies[body])[0]
            if np.any(too_close):
                idx = np.where(self.in_night_pool)[0][too_close]
                tileIDs = self.tiles.tileID[idx]
                self.log.debug('  Tiles within {} deg of {}: {}.'.format(
                    self.avoid_bodies[body], body, ','.join([str(ID) for ID in tileIDs])))
                avoid_idx.extend(idx)
        self.in_night_pool[avoid_idx] = False
        # Initialize moon tracking during this night.
        self.moon_DECRA = desisurvey.ephem.get_object_interpolator(self.night_ephem, 'moon', altaz=False)
        self.moon_ALTAZ = desisurvey.ephem.get_object_interpolator(self.night_ephem, 'moon', altaz=True)
        # Initialize sun tracking during this night.
        self.sun_DECRA = desisurvey.ephem.get_object_interpolator(self.night_ephem, 'sun', altaz=False) 
        self.sun_ALTAZ = desisurvey.ephem.get_object_interpolator(self.night_ephem, 'sun', altaz=True) 

    def select_program(self, mjd_now, ETC, verbose=False):
        """Select program to observe now.
        """
        if mjd_now < self.night_changes[0]:
            if verbose:
                self.log.warning('Tile requested before start of night.')
        if mjd_now > self.night_changes[-1]:
            if verbose:
                self.log.warning('Tile requested after end of night.')
        idx = 0
        while ((idx + 1 < len(self.night_changes)) and
               (mjd_now >= self.night_changes[idx + 1])):
            idx += 1
        idx = min(len(self.night_programs)-1, idx)
        program = self.night_programs[idx]
        # How much time remaining in this program?
        mjd_program_end = self.night_changes[idx + 1]
        nommidpt = mjd_now + (ETC.TEXP_TOTAL[program]/2)
        if (nommidpt > mjd_program_end) & (idx != len(self.night_programs)-1):
            idx += 1
            program = self.night_programs[idx]
            mjd_program_end = self.night_changes[idx + 1]
        return program, mjd_program_end

    def next_tile(self, mjd_now, ETC, seeing, transp, skylevel, HA_sigma=15., greediness=0.,
                  program=None, verbose=False):
        r"""Select the next tile to observe.

        The :meth:`init_night` method must be called before calling this
        method during a night.

        The (log) score for each observable tile is calculated as:

        .. math::

            -(1 - g)\,\frac{1}{2} \left( \frac{\text{HA} - \text{HA}_0}{\sigma_{\text{HA}}}
            \right)^2 - g \log \frac{t_\text{exp}}{t_\text{nom}} + \log P

        where :math:`\text{HA}` and :math:`\text{HA}_0` are the current and design
        hour angles, respectively, :math:`g` is the ``greediness`` parameter below,
        and :math:`P` are the tile priorities used to implement survey strategy
        and updated via :meth:`update_tiles`.

        Parameters
        ----------
        mjd_now : float
            Time when the decision is being made.
        ETC : :class:`desisurvey.etc.ExposureTimeCalculator`
            Object with methods ``could_complete()`` and ``weather_factor()``.
            Normally an instance of :class:`desisurvey.etc.ExposureTimeCalculator`.
        seeing : float
            Estimate of current atmospherid seeing in arcseconds.
        transp : float
            Estimate of current atmospheric transparency in the range 0-1.
        HA_sigma : float
            RMS in degrees for the Gaussian penalty applied to tiles observed
            away from their design hour angle.
        greediness : float
            Parameter that controls the balance between observing at the design
            hour angle and observing tiles with the small exposure-time factor.
            Set this value to zero to only consider hour angle or to one to
            only consider isntantaneous efficiency. The meaning of intermediate
            values will depend on the value of ``HA_sigma`` and how exposure
            factors are calculated. Refer to the equation above for details.
            Must be between 0 and 1.
        program : string
            PROGRAM of tile to select.  Default of None selects the appropriate
            PROGRAM given current moon/twilight conditions.  Forcing a particular
            program leads PROGEND to be infinity.

        Returns
        -------
        tuple
            Tuple (TILEID,PASSNUM,SNR2FRAC,EXPFAC,AIRMASS,PROGRAM,PROGEND)
            giving the ID and associated properties of the selected tile.
            When no tile is observable, only the last two tuple fields
            will be valid, and this method should be called again after
            some dead-time delay.  The tuple fields are:

             - TILEID: ID of the tile to observe.
             - PASSNUM: pass number of the tile to observe.
             - SNR2FRAC: fractional SNR2 already accumulated for the selected tile.
             - EXPFAC: initial exposure-time factor for the selected tile.
             - AIRMASS: initial airmass of the selected tile.
             - PROGRAM: scheduled program at ``mjd_now``, which might be
               different from the program of the selected (TILEID, PASSNUM).
             - PROGEND: MJD timestamp when the scheduled program ends.
        """
        if self.night is None:
            raise ValueError('Must call init_night() before next_tile().')
        if greediness < 0 or greediness > 1:
            raise ValueError('Expected greediness between 0 and 1.')
        self.tile_sel = np.ones(self.tiles.ntiles, dtype=bool)
        if program is None:
            # Which program are we in?
            program, mjd_program_end = self.select_program(
                mjd_now, ETC, verbose=verbose)
            self.tile_sel &= self.tiles.allowed_in_conditions[program]
            if verbose:
                self.log.info(
                    'Selecting a tile observable in {} conditions.'.format(
                        program))
        else:
            self.tile_sel &= self.tiles.program_mask[program]
            mjd_program_end = self.night_changes[-1]  # end of night?
        # Select available tiles in this program.
        self.tile_sel &= self.in_night_pool
        if not np.any(self.tile_sel):
            if verbose:
                self.log.warning('No available tiles in requested program.')
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
        absha = np.abs(((self.hourangle + 180) % 360)-180)
        self.tile_sel &= (absha < self.max_ha)
        if not np.any(self.tile_sel):
            if verbose:
                self.log.warning('No tiles left to observe after HA/airmass cut.')
            return None, None, None, None, None, program, mjd_program_end
        # Is the moon up?
        if mjd_now > self.night_ephem['moonrise'] and mjd_now < self.night_ephem['moonset']:
            moon_is_up = True
            # Calculate the moon (RA,DEC).
            moonDEC, moonRA = self.moon_DECRA(mjd_now)
            # Identify tiles that are too close to the moon to observe now.
            too_close = desisurvey.utils.separation_matrix(
                [moonRA], [moonDEC],
                self.tiles.tileRA[self.tile_sel], self.tiles.tileDEC[self.tile_sel],
                self.avoid_bodies['moon'])[0]
            idx = np.where(self.tile_sel)[0][too_close]
            self.tile_sel[idx] = False
            if not np.any(self.tile_sel):
                if verbose:
                    self.log.warning('No tiles left to observe after moon separation cut.')
                # No tiles left to observe after moon avoidance veto.
                return None, None, None, None, None, program, mjd_program_end
        else:
            moon_is_up = False
        # Estimate exposure factors for all available tiles.
        self.exposure_factor[:] = 1e8
        self.exposure_factor[self.tile_sel] = self.tiles.dust_factor[self.tile_sel]
        self.exposure_factor[self.tile_sel] *= desisurvey.etc.airmass_exposure_factor(self.airmass[self.tile_sel])
        # Apply global weather factors that are the same for all tiles.
        self.exposure_factor[self.tile_sel] /= ETC.weather_factor(seeing, transp)
        if not np.any(self.tile_sel):
            return None, None, None, None, None, program, mjd_program_end
        # Calculate (the log of a) Gaussian multiplicative penalty for
        # observing tiles away from their design hour angle.
        dHA = self.hourangle[self.tile_sel] - self.design_hourangle_cond[self.tile_sel, self.tiles.PROGRAM_INDEX[program]]
        dHA[dHA >= 180.] -= 360
        dHA[dHA < -180] += 360
        assert np.all((dHA >= -180) & (dHA < 180))
        # Calculate a score that combines dHA and instantaneous efficiency.
        log_score = (
            -0.5 * (dHA / HA_sigma) ** 2 * (1 - greediness) +
            -np.log(self.exposure_factor[self.tile_sel]) * greediness)
        # Add tile priorities.
        log_score += self.log_priority[self.tile_sel]
        # Select the tile with the highest (log) score.
        idx = np.where(self.tile_sel)[0][np.argmax(log_score)]

        # Return info about the selected tile and scheduled program.
        return (self.tiles.tileID[idx], self.tiles.passnum[idx],
                self.snr2frac[idx], self.exposure_factor[idx],
                self.airmass[idx], program, mjd_program_end)

    def update_snr(self, tileID, snr2frac, lastexpid):
        """Update SNR for one tile.

        A tile whose update ``snr2frac`` exceeds the ``min_snr2frac``
        configuration parameter will be considered completed, and
        not scheduled for future observing.

        Parameters
        ----------
        tileID : int
            ID of the tile to update.
        snr2frac : float
            New value of the fractional SNR2 accumulated for this tile, including
            all previous exposures.
        """
        idx = self.tiles.index(tileID)
        self.snr2frac[idx] = snr2frac
        self.lastexpid[idx] = lastexpid
        if self.snr2frac[idx] >= self.min_snr2frac:
            if self.ignore_completed_priority <= 0:
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
