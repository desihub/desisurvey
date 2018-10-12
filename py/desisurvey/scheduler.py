"""Schedule observations during an observing night.

This module supercedes desisurvey.old.schedule.
"""
from __future__ import print_function, division

import numpy as np

import astropy.table
import astropy.units as u

import desiutil.log

import desimodel.io

import desisurvey.config
import desisurvey.utils
import desisurvey.etc
import desisurvey.ephemerides


class Scheduler(object):
    """Create a new Scheduler.

    Reads global configuration, tiles, design hour angles and ephemerides.

    The tiles, programs and passes to observe are specified by the tiles file.
    Program names are predefined in our config, but not all programs need
    to be represented.  Pass numbers are arbitrary integers and do not
    need to be consecutive or dense.

    All output uses desiutil.log.
    """
    def __init__(self, verbose=True):
        self.log = desiutil.log.get_logger()
        # Load our configuration.
        config = desisurvey.config.Configuration()
        valid_programs = list(config.programs.keys)
        ##################################################################
        ### self.min_snr2frac = config.min_snr2_fraction()
        # Keep this at one until we simulate errors in ETC integration.
        self.min_snr2frac = 1.
        ##################################################################
        GRAY = desisurvey.config.Configuration().programs.GRAY
        self.max_prod = GRAY.max_moon_illumination_altitude_product().to(u.deg).value
        self.max_frac = GRAY.max_moon_illumination()
        self.threshold_alt = self.max_prod / self.max_frac
        self.max_airmass = desisurvey.utils.cos_zenith_to_airmass(np.sin(config.min_altitude()))
        # Read the tiles to schedule.
        tiles = desimodel.io.load_tiles(
            onlydesi=True, extra=False, tilesfile=config.tiles_file())
        # Save tile info needed for scheduling.
        self.tileid = tiles['TILEID'].copy()
        if not np.all(np.diff(self.tileid) > 0):
            raise RuntimeError('Tile IDs are not increasing.')
        self.passnum = tiles['PASS'].copy()
        self.tileRA = np.radians(tiles['RA'])
        self.tileDEC = np.radians(tiles['DEC'])
        # Build program <-> pass mappings. The programs present must be a subset
        # of those defined in our config. Pass numbers are arbitrary integers
        # and do not need to be consecutive or dense.
        tile_programs = np.unique(tiles['PROGRAM'])
        unknown = set(tile_programs) - set(valid_programs)
        if unknown:
            raise RuntimeError('Cannot schedule unknown program(s): {}.'.format(unknown))
        self.program_passes = {
            p: np.unique(self.passnum[tiles['PROGRAM'] == p]) for p in tile_programs}
        self.pass_program = {}
        for p in tile_programs:
            self.pass_program.update({passnum: p for passnum in self.program_passes[p]})
        # Save tile programs in canonical order.
        self.programs = [p for p in valid_programs if p in tile_programs]
        # Build a dictionary for mapping from program name to a small index.
        self.program_index = {pname: pidx for pidx, pname in enumerate(self.programs)}
        # Initialize arrays
        ntiles = len(self.tileid)
        self.snr2frac = np.zeros(ntiles)
        self.exposure_factor = np.zeros(ntiles)
        self.hour_angle = np.zeros(ntiles)
        self.cosdHA = np.zeros(ntiles)
        self.airmass = np.zeros(ntiles)
        self.avail = np.ones(ntiles, bool)
        self.completed_by_pass = np.zeros(len(self.pass_program), np.int32)
        self.program_mask = {}
        self.tile_sel = np.zeros(ntiles, bool)
        self.LST = 0.
        for p in self.programs:
            mask = np.zeros(ntiles, bool)
            for pnum in self.program_passes[p]:
                mask |= (self.passnum == pnum)
            self.program_mask[p] = mask
        # Calculate and save dust exposure factors.
        self.dust_factor = desisurvey.etc.dust_exposure_factor(tiles['EBV_MED'])
        # Save design hour angles in radians.
        surveyinit_t = astropy.table.Table.read(config.get_path('surveyinit.fits'))
        self.design_hour_angle = np.radians(surveyinit_t['HA'].data)
        # Precompute coefficients to calculate tile observing airmass.
        latitude = np.radians(config.location.latitude())
        tile_dec = np.radians(tiles['DEC'])
        self.tile_coef_A = np.sin(tile_dec) * np.sin(latitude)
        self.tile_coef_B = np.cos(tile_dec) * np.cos(latitude)
        # Load the ephemerides to use.
        self.ephem = desisurvey.ephemerides.Ephemerides()
        if verbose:
            self.log.info(
                'Created scheduler for {} passes of {}.'.format(
                len(self.pass_program), ','.join(self.programs)))

    def reset(self):
        self.snr2frac[:] = 0.
        self.avail[:] = True
        self.completed_by_pass[:] = 0
        self.tile_sel[:] = False

    def init_night(self, night, use_twilight, verbose=False):
        self.use_twilight = use_twilight
        self.night_ephem = self.ephem.get_night(night)
        # Lookup the program for this night.
        self.night_programs, self.night_changes = self.ephem.get_night_program(
            night, include_twilight=use_twilight)
        if verbose:
            midnight = self.night_ephem['noon'] + 0.5
            self.log.info('Program: {}'.format(self.night_programs))
            self.log.info('Changes: {}'.format(np.round(24 * (self.night_changes - midnight), 3)))
        # Initialize linear interpolation of MJD -> LST in radians during this night.
        self.MJD0, MJD1 = self.night_ephem['brightdusk'], self.night_ephem['brightdawn']
        self.LST0, LST1 = np.radians(
            [self.night_ephem['brightdusk_LST'], self.night_ephem['brightdawn_LST']])
        self.dLST = (LST1 - self.LST0) / (MJD1 - self.MJD0)

        self.night_index = 0
        # Is this useful?
        self.last_idx = None

    def next_tile(self, mjd_now, ETC, seeing, transp, method='design'):
        """Return the next (tileid, program) to observe or None.
        """
        # Which program are we in?
        while mjd_now >= self.night_changes[self.night_index + 1]:
            self.night_index += 1
        program = self.night_programs[self.night_index]
        # How much time remaining in this program?
        mjd_program_end = self.night_changes[self.night_index + 1]
        t_remaining = mjd_program_end - mjd_now
        # Select available tiles in this program.
        self.tile_sel = self.program_mask[program] & self.avail
        if not np.any(self.tile_sel):
            return None, None, None, None, program, mjd_program_end
        # Calculate the local apparent sidereal time in radians.
        self.LST = self.LST0 + self.dLST * (mjd_now - self.MJD0)
        # Calculate the hour angle of each available tile in radians.
        #######################################################
        ### should be offset to estimated exposure midpoint ###
        #######################################################
        self.hour_angle[:] = 0.
        self.hour_angle[self.tile_sel] = self.LST - self.tileRA[self.tile_sel]
        # Calculate the airmass of each available tile.
        self.airmass[:] = self.max_airmass
        cosZ = self.tile_coef_A[self.tile_sel] + self.tile_coef_B[self.tile_sel] * np.cos(self.hour_angle[self.tile_sel])
        self.airmass[self.tile_sel] = desisurvey.utils.cos_zenith_to_airmass(cosZ)
        self.tile_sel &= self.airmass < self.max_airmass
        if not np.any(self.tile_sel):
            return None, None, None, None, program, mjd_program_end
        # Estimate exposure factors for all available tiles.
        self.exposure_factor[:] = 1e8
        self.exposure_factor[self.tile_sel] = self.dust_factor[self.tile_sel]
        self.exposure_factor[self.tile_sel] *= desisurvey.etc.airmass_exposure_factor(self.airmass[self.tile_sel])
        # Apply global weather factors that are the same for all tiles.
        global_factor = desisurvey.etc.seeing_exposure_factor(seeing) * desisurvey.etc.transparency_exposure_factor(transp)
        self.exposure_factor[self.tile_sel] *= global_factor
        # Restrict to tiles that could be completed in the remaining time.
        self.tile_sel[self.tile_sel] &= ETC.could_complete(
            t_remaining, program, self.snr2frac[self.tile_sel], self.exposure_factor[self.tile_sel])
        if not np.any(self.tile_sel):
            return None, None, None, None, program, mjd_program_end
        if method == 'greedy':
            # Pick the tile with the smallest exposure factor.
            idx = np.argmin(self.exposure_factor)
        else:
            # Pick the tile that is closest to its design hour angle.
            self.cosdHA[:] = -1
            self.cosdHA[self.tile_sel] = np.cos(self.hour_angle[self.tile_sel] - self.design_hour_angle[self.tile_sel])
            idx = np.argmax(self.cosdHA)
        return self.tileid[idx], self.passnum[idx], self.snr2frac[idx], self.exposure_factor[idx], program, mjd_program_end

    def get_tile_index(self, tileid):
        idx = np.searchsorted(self.tileid, tileid)
        if self.tileid[idx] != tileid:
            raise ValueError('Invalid tileid: {}.'.format(tileid))
        return idx

    def update_tile(self, tileid, snr2frac):
        """Update SNR for one tile and return True if any tiles remain.
        """
        idx = self.get_tile_index(tileid)
        self.snr2frac[idx] = snr2frac
        if self.snr2frac[idx] >= self.min_snr2frac:
            self.avail[idx] = False
            self.completed_by_pass[self.passnum[idx]] += 1
        # Is this useful?
        self.last_idx = idx

    def complete(self):
        return not any(self.avail)