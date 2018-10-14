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
import desisurvey.tiles
import desisurvey.ephemerides


class Scheduler(object):
    """Create a new Scheduler.

    Reads global configuration, tiles, design hour angles and ephemerides.

    The tiles, programs and passes to observe are specified by the tiles file.
    Program names are predefined in our config, but not all programs need
    to be represented.  Pass numbers are arbitrary integers and do not
    need to be consecutive or dense.
    """
    def __init__(self, tiles_file=None):
        self.log = desiutil.log.get_logger()
        # Load our configuration.
        config = desisurvey.config.Configuration()
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
        # Load static tile info.
        self.tiles = desisurvey.tiles.get_tiles(tiles_file)
        # Initialize arrays
        ntiles = self.tiles.ntiles
        self.snr2frac = np.zeros(ntiles)
        self.exposure_factor = np.zeros(ntiles)
        self.hour_angle = np.zeros(ntiles)
        self.cosdHA = np.zeros(ntiles)
        self.airmass = np.zeros(ntiles)
        self.avail = np.ones(ntiles, bool)
        self.completed_by_pass = np.zeros(self.tiles.npasses, np.int32)
        self.tile_sel = np.zeros(ntiles, bool)
        self.LST = 0.
        # Save design hour angles in degrees.
        surveyinit_t = astropy.table.Table.read(config.get_path('surveyinit.fits'))
        self.design_hour_angle = surveyinit_t['HA'].data.copy()
        # Load the ephemerides to use.
        self.ephem = desisurvey.ephemerides.Ephemerides()

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
        # Initialize linear interpolation of MJD -> LST in degrees during this night.
        self.MJD0, MJD1 = self.night_ephem['brightdusk'], self.night_ephem['brightdawn']
        self.LST0, LST1 = [
            self.night_ephem['brightdusk_LST'], self.night_ephem['brightdawn_LST']]
        self.dLST = (LST1 - self.LST0) / (MJD1 - self.MJD0)

        self.night_index = 0
        # Is this useful?
        self.last_idx = None

    def next_tile(self, mjd_now, ETC, seeing, transp, method='design'):
        """Return the next tile to observe or None.
        """
        # Which program are we in?
        while mjd_now >= self.night_changes[self.night_index + 1]:
            self.night_index += 1
        program = self.night_programs[self.night_index]
        # How much time remaining in this program?
        mjd_program_end = self.night_changes[self.night_index + 1]
        t_remaining = mjd_program_end - mjd_now
        # Select available tiles in this program.
        self.tile_sel = self.tiles.program_mask[program] & self.avail
        if not np.any(self.tile_sel):
            return None, None, None, None, None, program, mjd_program_end
        # Calculate the local apparent sidereal time in degrees.
        self.LST = self.LST0 + self.dLST * (mjd_now - self.MJD0)
        # Calculate the hour angle of each available tile in degrees.
        #######################################################
        ### should be offset to estimated exposure midpoint ###
        #######################################################
        self.hour_angle[:] = 0.
        self.hour_angle[self.tile_sel] = self.LST - self.tiles.tileRA[self.tile_sel]
        # Calculate the airmass of each available tile.
        self.airmass[:] = self.max_airmass
        self.airmass[self.tile_sel] = self.tiles.airmass(
            self.hour_angle[self.tile_sel], self.tile_sel)
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
                self.hour_angle[self.tile_sel] - self.design_hour_angle[self.tile_sel]))
            idx = np.argmax(self.cosdHA)
        return (self.tiles.tileID[idx], self.tiles.passnum[idx],
                self.snr2frac[idx], self.exposure_factor[idx],
                self.airmass[idx], program, mjd_program_end)

    def update_tile(self, tileID, snr2frac):
        """Update SNR for one tile and return True if any tiles remain.
        """
        idx = self.tiles.index(tileID)
        self.snr2frac[idx] = snr2frac
        if self.snr2frac[idx] >= self.min_snr2frac:
            self.avail[idx] = False
            passidx = self.tiles.pass_index[self.tiles.passnum[idx]]
            self.completed_by_pass[passidx] += 1
        # Is this useful?
        self.last_idx = idx

    def complete(self):
        return not any(self.avail)
