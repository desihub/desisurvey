"""Manage static information associated with tiles, programs and passes.
"""
from __future__ import print_function, division

import numpy as np

import desimodel.io
import desiutil.log

import desisurvey.config
import desisurvey.utils
import desisurvey.etc


class Tiles(object):
    """Manage static info associated with the tiles file.
    
    The ``tiles_file`` configuration parameter determines which tiles
    file is read.
    
    Each tile has an assigned program name and pass number.
    Program names are predefined in our config, but not all programs need
    to be represented.  Pass numbers are arbitrary integers and do not
    need to be consecutive or dense.

    Parameters
    ----------
    tile_file : str or None
        Name of the tiles file to use or None for the default specified
        in our configuration.
    """
    def __init__(self, tiles_file=None):
        log = desiutil.log.get_logger()
        config = desisurvey.config.Configuration()
        valid_programs = list(config.programs.keys)
        tiles_file = tiles_file or config.tiles_file()
        tiles = desimodel.io.load_tiles(
            onlydesi=True, extra=False, tilesfile=tiles_file)
        # Copy tile arrays.
        self.tileID = tiles['TILEID'].copy()
        self.passnum = tiles['PASS'].copy()
        self.tileRA = tiles['RA'].copy()
        self.tileDEC = tiles['DEC'].copy()
        # Count tiles.
        self.ntiles = len(self.tileID)
        self.pass_ntiles = {p: np.count_nonzero(self.passnum == p)
                            for p in np.unique(self.passnum)}
        # Get list of passes.
        self.passes = np.unique(self.passnum)
        self.npasses = len(self.passes)
        # Map each pass to a small integer index.
        self.pass_index = {p: idx for idx, p in enumerate(self.passes)}
        # Can remove this when tile_index no longer uses searchsorted.
        if not np.all(np.diff(self.tileID) > 0):
            raise RuntimeError('Tile IDs are not increasing.')
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
        # Build tile masks for each program.
        self.program_mask = {}
        for p in self.programs:
            mask = np.zeros(self.ntiles, bool)
            for pnum in self.program_passes[p]:
                mask |= (self.passnum == pnum)
            self.program_mask[p] = mask
        # Calculate and save dust exposure factors.
        self.dust_factor = desisurvey.etc.dust_exposure_factor(tiles['EBV_MED'])
        # Precompute coefficients to calculate tile observing airmass.
        latitude = np.radians(config.location.latitude())
        tile_dec_rad = np.radians(self.tileDEC)
        self.tile_coef_A = np.sin(tile_dec_rad) * np.sin(latitude)
        self.tile_coef_B = np.cos(tile_dec_rad) * np.cos(latitude)

    def airmass(self, hour_angle, mask=None):
        """Calculate tile airmass given hour angle.

        Parameters
        ----------
        hour_angle : array
            Array of hour angles in degrees to use. If mask is None, then should have length
            ``self.ntiles``.  Otherwise, should have a value per non-zero entry in the mask.
        mask : array or None
            Boolean mask of which tiles to perform the calculation for.

        Returns
        -------
        array
            Array of airmasses corresponding to each input hour angle.
        """
        hour_angle = np.deg2rad(hour_angle)
        mask = mask or slice(None)
        cosZ = self.tile_coef_A[mask] + self.tile_coef_B[mask] * np.cos(hour_angle)
        return desisurvey.utils.cos_zenith_to_airmass(cosZ)

    def index(self, tileID):
        """Map tile ID to array index.

        Parameters
        ----------
        tileID : int or array
            Tile ID value(s) to convert.

        Returns
        -------
        int or array
            Index into internal per-tile arrays corresponding to each input tile ID.
        """
        scalar = np.isscalar(tileID)
        tileID = np.atleast_1d(tileID)
        idx = np.searchsorted(self.tileID, tileID)
        bad = self.tileID[idx] != tileID
        if np.any(bad):
            raise ValueError('Invalid tile ID(s): {}.'.format(tileID[bad]))
        return idx[0] if scalar else idx

_cached_tiles = {}

def get_tiles(tiles_file=None, use_cache=True, write_cache=True):
    """Return a Tiles object with optional caching.
    """
    global _cached_tiles

    log = desiutil.log.get_logger()
    config = desisurvey.config.Configuration()
    valid_programs = list(config.programs.keys)
    tiles_file = tiles_file or config.tiles_file()

    if use_cache and tiles_file in _cached_tiles:
        tiles = _cached_tiles[tiles_file]
        log.info('Using cached tiles for "{}".'.format(tiles_file))
    else:
        tiles = Tiles(tiles_file)
        log.info('Initialized tiles from "{}".'.format(tiles_file))
        for pname in tiles.programs:
            pinfo = []
            for passnum in tiles.program_passes[pname]:
                pinfo.append('{}({})'.format(passnum, tiles.pass_ntiles[passnum]))
            log.info('{:6s} passes(tiles): {}.'.format(pname, ', '.join(pinfo)))

    if write_cache:
        _cached_tiles[tiles_file] = tiles
    else:
        log.info('Tiles not cached for "{}".'.format(tiles_file))

    return tiles