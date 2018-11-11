"""Manage static information associated with tiles, programs and passes.
"""
from __future__ import print_function, division

import numpy as np

import astropy.units as u

import desimodel.io
import desiutil.log

import desisurvey.config
import desisurvey.utils
import desisurvey.etc


class Tiles(object):

    # Define the valid programs in canonical order.
    PROGRAMS = ['DARK', 'GRAY', 'BRIGHT']
    # Define a mapping from program name to index 0,1,2.
    # Note that this mapping is independent of the programs actually present
    # in a tiles file.
    PROGRAM_INDEX = {pname: pidx for pidx, pname in enumerate(PROGRAMS)}

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
        # Read the specified tiles file.
        self.tiles_file = tiles_file or config.tiles_file()
        tiles = desimodel.io.load_tiles(
            onlydesi=True, extra=False, tilesfile=self.tiles_file)
        # Check for any unknown program names.
        tile_programs = np.unique(tiles['PROGRAM'])
        unknown = set(tile_programs) - set(self.PROGRAMS)
        if unknown:
            raise RuntimeError('Cannot schedule unknown program(s): {}.'.format(unknown))
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
        # Build program -> [passes] maps. A program with no tiles will map to an empty array.
        self.program_passes = {
            p: np.unique(self.passnum[tiles['PROGRAM'] == p]) for p in self.PROGRAMS}
        # Build pass -> program maps.
        self.pass_program = {}
        for p in self.PROGRAMS:
            self.pass_program.update({passnum: p for passnum in self.program_passes[p]})
        # Build tile masks for each program. A program will no tiles with have an empty mask.
        self.program_mask = {}
        for p in self.PROGRAMS:
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
        # Placeholders for overlap attributes that are expensive to calculate
        # so we use lazy evaluation the first time they are accessed.
        self._tile_over = None
        self._overlapping = None

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
        if mask is None:
            mask = slice(None)
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

    @property
    def tile_over(self):
        """Dictionary of tile masks.

        tile_over[passnum] identifies all tiles from a fiber-assignment
        dependent pass that cover at least one tile in passnum.

        Uses the config parameter ``fiber_assignment_order`` to
        determine fiber-assignment dependencies.
        """
        if self._tile_over is None:
            self._calculate_overlaps()
        return self._tile_over

    @property
    def overlapping(self):
        """Dictionary of tile overlap matrices.

        overlapping[passnum][j, k] is True if the j-th tile of passnum is
        overlapped by the k-th tile of tile_over[passnum]. There is no
        dictionary entry when the mask tile_over[passnum] is empty.
        """
        if self._overlapping is None:
            self._calculate_overlaps()
        return self._overlapping

    def _calculate_overlaps(self):
        """Initialize attributes _overlapping and _tile_over.

        Uses the config parameters ``fiber_assignment_order`` and
        ``tile_diameter`` to determine overlap dependencies.

        This is relatively slow, so only used the first time our ``tile_over``
        or ``overlapping`` properties are accessed.
        """
        self._overlapping = {}
        self._tile_over = {}
        config = desisurvey.config.Configuration()
        fiberassign_order = config.fiber_assignment_order
        tile_diameter = 2 * config.tile_radius().to(u.deg).value
        for passnum in self.passes:
            under = self.passnum == passnum
            over = np.zeros_like(under)
            key = 'P{}'.format(passnum)
            if key in fiberassign_order.keys:
                overpasses = getattr(fiberassign_order, key)()
                for overpass in overpasses.split('+'):
                    if not len(overpass) == 2 and overpass[0] == 'P':
                        raise RuntimeError(
                            'Invalid pass in fiber_assignment_order: {}.'
                            .format(overpass))
                    over |= (self.passnum == int(overpass[1]))
                self.overlapping[passnum] = desisurvey.utils.separation_matrix(
                    self.tileRA[under], self.tileDEC[under],
                    self.tileRA[over], self.tileDEC[over], tile_diameter)
            self.tile_over[passnum] = over


_cached_tiles = {}

def get_tiles(tiles_file=None, use_cache=True, write_cache=True):
    """Return a Tiles object with optional caching.
    """
    global _cached_tiles

    log = desiutil.log.get_logger()
    config = desisurvey.config.Configuration()
    tiles_file = tiles_file or config.tiles_file()

    if use_cache and tiles_file in _cached_tiles:
        tiles = _cached_tiles[tiles_file]
        log.debug('Using cached tiles for "{}".'.format(tiles_file))
    else:
        tiles = Tiles(tiles_file)
        log.info('Initialized tiles from "{}".'.format(tiles_file))
        for pname in Tiles.PROGRAMS:
            pinfo = []
            for passnum in tiles.program_passes[pname]:
                pinfo.append('{}({})'.format(passnum, tiles.pass_ntiles[passnum]))
            log.info('{:6s} passes(tiles): {}.'.format(pname, ', '.join(pinfo)))

    if write_cache:
        _cached_tiles[tiles_file] = tiles
    else:
        log.info('Tiles not cached for "{}".'.format(tiles_file))

    return tiles