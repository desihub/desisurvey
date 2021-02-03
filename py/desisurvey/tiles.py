"""Manage static information associated with tiles, programs and passes.

Each tile has an assigned program name and pass number. The program names
(DARK, GRAY, BRIGHT) are predefined in terms of conditions on the
ephemerides, but not all programs need to be present in a tiles file.
Pass numbers are arbitrary integers and do not need to be consecutive or dense.

To ensure consistent and efficient usage of static tile info, all code
should use::

    tiles = desisurvey.tiles.get_tiles()

To use a non-standard tiles file, change the configuration before the
first call to ``get_tiles()`` with::

    config = desisurvey.config.Configuration()
    config.tiles_file.set_value(name)

The :class:`Tiles` class returned by :func:`get_tiles` is a wrapper around
the FITS table contained in a tiles file, that adds some precomputed derived
attributes for consistency and efficiency.
"""
from __future__ import print_function, division

import re

import numpy as np

import astropy.units as u

import desimodel.io
import desiutil.log

import desisurvey.config
import desisurvey.utils
import desisurvey.etc


class Tiles(object):
    """Manage static info associated with the tiles file.

    Parameters
    ----------
    tile_file : str or None
        Name of the tiles file to use or None for the default specified
        in our configuration.
    """
    def __init__(self, tiles_file=None, bgs_footprint=None):
        log = desiutil.log.get_logger()
        config = desisurvey.config.Configuration()
        # Read the specified tiles file.
        self.tiles_file = tiles_file or config.tiles_file()
        commissioning = getattr(config, 'commissioning', False)
        if not commissioning:
            tiles = desimodel.io.load_tiles(
                onlydesi=True, extra=False, tilesfile=self.tiles_file)
        else:
            tiles = desimodel.io.load_tiles(
                onlydesi=False, extra=True, tilesfile=self.tiles_file)
        # Check for any unknown program names.
        tile_programs = np.unique(tiles['PROGRAM'])
        unknown = set(tile_programs) - set(self.PROGRAMS)
        if unknown and not commissioning:
            raise RuntimeError('Cannot schedule unknown program(s): {}.'.format(unknown))

        if bgs_footprint is not None: 
            # impose reduced footprint for BRIGHT tiles 
            tiles, _reduced_footprint = self._reduced_bgs_footprint(tiles, bgs_footprint) 
            self._reduced_footprint = _reduced_footprint.copy() 
        print(tiles.shape) 

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
        if commissioning:
            Tiles.PROGRAMS = [x for x in np.sort(np.unique(tiles['PROGRAM']))]
            for requiredprogram in ['DARK', 'GRAY', 'BRIGHT']:
                if requiredprogram not in Tiles.PROGRAMS:
                    Tiles.PROGRAMS = [requiredprogram] + Tiles.PROGRAMS
            self.PROGRAMS = Tiles.PROGRAMS
            Tiles.PROGRAM_INDEX = {pname: pidx
                                   for pidx, pname in enumerate(Tiles.PROGRAMS)}
            self.PROGRAM_INDEX = Tiles.PROGRAM_INDEX
            
        # Build program -> [passes] maps. A program with no tiles will map to an empty array.
        self.program_passes = {
            p: np.unique(self.passnum[tiles['PROGRAM'] == p]) for p in self.PROGRAMS}
        # Build pass -> program maps.
        self.pass_program = {}
        for p in self.PROGRAMS:
            self.pass_program.update({passnum: p for passnum in self.program_passes[p]})
        for p in np.unique(self.passnum):
            if len(np.unique(tiles['PROGRAM'][tiles['PASS'] == p])) != 1:
                raise ValueError('At most one program per pass.')
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
        self._fiberassign_delay = None

    PROGRAMS = ['DARK', 'GRAY', 'BRIGHT']
    """Enumeration of the valid programs in their canonical order."""

    PROGRAM_INDEX = {pname: pidx for pidx, pname in enumerate(PROGRAMS)}
    """Canonical mapping from program name to a small integer.

    Note that this mapping is independent of the programs actually present
    in a tiles file.
    """

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

    def index(self, tileID, return_mask=False):
        """Map tile ID to array index.

        Parameters
        ----------
        tileID : int or array
            Tile ID value(s) to convert.
        mask : bool
            if mask=True, an additional mask array is returned, indicating which
            IDs were present in the tile array.  Otherwise, an exception is
            raised if tiles were not found.

        Returns
        -------
        int or array
            Index into internal per-tile arrays corresponding to each input tile ID.
        """
        scalar = np.isscalar(tileID)
        tileID = np.atleast_1d(tileID)
        idx = np.searchsorted(self.tileID, tileID)
        bad = self.tileID[idx] != tileID
        if not return_mask and np.any(bad):
            raise ValueError('Invalid tile ID(s): {}.'.format(tileID[bad]))
        mask = ~bad
        idx = idx[0] if scalar else idx
        mask = mask[0] if scalar else mask
        res = idx
        if return_mask:
            res = (res, mask)
        return res

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

    @property
    def fiberassign_delay(self):
        """Delay between covering a tile and when it can be fiber assigned.

        Units are determined by the value of the fiber_assignment_cadence
        configuration parameter.
        """
        if self._fiberassign_delay is None:
            self._calculate_overlaps()
        return self._fiberassign_delay


    def _calculate_overlaps(self):
        """Initialize attributes _overlapping and _tile_over.

        Uses the config parameters ``fiber_assignment_order`` and
        ``tile_diameter`` to determine overlap dependencies.

        This is relatively slow, so only used the first time our ``tile_over``
        or ``overlapping`` properties are accessed.
        """
        self._overlapping = {}
        self._tile_over = {}
        self._fiberassign_delay = np.full(self.ntiles, -1, int)
        config = desisurvey.config.Configuration()
        tile_diameter = 2 * config.tile_radius().to(u.deg).value

        # Validate and parse config.fiber_assignment_order into a dictionary.
        fiberassign_order = config.fiber_assignment_order
        Pn = re.compile('^P(\d+)$')
        RHS = re.compile('^(P\d+(?:\+P\d+)*) delay (\d+)$')
        fa_rules = {passnum: dict(coveredby=[], delay=0) for passnum in self.passes}
        for key in fiberassign_order.keys:
            matched = Pn.match(key)
            if not matched:
                raise ValueError('Invalid fiber_assignment_order Pn key: "{}".'.format(key))
            under_pass = int(matched.group(1))
            value = getattr(fiberassign_order, key)()
            matched = RHS.match(value)
            if not matched:
                raise ValueError('Invalid fiber_assignment_order RHS value: "{}".'.format(value))
            over_passes = [int(Pn.match(token).group(1)) for token in matched.group(1).split('+')]
            delay = int(matched.group(2))
            fa_rules[under_pass] = dict(coveredby=over_passes, delay=delay)

        # Initialize the data structures behind the attributes defined above.
        for under_pass in self.passes:
            fa_rule = fa_rules[under_pass]
            under_sel = self.passnum == under_pass
            # Save the delay for tiles in this pass.
            self._fiberassign_delay[under_sel] = fa_rule['delay']
            # Build and save a mask of all tiles in passes that cover this pass.
            over_sel = np.zeros_like(under_sel)
            for over_pass in fa_rule['coveredby']:
                over_sel |= (self.passnum == over_pass)
            self.tile_over[under_pass] = over_sel
            if np.any(over_sel):
                # Calculate a boolean matrix of overlaps between tiles in under_pass
                # and over_passes.
                self.overlapping[under_pass] = desisurvey.utils.separation_matrix(
                    self.tileRA[under_sel], self.tileDEC[under_sel],
                    self.tileRA[over_sel], self.tileDEC[over_sel], tile_diameter)

    def _reduced_bgs_footprint(self, tiles, bgs_footprint): 
        ''' impose reduced footprint for BGS. Depending on the specified number
        of square degrees in bgs_footprint, remove BRIGHT tiles close to the
        ecliptic while keeping NGC a contiguous piece.  
        '''
        from astropy.coordinates import SkyCoord
        assert bgs_footprint < 14000.

        # bgs tiles 
        is_bgs = (tiles['PROGRAM'] == 'BRIGHT')

        # get ecliptic coordinates of tiles 
        tile_coord = SkyCoord(ra=tiles['RA'] * u.deg, dec=tiles['DEC'] * u.deg, frame='icrs')
        ecl_lat = tile_coord.barycentrictrueecliptic.lat.to(u.deg).value

        is_ngc = (tile_coord.galactic.b.value >= 0)
        is_sgc = (tile_coord.galactic.b.value < 0)
    
        # reduced BGS footprint based on specified sq. deg.
        if bgs_footprint == 13000: 
            bgsfoot = (
                    (is_bgs & is_ngc) |
                    (is_bgs & is_sgc & (np.abs(tile_coord.barycentrictrueecliptic.lat.to(u.deg).value) > 6.5)))
        elif bgs_footprint == 12000: 
            bgsfoot = (
                    (is_bgs & is_ngc & (tile_coord.barycentrictrueecliptic.lat.to(u.deg).value > 8.1)) | 
                    (is_bgs & is_sgc))
        elif bgs_footprint == 11000:
            bgsfoot = (
                    (is_bgs & is_ngc & (tile_coord.barycentrictrueecliptic.lat.to(u.deg).value > 8.1)) | 
                    (is_bgs & is_sgc & (np.abs(tile_coord.barycentrictrueecliptic.lat.to(u.deg).value) > 6.5)))
        elif bgs_footprint == 10000: 
            bgsfoot = (
                    (is_bgs & is_ngc & (tile_coord.barycentrictrueecliptic.lat.to(u.deg).value > 12.6)) | 
                    (is_bgs & is_sgc & (np.abs(tile_coord.barycentrictrueecliptic.lat.to(u.deg).value) > 9.7)))
        else: 
            raise NotImplementedError

        # only keep non BRIGHT tiles or BRIGHT tiles far from ecliptic
        reduced_footprint = (~is_bgs) | bgsfoot 

        print('  reduced BGS footprint (v2)')
        print('  %i of %i BRIGHT tiles removed for reduced footprint' %
                (np.sum(is_bgs & ~bgsfoot), np.sum(is_bgs)))
        print('  ~%f sq.deg' % (np.sum(bgsfoot)/np.sum(is_bgs) * 14000.))

        return tiles[reduced_footprint], reduced_footprint 


_cached_tiles = {}

def get_tiles(tiles_file=None, use_cache=True, write_cache=True, bgs_footprint=None):
    """Return a Tiles object with optional caching.

    You should normally always use the default arguments to ensure
    that tiles are defined consistently and efficiently between
    different classes.

    Parameters
    ----------
    tiles_file : str or None
        Use the specified name to override config.tiles_file.
    use_cache : bool
        Use tiles previously cached in memory when True.
        Otherwise, (re)load tiles from disk.
    write_cache : bool
        If tiles need to be loaded from disk with this call,
        save them in a memory cache for future calls.
    """
    global _cached_tiles

    log = desiutil.log.get_logger()
    config = desisurvey.config.Configuration()
    tiles_file = tiles_file or config.tiles_file()

    if use_cache and tiles_file in _cached_tiles:
        tiles = _cached_tiles[tiles_file]
        log.debug('Using cached tiles for "{}".'.format(tiles_file))
        print('  Using cached tiles for "{}".'.format(tiles_file))
        for pname in Tiles.PROGRAMS:
            pinfo = []
            for passnum in tiles.program_passes[pname]:
                pinfo.append('{}({})'.format(passnum, tiles.pass_ntiles[passnum]))
            print('{:6s} passes(tiles): {}.'.format(pname, ', '.join(pinfo)))
    else:
        tiles = Tiles(tiles_file, bgs_footprint=bgs_footprint)
        log.info('Initialized tiles from "{}".'.format(tiles_file))
        print('  Initialized tiles from "{}".'.format(tiles_file))
        for pname in Tiles.PROGRAMS:
            pinfo = []
            for passnum in tiles.program_passes[pname]:
                pinfo.append('{}({})'.format(passnum, tiles.pass_ntiles[passnum]))
            print('{:6s} passes(tiles): {}.'.format(pname, ', '.join(pinfo)))
            log.info('{:6s} passes(tiles): {}.'.format(pname, ', '.join(pinfo)))

    if write_cache:
        _cached_tiles[tiles_file] = tiles
    else:
        log.info('Tiles not cached for "{}".'.format(tiles_file))

    return tiles
