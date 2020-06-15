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
from astropy.time import Time

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
    def __init__(self, tiles_file=None):
        config = desisurvey.config.Configuration()
        # Read the specified tiles file.
        self.tiles_file = tiles_file or config.tiles_file()
        commissioning = getattr(config, 'commissioning', False)
        if not isinstance(commissioning, bool):
            commissioning = commissioning()
        if not commissioning:
            tiles = desimodel.io.load_tiles(
                onlydesi=True, extra=False, tilesfile=self.tiles_file)
        else:
            tiles = desimodel.io.load_tiles(
                onlydesi=False, extra=True, tilesfile=self.tiles_file)
        nogray = getattr(config, 'tiles_nogray', False)
        if not isinstance(nogray, bool):
            nogray = nogray()
        self.nogray = nogray
        if self.nogray:
            m = (tiles['PROGRAM'] == 'GRAY') | (tiles['PROGRAM'] == 'DARK')
            tiles['PROGRAM'][m] = 'DARK'
            obscond = self.OBSCONDITIONS['DARK'] | self.OBSCONDITIONS['GRAY']
            m = (tiles['OBSCONDITIONS'] & obscond) != 0
            tiles['OBSCONDITIONS'][m] |= obscond
        # Copy tile arrays.
        self.tileID = tiles['TILEID'].copy()
        self.tileprogram = tiles['PROGRAM'].copy()
        self.tileRA = tiles['RA'].copy()
        self.tileDEC = tiles['DEC'].copy()
        self.tileobsconditions = tiles['OBSCONDITIONS'].copy()
        self.tileprogram = np.array([p.strip() for p in tiles['PROGRAM']])
        # Count tiles.
        self.ntiles = len(self.tileID)
        # Can remove this when tile_index no longer uses searchsorted.
        if not np.all(np.diff(self.tileID) > 0):
            raise RuntimeError('Tile IDs are not increasing.')
        self.programs = [x for x in np.unique(tiles['PROGRAM'])]
        self.program_index = {pname: pidx
                              for pidx, pname in enumerate(self.programs)}

        # Build tile masks for each program. A program will no tiles with have an empty mask.
        self.program_mask = {}
        for p in self.PROGRAMS:
            self.program_mask[p] = self.tileprogram == p
        self.allowed_in_conditions = {}
        for p in self.OBSCONDITIONS:
            mask = (self.tileobsconditions & self.OBSCONDITIONS[p]) != 0
            self.allowed_in_conditions[p] = mask
        # Calculate and save dust exposure factors.
        self.dust_factor = desisurvey.etc.dust_exposure_factor(tiles['EBV_MED'])
        # Precompute coefficients to calculate tile observing airmass.
        latitude = np.radians(config.location.latitude())
        tile_dec_rad = np.radians(self.tileDEC)
        self.tile_coef_A = np.sin(tile_dec_rad) * np.sin(latitude)
        self.tile_coef_B = np.cos(tile_dec_rad) * np.cos(latitude)
        # Placeholders for overlap attributes that are expensive to calculate
        # so we use lazy evaluation the first time they are accessed.
        self._overlapping = None
        self._fiberassign_delay = None

    CONDITIONS = ['DARK', 'GRAY', 'BRIGHT']
    CONDITION_INDEX = {cond: i for i, cond in enumerate(CONDITIONS)}
    OBSCONDITIONS = {'DARK': 1, 'GRAY': 2, 'BRIGHT': 4}
    """Mapping of night conditions to OBSCONDITIONS bit mask.

    Tiles that may be observed in DARK/GRAY/BRIGHT conditions should have
    (obsconditions & OBSCONDITIONS[program]) != 0.
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

    def airmass_at_mjd(self, mjd, mask=None):
        """Calculate tile airmass at given MJD.

        Parameters
        ----------
        mjd : array
            Array of MJD to use.  If mask is None, then should have length
            ``self.ntiles``.  Otherwise, should have a value per non-zero entry
            in the mask.
        mask : array or None
            Boolean mask of which tiles to perform the calculation for.

        Returns
        -------
        array
            Array of airmasses corresponding to each input hour angle.
        """
        if len(mjd) == 0:
            return np.zeros(0, dtype='f8')
        tt = Time(mjd, format='mjd', location=desisurvey.utils.get_location())
        lst = tt.sidereal_time('apparent').to(u.deg).value

        ha = lst - self.tileRA[mask]
        return self.airmass(ha, mask=mask)

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
        idx = np.clip(idx, 0, len(self.tileID)-1)
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

    def allowed_in_conditions(self, cond):
        return (self.tileobsconditions & self.OBSCONDITIONS[cond]) != 0

    @property
    def overlapping(self):
        """Dictionary of tile overlap matrices.

        overlapping[program][j, k] is True if the j-th tile of program is
        overlapped by the k-th tile of tile_over[program]. There is no
        dictionary entry when the mask tile_over[program] is empty.
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
        """Initialize attributes _overlapping.

        Uses the config parameters ``fiber_assignment_delay`` and
        ``tile_diameter`` to determine overlap dependencies.

        This is relatively slow, so only used the first time ``overlapping``
        properties are accessed.
        """
        self._overlapping = [[] for _ in range(self.ntiles)]
        self._fiberassign_delay = np.full(self.ntiles, -1, int)
        config = desisurvey.config.Configuration()
        tile_diameter = 2 * config.tile_radius()

        fiber_assignment_delay = config.fiber_assignment_delay
        for program in Tiles.PROGRAMS:
            delay = getattr(fiber_assignment_delay, program, None)
            if delay is not None:
                delay = delay()
            else:
                delay = -1
            m = self.program_mask[program]
            rownum = np.flatnonzero(m)
            self._fiberassign_delay[m] = delay
            # self._overlapping: list of lists, giving tiles overlapping each
            # tile
            if delay < 0:
                continue
            from astropy.coordinates import SkyCoord, search_around_sky
            c = SkyCoord(self.tileRA[m]*u.deg, self.tileDEC[m]*u.deg)
            idx1, idx2, sep2d, dist3d = search_around_sky(c, c, tile_diameter)
            for ind1, ind2 in zip(idx1, idx2):
                if ind1 == ind2:
                    # ignore self matches
                    continue
                self._overlapping[rownum[ind1]].append(rownum[ind2])


_cached_tiles = {}

def get_tiles(tiles_file=None, use_cache=True, write_cache=True):
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
    else:
        tiles = Tiles(tiles_file)
        log.info('Initialized tiles from "{}".'.format(tiles_file))
        for pname in tiles.programs:
            log.info('{:6s}: {} tiles'.format(
                pname, np.sum(tiles.program_mask[pname])))

    if write_cache:
        _cached_tiles[tiles_file] = tiles
    else:
        log.info('Tiles not cached for "{}".'.format(tiles_file))

    return tiles


def get_nominal_program_times(tileprogram, config=None):
    """Return nominal times for given programs in seconds."""
    if config is None:
        config = desisurvey.config.Configuration()
    cfgnomtimes = config.nominal_exposure_time
    nomtimes = []
    unknownprograms = []
    nunknown = 0
    for program in tileprogram:
        nomprogramtime = getattr(cfgnomtimes, program, 300)
        if not isinstance(nomprogramtime, int):
            nomprogramtime = nomprogramtime().to(u.s).value
        else:
            unknownprograms.append(program)
            nunknown += 1
        nomtimes.append(nomprogramtime)
    if nunknown > 0:
        log = desiutil.log.get_logger()
        log.info(('%d observations of unknown programs\n' % nunknown) +
                 'unknown programs: '+' '.join(np.unique(unknownprograms)))
    nomtimes = np.array(nomtimes)
    return nomtimes
