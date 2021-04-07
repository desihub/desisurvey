"""Manage static information associated with tiles, programs and passes.

Each tile has an assigned program name. The program names
(DARK, BRIGHT) are predefined in terms of conditions on the
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

import os

import numpy as np

import astropy.units as u
from astropy.time import Time
from astropy.table import Table

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
        self.nogray = config.tiles_nogray()

        # Read the specified tiles file.
        self.tiles_file = tiles_file or config.tiles_file()
        self.tiles_file = find_tile_file(self.tiles_file)

        tiles = self.read_tiles_table()

        # Copy tile arrays.
        self.tileID = tiles['TILEID'].data.copy()
        self.tileRA = tiles['RA'].data.copy()
        self.tileDEC = tiles['DEC'].data.copy()
        self.tileprogram = np.array([p.strip() for p in tiles['PROGRAM']])
        self.tilepass = tiles['PASS'].data.copy()
        self.designha = None
        if 'DESIGNHA' in tiles.dtype.names:
            self.designha = tiles['DESIGNHA'].data.copy()

        self.tileobsconditions = self.get_conditions()
        if self.nogray:
            mgray = self.tileobsconditions == 'GRAY'
            self.tileobsconditions[mgray] = 'DARK'

        self.in_desi = tiles['IN_DESI'].data.copy() != 0

        # Count tiles.
        self.ntiles = len(self.tileID)
        # Can remove this when tile_index no longer uses searchsorted.
        if not np.all(np.diff(self.tileID) > 0):
            raise RuntimeError('Tile IDs are not increasing.')
        self.programs = [x for x in np.unique(tiles['PROGRAM'].data)]
        self.program_index = {pname: pidx
                              for pidx, pname in enumerate(self.programs)}

        # Build tile masks for each program. A program will no tiles with have an empty mask.
        self.program_mask = {}
        for p in self.programs:
            self.program_mask[p] = self.tileprogram == p
        # Calculate and save dust exposure factors.
        self.dust_factor = desisurvey.etc.dust_exposure_factor(tiles['EBV_MED'].data)
        # Precompute coefficients to calculate tile observing airmass.
        latitude = np.radians(config.location.latitude())
        tile_dec_rad = np.radians(self.tileDEC)
        self.tile_coef_A = np.sin(tile_dec_rad) * np.sin(latitude)
        self.tile_coef_B = np.cos(tile_dec_rad) * np.cos(latitude)
        # Placeholders for overlap attributes that are expensive to calculate
        # so we use lazy evaluation the first time they are accessed.
        self._overlapping = None
        self._fiberassign_delay = None

        # Calculate the maximum |HA| in degrees allowed for each tile to stay
        # above the survey minimum altitude
        cosZ_min = np.cos(90 * u.deg - config.min_altitude())
        cosHA_min = (
            (cosZ_min - np.sin(self.tileDEC * u.deg) * np.sin(latitude)) /
            (np.cos(self.tileDEC * u.deg) * np.cos(latitude))).value
        cosHA_min = np.clip(cosHA_min, -1, 1)
        self.max_abs_ha = np.degrees(np.arccos(cosHA_min))
        m = ~np.isfinite(self.max_abs_ha) | (self.max_abs_ha < 3.75)
        self.max_abs_ha[m] = 7.5  # always give at least a half hour window.

    CONDITIONS = ['DARK', 'GRAY', 'BRIGHT']
    CONDITION_INDEX = {cond: i for i, cond in enumerate(CONDITIONS)}

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
        mjd = np.atleast_1d(mjd)
        if len(mjd) == 0:
            return np.zeros(0, dtype='f8')
        tt = Time(mjd, format='mjd', location=desisurvey.utils.get_location())
        lst = tt.sidereal_time('apparent').to(u.deg).value

        ha = lst - self.tileRA[mask]
        return self.airmass(ha, mask=mask)

    def airmass_second_derivative(self, HA, mask=None):
        """Calculate second derivative of airmass with HA.

        Useful for determining how close to design airmass we have to get
        for different tiles.  When this is large, we really need to observe
        things right at their design angles.  When it's small, we have more
        flexibility.
        """
        x = self.airmass(HA, mask=mask)
        if mask is not None:
            b = self.tile_coef_B[mask]
        else:
            b = self.tile_coef_B
        d2rad = b*x**2 * (2*b*x*np.sin(np.radians(HA))**2 +
                          np.cos(np.radians(HA)))
        return d2rad * (np.pi/180)**2

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
        if np.any(tileID < 0):
            raise ValueError('tileIDs must positive!')
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

    def get_conditions(self):
        res = []
        config = desisurvey.config.Configuration()
        for program in self.tileprogram:
            tprogram = getattr(config.programs, program, None)
            if tprogram is None:
                res.append('NONE')
            else:
                res.append(tprogram.conditions())
        return np.array(res)

    def allowed_in_conditions(self, cond):
        if self.nogray and (cond == 'GRAY'):
            cond = 'DARK'
        return (self.tileobsconditions == cond)

    @property
    def overlapping(self):
        """Dictionary of tile overlap matrices.

        overlapping[i] is the list of tile row numbers that overlap the
        tile with row number i.

        Overlapping tiles are only computed within a program; a tile cannot
        overlap a tile of a different program.  If fiber_assignment_delay is
        negative, tile do not overlap one another within a program.
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
        for program in self.programs:
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
                # this program doesn't have overlapping tile requirements
                continue
            from astropy.coordinates import SkyCoord, search_around_sky
            c = SkyCoord(self.tileRA[m]*u.deg, self.tileDEC[m]*u.deg)
            idx1, idx2, sep2d, dist3d = search_around_sky(c, c, tile_diameter)
            for ind1, ind2 in zip(idx1, idx2):
                if ind1 == ind2:
                    # ignore self matches
                    continue
                self._overlapping[rownum[ind1]].append(rownum[ind2])

    def read_tiles_table(self):
        """Read and trim the tiles table.

        Must be called after self.tiles_file and self.nogray
        member variables have been set.
        """
        config = desisurvey.config.Configuration()
        tiles = Table.read(self.tiles_file)
        # control truncation
        tileprogram = np.zeros(len(tiles), dtype='U20')
        tileprogram[:] = tiles['PROGRAM']
        # convert dark/gray/bright to canonical DARK/GRAY/BRIGHT
        progupper = np.array([t.upper().strip() for t in tileprogram])
        for p in ['DARK', 'BRIGHT', 'GRAY', 'BACKUP']:
            m = progupper == p
            tileprogram[m] = p
        if self.nogray:
            m = (tileprogram == 'GRAY') | (tileprogram == 'DARK')
            tileprogram[m] = 'DARK'
        tiles['PROGRAM'] = tileprogram
        trim = config.tiles_trim()
        if trim:
            tiles = tiles[tiles['IN_DESI'] != 0]
            tprograms = np.unique(tiles['PROGRAM'])
            programinconfig = np.isin(tprograms,
                                      [x for x in config.programs.keys])
            log = desiutil.log.get_logger()
            keep = np.ones(len(tiles), dtype='bool')
            if np.any(~programinconfig):
                for program in tprograms[~programinconfig]:
                    keep[tiles['PROGRAM'] == program] = 0
                log.info('Removing the following programs from the tile '
                         'file: ' + ' '.join(tprograms[~programinconfig]))
            tiles = tiles[keep]
        if not np.all(np.diff(tiles['TILEID']) > 0):
            tiles['TILEID'] = np.arange(len(tiles))
            log = desiutil.log.get_logger()
            log.warning('Tile file TILEID are not ascending; rewriting.')
        return tiles


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
            log.debug('{:6s}: {} tiles'.format(
                pname, np.sum(tiles.program_mask[pname])))

    if write_cache:
        _cached_tiles[tiles_file] = tiles
    else:
        log.info('Tiles not cached for "{}".'.format(tiles_file))

    return tiles


def find_tile_file(filename):
    log = desiutil.log.get_logger()
    if os.path.isabs(filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(
                'tile file not found at {}'.format(filename))
        return filename
    dmname = desimodel.io.findfile(os.path.join('footprint', filename))
    dsdirname = os.environ.get('DESISURVEY_OUTPUT', None)
    if dsdirname is not None:
        dsname = os.path.join(dsdirname, filename)
    else:
        dsname = ''
    localname = filename
    namedict = dict(DESISURVEY=(dsname, os.path.exists(dsname)),
                    DESIMODEL=(dmname, os.path.exists(dmname)),
                    LOCAL=(localname, os.path.exists(localname)))
    for key in namedict:
        if namedict[key][1]:
            fn = namedict.pop(key)[0]
            others = [key for (key, (name, exists)) in namedict.items()
                      if exists]
            if len(others) > 0:
                log.info('Using {} filename, '.format(fn) +
                         'ignoring other files of same name: ' +
                         ' '.join(others))
            return fn
    raise FileNotFoundError('tile file not found at {}'.format(filename))


def get_nominal_program_times(tileprogram, config=None,
                              return_timetypes=False):
    """Return nominal times for given programs in seconds."""
    if config is None:
        config = desisurvey.config.Configuration()
    progconf = config.programs
    nomtimes = []
    timetypes = []
    unknownprograms = []
    nunknown = 0
    for program in tileprogram:
        tprogconf = getattr(progconf, program, None)
        if tprogconf is None:
            nomprogramtime = 300
            unknownprograms.append(program)
            nunknown += 1
            timetype = 'ELG'
        else:
            nomprogramtime = getattr(tprogconf, 'efftime')()
            timetype = getattr(tprogconf, 'efftime_type')()
        if not isinstance(nomprogramtime, int):
            nomprogramtime = nomprogramtime.to(u.s).value
        nomtimes.append(nomprogramtime)
        timetypes.append(timetype)
    if nunknown > 0:
        log = desiutil.log.get_logger()
        log.debug(('%d observations of unknown programs:' % nunknown) +
                  ' '.join(np.unique(unknownprograms)))
    nomtimes = np.array(nomtimes)
    timetypes = np.array(timetypes)
    ret = nomtimes
    if return_timetypes:
        ret = (ret, timetypes)
    return ret
