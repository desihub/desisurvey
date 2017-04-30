"""Track progress of completed DESI observations.
"""
from __future__ import print_function, division


import numpy as np

import astropy.table

import desimodel.io

import desisurvey.config

# Increment this value whenever a non-backwards compatible change to the
# table schema is introduced.
_version = 1

class Progress(object):
    """Initialize a progress tracking object.

    The tracker can either be loaded from a file or created from scratch.

    The progress table is designed to minmize duplication of static tile data
    that is already tabulated in the footprint definition table, except for
    the PASS, RA, DEC columns which are useful for generating plots.

    The progress table also does not capture ephemeris data that can be
    easily reproduced from an exposure time stamp.

    Parameters
    ----------
    filename : str or None
        Read an existing progress record from the specified file name. A
        relative path name refers to the :meth:`configuration output path
        <desisurvey.config.Configuration.get_path>`. Creates a new progress
        record from sratch when None.
    max_exposures : int
        Maximum number of exposures of a single tile that a newly created
        table will allocate space for.  Ignored when a previous file name
        is being read.
    """
    def __init__(self, filename=None, max_exposures=16):

        if filename is None:
            # Load the list of tiles to observe.
            tiles = astropy.table.Table(
                desimodel.io.load_tiles(onlydesi=True, extra=False))
            num_tiles = len(tiles)
            # Initialize a new progress table.
            meta = dict(VERSION=_version)
            table = astropy.table.Table()
            table['tileid'] = astropy.table.Column(
                length=num_tiles, dtype=np.int32,
                description='DESI footprint tile ID')
            table['pass'] = astropy.table.Column(
                length=num_tiles, dtype=np.int32,
                description='Observing pass number starting at zero')
            table['ra'] = astropy.table.Column(
                length=num_tiles, description='TILE center RA in degrees',
                unit='deg', format='%.1f')
            table['dec'] = astropy.table.Column(
                length=num_tiles, description='TILE center DEC in degrees',
                unit='deg', format='%.1f')
            table['status'] = astropy.table.Column(
                length=num_tiles, dtype=np.int32,
                description='Observing status: 0=none, 1=partial, 2=done')
            # Add per-exposure columns.
            table['mjd'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.5f',
                description='MJD of exposure start time')
            table['exptime'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.1f',
                description='Exposure duration in seconds', unit='s')
            table['snrfrac'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.3f',
                description='Fraction of target S/N ratio achieved')
            table['airmass'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.1f',
                description='Estimate airmass of observation')
            table['seeing'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.1f',
                description='Estimate FWHM seeing of observation in arcsecs',
                unit='arcsec')
            # Copy tile data.
            table['tileid'] = tiles['TILEID']
            table['pass'] = tiles['PASS']
            table['ra'] = tiles['RA']
            table['dec'] = tiles['DEC']
            # Initialize other columns.
            table['status'] = 0
            table['mjd'] = 0.
            table['exptime'] = 0.
            table['snrfrac'] = 0.
            table['airmass'] = 0.
            table['seeing'] = 0.

        else:
            config = desisurvey.config.Configuration()
            table = astropy.table.Table.read(config.get_path(filename))
            # Check that this table has the current version.
            if table.meta['VERSION'] != _version:
                raise RuntimeError(
                    'Progress table has incompatible version {0}.'
                    .format(table.meta['VERSION']))

        # Initialize attributes from table data.
        self._table = table

    @property
    def max_exposures(self):
        """Maximum allowed number of exposures of a single tile."""
        return len(self._table[0]['mjd'])

    def save(self, filename='progress.fits', overwrite=True):
        """Save the current progress to a file.

        The saved file can be restored from disk using our constructor.

        Parameters
        ----------
        filename : str or None
            Read an existing progress record from the specified file name. A
            relative path name refers to the :meth:`configuration output path
            <desisurvey.config.Configuration.get_path>`. Creates a new progress
            record from sratch when None.
        overwrite : bool
            Silently overwrite any existing file when this is True.
        """
        config = desisurvey.config.Configuration()
        self._table.write(config.get_path(filename), overwrite=overwrite)

    def add_exposure(self, tile_id, mjd, exptime, snrfrac, airmass, seeing):
        """
        """
        # Lookup the row for this tile.
        row_sel = np.where(self._table['tileid'] == tile_id)
        if not row_sel:
            raise ValueError('Invalid tile_id {0}.'.format(tile_id))
        row = self._table[row_sel[0]]
        assert len(row) == self.max_exposures

        # Check that we have not reached the maximum allowed exposures.
        num_exp = np.count_nonzero(row['mjd'] > 0)
        if num_exp == self.max_exposures:
            raise RuntimeError(
                'Reached maximum exposure limit ({0}) for tile_id {1}.'
                .format(self.max_exposures, tile_id))

        # Save this exposure.
        row['mjd'][num_exp] = mjd
        row['exptime'][num_exp] = exptime
        row['snrfrac'][num_exp] = snrfrac
        row['airmass'][num_exp] = airmass
        row['seeing'][num_exp] = seeing

        # Update this tile's status.
        row['status'] = 1 if row['snrfrac'].sum() < 1 else 2
