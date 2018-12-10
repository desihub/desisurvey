"""Track progress of completed DESI observations.
"""
from __future__ import print_function, division

import os.path

import numpy as np

import astropy.table
import astropy.units as u

import desimodel.io

import desiutil.log

import desisurvey.config
import desisurvey.utils

# Increment this value whenever a non-backwards compatible change to the
# table schema is introduced.
_version = 4

class Progress(object):
    """Initialize a progress tracking object.

    Progress can either be restored from a file or created from scratch.

    The progress table is designed to minmize duplication of static tile data
    that is already tabulated in the footprint definition table, except for
    the PASS, RA, DEC columns which are useful for generating plots.

    The progress table also does not capture ephemeris data that can be
    easily reproduced from an exposure time stamp.

    Parameters
    ----------
    restore : str or astropy.table.Table or None
        Read an existing progress record from the specified file name or
        an exsiting table. A relative path name refers to the
        :meth:`configuration output path
        <desisurvey.config.Configuration.get_path>`. Creates a new progress
        record from sratch when None.
    max_exposures : int
        Maximum number of exposures of a single tile that a newly created
        table will allocate space for.  Ignored when restoring a previous
        progress record.
    """
    def __init__(self, restore=None, max_exposures=32):

        self.log = desiutil.log.get_logger()

        # Lookup the completeness SNR2 threshold to use.
        config = desisurvey.config.Configuration()
        self.min_snr2 = config.min_snr2_fraction()

        if restore is None:
            # Load the list of tiles to observe.
            tiles = astropy.table.Table(
                desimodel.io.load_tiles(onlydesi=True, extra=False,
                    tilesfile=config.tiles_file() ))
            num_tiles = len(tiles)
            # Initialize a new progress table.
            meta = dict(VERSION=_version)
            table = astropy.table.Table(meta=meta)
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
            table['covered'] = astropy.table.Column(
                length=num_tiles, dtype=np.int32,
                description='Tile covered on this day number >=0 (or -1)')
            table['available'] = astropy.table.Column(
                length=num_tiles, dtype=np.int32,
                description='Tile available on this day number >=0 (or -1)')
            table['planned'] = astropy.table.Column(
                length=num_tiles, dtype=np.int32,
                description='Tile first planned on this day number >=0 (or -1)')
            # Add per-exposure columns.
            table['mjd'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.5f',
                description='MJD of exposure start time')
            table['exptime'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.1f',
                description='Exposure duration in seconds', unit='s')
            table['snr2frac'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.3f',
                description='Fraction of target S/N**2 ratio achieved')
            table['airmass'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.1f',
                description='Estimated airmass of observation')
            table['seeing'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.1f',
                description='Estimated FWHM seeing of observation in arcsecs',
                unit='arcsec')
            table['transparency'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.1f',
                description='Estimated transparency of observation')
            table['moonfrac'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.3f',
                description='Moon illuminated fraction (0-1)')
            table['moonalt'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.1f',
                description='Moon altitude angle in degrees', unit='deg')
            table['moonsep'] = astropy.table.Column(
                length=num_tiles, shape=(max_exposures,), format='%.1f',
                description='Moon-tile separation angle in degrees', unit='deg')
            # Copy tile data.
            table['tileid'] = tiles['TILEID']
            table['pass'] = tiles['PASS']
            table['ra'] = tiles['RA']
            table['dec'] = tiles['DEC']
            # Initialize other columns.
            table['status'] = 0
            table['covered'] = -1
            table['available'] = -1
            table['planned'] = -1
            table['mjd'] = 0.
            table['exptime'] = 0.
            table['snr2frac'] = 0.
            table['airmass'] = 0.
            table['seeing'] = 0.
            table['transparency'] = 0.

        else:
            if isinstance(restore, Progress):
                table = restore._table
            elif isinstance(restore, astropy.table.Table):
                table = restore
            else:
                filename = config.get_path(restore)
                if not os.path.exists(filename):
                    raise ValueError('Invalid restore: {0}.'.format(restore))
                table = astropy.table.Table.read(filename)
                self.log.info('Loaded progress from {0}.'.format(filename))
            # Check that this table has the current version.
            if table.meta['VERSION'] != _version:
                raise RuntimeError(
                    'Progress table has incompatible version {0}.'
                    .format(table.meta['VERSION']))
            # Check that the status column matches the current min_snr2.
            snr2sum = table['snr2frac'].data.sum(axis=1)
            if not np.all(snr2sum >= 0):
                raise RuntimeError('Found invalid snr2frac values.')
            status = np.ones_like(table['status'])
            status[snr2sum == 0] = 0
            status[snr2sum >= self.min_snr2] = 2
            if not np.all(table['status'] == status):
                self.log.warn('Updating status values for min(SNR2) = {0:.1f}.'
                              .format(self.min_snr2))
                table['status'] = status
            # We could do more sanity checks here, but they shouldn't be
            # necessary unless the table has been modified outside this class.

        # Initialize attributes from table data.
        self._table = table
        mjd = table['mjd'].data
        observed = mjd > 0
        if np.any(observed):
            self._num_exp = np.count_nonzero(observed)
            self._first_mjd = np.min(mjd[observed])
            self._last_mjd = np.max(mjd[observed])
            last = np.argmax(mjd.max(axis=1))
            self._last_tile = self._table[last]
        else:
            self._num_exp = 0
            self._first_mjd = self._last_mjd = 0.
            self._last_tile = None

    @property
    def num_exp(self):
        """Number of exposures recorded."""
        return self._num_exp

    @property
    def num_tiles(self):
        """Number of tiles in DESI footprint"""
        return len(self._table)

    @property
    def first_mjd(self):
        """MJD of first exposure or 0 if no exposures have been added."""
        return self._first_mjd

    @property
    def last_mjd(self):
        """MJD of most recent exposure or 0 if no exposures have been added."""
        return self._last_mjd

    @property
    def last_tile(self):
        """Row corresponding to the last observed tile, or None."""
        return self._last_tile

    @property
    def max_exposures(self):
        """Maximum allowed number of exposures of a single tile."""
        return len(self._table[0]['mjd'])

    def completed(self, include_partial=True, only_passes=None, as_tuple=False):
        """Number of tiles completed.

        Completion is based on the sum of ``snr2frac`` values for all exposures
        of each tiles.  A completed tile (with ``status`` of 2) counts as one
        towards the completion value, even if its ``snr2frac`` exceeds the
        minimum required SNR**2 fraction.

        Can be combined with :meth:`copy_range` to reconstruct the number of
        completed observations over an arbitrary date range.

        Parameters
        ----------
        include_partial : bool
            Include partially completed tiles according to their sum of snfrac
            values.
        only_passes : tuple or int or None
            Only include tiles in the specified pass or passes.  All passes
            are included when None.
        as_tuple : bool
            Return (num_complete, num_total, percent_complete) as a tuple
            instead of just num_complete.

        Returns
        -------
        float or tuple
            Either num_complete or (num_complete, num_total, percent_complete)
            depending on ``as_tuple``.  The number of tiles completed will
            always be an integer (returned as a float) when ``include_partial``
            is False, and will generally be non-integer otherwise.
        """
        # Restrict to the specified pass(es) if requested.
        if only_passes is not None:
            try:
                only_passes = tuple(only_passes)
            except TypeError:
                only_passes = only_passes,
            sel = np.in1d(self._table['pass'].data, only_passes)
            table = self._table[sel]
        else:
            table = self._table
        # Calculate the total SNR**2 for each tile.
        snr2sum = table['snr2frac'].data.sum(axis=1)
        # Count fully completed tiles as 1.
        completed = snr2sum >= self.min_snr2
        num_complete = float(np.count_nonzero(completed))
        if include_partial:
            # Add partial SNR**2 sums.
            num_complete += snr2sum[~completed].sum()
        if as_tuple:
            num_total = len(table)
            percent_complete = 100. * num_complete / num_total
            return num_complete, num_total, percent_complete
        else:
            return num_complete

    def save(self, filename, overwrite=True):
        """Save the current progress to a file.

        The saved file can be restored from disk using our constructor,
        although column descriptions will be lost since they are not
        propagated when writing a table to a FITS file.

        Parameters
        ----------
        filename : str
            Name of the file where the progress record should be saved. A
            relative path name refers to the :meth:`configuration output path
            <desisurvey.config.Configuration.get_path>`.
        overwrite : bool
            Silently overwrite any existing file when this is True.
        """
        config = desisurvey.config.Configuration()
        filename = config.get_path(filename)
        self._table.write(filename, overwrite=overwrite)
        self.log.info('Saved progress to {0}.'.format(filename))

    def get_tile(self, tile_id):
        """Lookup the progress of a single tile.

        Parameters
        ----------
        tile_id : integer
            Valid DESI footprint tile ID.

        Returns
        -------
        astropy.table.Row
            Row of progress table for the requested tile.
        """
        row_sel = np.where(self._table['tileid'] == tile_id)[0]
        if len(row_sel) != 1:
            raise ValueError('Invalid tile_id {0}.'.format(tile_id))
        return self._table[row_sel[0]]

    def get_summary(self, include='observed'):
        """Get a per-tile summary of progress so far.

        Returns a new table so any modifications are decoupled from our
        internal table.  Exposure MJD values are summarized as separate
        ``mjd_min`` and ``mjd_max`` columns, with both equal to zero for
        un-observed tiles. The summary ``exptime`` and ``snr2frac`` columns
        are sums of the individual exposures.  The summary ``airmass``,
        ``seeing`` and ``transparency`` columns are means. A ``nexp`` column
        counts the number of exposures for each tile.  The moon parameters are
        not summarized.

        Can be combined with :meth:`copy_range` to summarize observations during
        a range of dates.

        Parameters
        ----------
        include : 'all', 'observed', or 'completed'
            Specify which tiles to include in the summary. The 'observed'
            selection will include tiles that have been observed at least
            once but have not yet reached their SNR**2 goal.
        """
        min_status = dict(all=0, observed=1, completed=2)
        if include not in min_status.keys():
            raise ValueError('Invalid include option: pick one of {0}.'
                             .format(', '.join(min_status.keys())))

        # Start a new summary table with the selected rows.
        sel = self._table['status'] >= min_status[include]
        summary = self._table[sel][[
            'tileid', 'pass', 'ra', 'dec', 'status',
            'covered', 'available', 'planned']]

        # Summarize exposure start times.
        col = self._table['mjd']
        mjd = col.data[sel]
        summary['mjd_min'] = astropy.table.Column(
            mjd[:, 0], unit=col.unit, format=col.format,
            description='First exposure start MJD')
        summary['mjd_max'] = astropy.table.Column(
            mjd.max(axis=1), unit=col.unit, format=col.format,
            description='Last exposure start MJD')

        # Sum the remaining per-exposure columns.
        for name in (
            'exptime', 'snr2frac', 'airmass', 'seeing', 'transparency'):
            col = self._table[name]
            summary[name] = astropy.table.Column(
                col.data[sel].sum(axis=1), unit=col.unit, format=col.format,
                description=col.description)

        # Convert the airmass, seeing and transparency sums to means.  We use
        # mean rather than median since it is easier to calculate with a
        # variable nexp.
        nexp = (mjd > 0).sum(axis=1).astype(int)
        mask = nexp > 0
        summary['airmass'][mask] /= nexp[mask]
        summary['seeing'][mask] /= nexp[mask]
        summary['transparency'][mask] /= nexp[mask]

        # Record the number of exposures in a new column.
        summary['nexp'] = nexp

        return summary

    def copy_range(self, mjd_min=None, mjd_max=None):
        """Return a copy of progress during a date range.

        Parameters
        ----------
        mjd_min : float or None
            Only include exposures with mjd >= mjd_min.
        mjd_max : float
            Only include exposures with mjd < mjd_max.

        Returns
        -------
        Progress
            New object with any exposures outside the specified MJD range
            zeroed out and ``status`` values updated accordingly.
        """
        if mjd_min and mjd_max and mjd_min >= mjd_max:
            raise ValueError('Expected mjd_min < mjd_max.')
        # Identify which exposures to drop.
        mjd = self._table['mjd'].data
        drop = (mjd == 0)
        if mjd_min is not None:
            drop |= (mjd < mjd_min)
        if mjd_max is not None:
            drop |= (mjd >= mjd_max)
        # Copy our table.
        table = self._table.copy()
        # Zero dropped exposures.
        for name in (
            'mjd', 'exptime', 'snr2frac', 'airmass', 'seeing', 'transparency'):
            table[name][drop] = 0.
        # Recompute the status column.
        snr2sum = table['snr2frac'].data.sum(axis=1)
        assert np.all(snr2sum >= 0)
        table['status'] = 1
        table['status'][snr2sum == 0] = 0
        table['status'][snr2sum >= self.min_snr2] = 2
        if mjd_max is not None:
            # Rewind the covered and available columns.
            config = desisurvey.config.Configuration()
            max_day_number = desisurvey.utils.day_number(mjd_max)
            table['covered'][table['covered'] > max_day_number] = -1
            table['available'][table['available'] > max_day_number] = -1
            table['planned'][table['planned'] > max_day_number] = -1
        # Return a new progress object with this table.
        return Progress(restore=table)

    def add_exposure(self, tile_id, start, exptime, snr2frac, airmass, seeing,
                     transparency, moonfrac, moonalt, moonsep):
        """Add a single exposure to the progress.

        Parameters
        ----------
        tile_id : int
            DESI footprint tile ID
        start : astropy.time.Time
            Exposure start time.  Must be after any previous exposure.
        exptime : astropy.units.Quantity
            Exposure open shutter time with units.
        snr2frac : float
            Fraction of the design SNR**2 achieved during this exposure.
        airmass : float
            Estimated airmass of this exposure.
        seeing : float
            Estimated FWHM seeing of this exposure in arcseconds.
        transparency : float
            Estimated atmospheric transparency of this exposure.
        moonfrac : float
            Moon illuminated fraction (0-1).
        moonalt : float
            Moon altitude angle in degrees.
        moonsep : float
            Moon-tile separation angle in degrees.
        """
        mjd = start.mjd
        self.log.info(
            'Adding {0:.1f} exposure #{1:06d} of {2} at {3} (MJD {4:.5f}).'
            .format(exptime, self.num_exp, tile_id, start.datetime, mjd))
        row = self.get_tile(tile_id)

        # Check that we have not reached the maximum allowed exposures.
        num_exp = np.count_nonzero(row['mjd'] > 0)
        if num_exp == self.max_exposures:
            raise RuntimeError(
                'Reached maximum exposure limit ({0}) for tile_id {1}.'
                .format(self.max_exposures, tile_id))

        # Check for increasing timestamps.
        if mjd <= self._last_mjd:
            raise ValueError('Exposure MJD {0:.5f} <= last MJD {1:.5f}.'
                             .format(mjd, self._last_mjd))

        # Remember the most recent exposure.
        self._last_mjd = mjd
        self._last_tile = row
        self._num_exp += 1

        # Remember the first exposure's timestamp.
        if self._first_mjd == 0:
            self._first_mjd = mjd

        # Save this exposure.
        row['mjd'][num_exp] = mjd
        row['exptime'][num_exp] = exptime.to(u.s).value
        row['snr2frac'][num_exp] = snr2frac
        row['airmass'][num_exp] = airmass
        row['seeing'][num_exp] = seeing
        row['transparency'][num_exp] = transparency
        row['moonfrac'][num_exp] = moonfrac
        row['moonalt'][num_exp] = moonalt
        row['moonsep'][num_exp] = moonsep

        # Update this tile's status.
        row['status'] = 1 if row['snr2frac'].sum() < self.min_snr2 else 2

    def get_exposures(self, start=None, stop=None,
                      tile_fields='tileid,pass,ra,dec,ebmv',
                      exp_fields=('night,mjd,exptime,seeing,transparency,' +
                                  'airmass,moonfrac,moonalt,moonsep,' +
                                  'program,flavor')):
        """Create a table listing exposures in time order.

        Parameters
        ----------
        start : date or None
            First date to include in the list of exposures, or date of the
            first observation if None.
        stop  : date or None
            Last date to include in the list of exposures, or date of the
            last observation if None.
        tile_fields : str
            Comma-separated list of per-tile field names to include. The
            special name 'index' denotes the index into the visible tile array.
            The special name 'ebmv' adds median E(B-V) values for each tile
            from the tile design file.
        exp_fields : str
            Comma-separated list of per-exposure field names to include. The
            special name 'snr2cum' denotes the cummulative snr2frac on each
            tile, since the start of the survey.  The special name 'night'
            denotes a string YYYYMMDD specifying the date on which each
            night starts. The special name 'lst' denotes the apparent local
            sidereal time of the shutter open timestamp. The special name
            'expid' denotes the index of each exposure in the full progress
            record starting from zero.

        Returns
        -------
        astropy.table.Table
            Table with the specified columns as uppercase and one row per exposure.
        """
        # Get MJD range to show.
        if start is None:
            start = self.first_mjd
        start = desisurvey.utils.local_noon_on_date(
            desisurvey.utils.get_date(start)).mjd
        if stop is None:
            stop = self.last_mjd
        stop = desisurvey.utils.local_noon_on_date(
            desisurvey.utils.get_date(stop)).mjd + 1
        if start >= stop:
            raise ValueError('Expected start < stop.')

        # Build a list of exposures in time sequence.
        table = self._table
        mjd = table['mjd'].data.flatten()
        order = np.argsort(mjd)
        tile_index = (order // self.max_exposures)

        # Assign each exposure a sequential index starting from zero.
        ntot = len(mjd)
        nexp = np.count_nonzero(mjd > 0)
        expid = np.empty(ntot, int)
        expid[order] = np.arange(nexp - ntot, nexp)

        # Restrict to the requested date range.
        first, last = np.searchsorted(mjd, [start, stop], sorter=order)
        tile_index = tile_index[first:last + 1]
        order = order[first:last + 1]
        assert np.all(expid[order] >= 0)

        # Create the output table.
        tileinfo = None
        output = astropy.table.Table()
        output.meta['EXTNAME'] = 'EXPOSURES'
        for name in tile_fields.split(','):
            name = name.lower()
            if name == 'index':
                output[name.upper()] = tile_index
            elif name == 'ebmv':
                if tileinfo is None:
                    config = desisurvey.config.Configuration()
                    tileinfo = astropy.table.Table(
                        desimodel.io.load_tiles(onlydesi=True, extra=False,
                        tilesfile=config.tiles_file()))
                    assert np.all(tileinfo['TILEID'] == table['tileid'])
                output[name.upper()] = tileinfo['EBV_MED'][tile_index]
            else:
                if name not in table.colnames or len(table[name].shape) != 1:
                    raise ValueError(
                        'Invalid tile field name: {0}.'.format(name))
                output[name.upper()] = table[name][tile_index]
        for name in exp_fields.split(','):
            name = name.lower()
            if name == 'snr2cum':
                snr2cum = np.cumsum(
                    table['snr2frac'], axis=1).flatten()[order]
                output[name.upper()] = astropy.table.Column(
                    snr2cum, format='%.3f',
                    description='Cummulative fraction of target S/N**2')
            elif name == 'night':
                mjd = table['mjd'].flatten()[order]
                night = np.empty(len(mjd), dtype=(str, 8))
                for i in range(len(mjd)):
                    night[i] = str(desisurvey.utils.get_date(mjd[i])).replace('-', '')
                output[name.upper()] = astropy.table.Column(
                    night,
                    description='Date at start of night when exposure taken')
            elif name == 'lst':
                mjd = table['mjd'].flatten()[order]
                times = astropy.time.Time(
                    mjd, format='mjd', location=desisurvey.utils.get_location())
                lst = times.sidereal_time('apparent').to(u.deg).value
                output[name.upper()] = astropy.table.Column(
                    lst, format='%.1f', unit='deg',
                    description='Apparent local sidereal time in degrees')
            elif name == 'program':
                exppass = table['pass'][tile_index]
                try:
                    from desimodel.footprint import pass2program
                    program = pass2program(exppass)
                except ImportError:
                    #- desimodel < 0.9.1 doesn't have pass2program, so
                    #- hardcode the mapping that it did have
                    program = np.empty(len(exppass), dtype=(str, 6))
                    program[:] = 'BRIGHT'
                    program[exppass < 4] = 'DARK'
                    program[exppass == 4] = 'GRAY'
                    
                proglen = len(max(program, key=len))
                if proglen < 6: # need at least six characters for 'CALIB' program
                    proglen = 6

                output[name.upper()] = astropy.table.Column(program,
                                                            dtype='<U{}'.format(proglen),
                                                            description='Program name')
            elif name == 'flavor':
                flavor = np.empty(len(exppass), dtype=(str, 7))
                flavor[:] = 'science'
                output[name.upper()] = astropy.table.Column(flavor,
                    description='Exposure flavor')
            elif name == 'expid':
                output[name.upper()] = astropy.table.Column(
                    expid[order], description='Exposure index')
            else:
                if name not in table.colnames or len(table[name].shape) != 2:
                    raise ValueError(
                        'Invalid exposure field name: {0}.'.format(name))
                output[name.upper()] = table[name].flatten()[order]

        return output
