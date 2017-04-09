from __future__ import print_function, division

import copy
import os
import sys
import pkg_resources

import numpy as np

import astropy.table

import desimodel.io
import desiutil.log

from desisurvey.utils import mjd2lst


class surveyPlan:
    """
    Main class for survey planning
    """

    def __init__(self, MJDstart, MJDend, tilesubset=None):
        """Initialises survey by reading in the file desi_tiles.fits
        and populates the class members.

        Arguments:
            MJDstart: day of the (re-)start of the survey
            MJDend: day of the end of the survey
            surveycal: list of dictionnaries with times of sunset, twilight, etc

        Optional:
            tilesubset: array of integer tileids to use; ignore others
        """
        self.log = desiutil.log.get_logger()

        # Read in DESI tile data
        tiles = astropy.table.Table(
            desimodel.io.load_tiles(onlydesi=True, extra=False))

        # Restrict to a subset of tiles if requested.
        if tilesubset is not None:
            tiles = tiles[tilesubset]

        numtiles = len(tiles)

        # Drop un-needed columns.
        tiles.remove_columns(['IN_DESI', 'AIRMASS', 'STAR_DENSITY', 'EXPOSEFAC'])

        # Add some new columns (more will be added later).
        for name, dtype in (('GAL_CAP', np.int8), ('SUBLIST', np.int8),
                            ('PRIORITY', np.int32), ('STATUS', np.int32)):
            tiles[name] = np.zeros(numtiles, dtype=dtype)

        # Determine which galactic cap each tile is in: -1=south, +1=north.
        tiles['GAL_CAP'] = -1
        tiles['GAL_CAP'][(tiles['RA'] > 75) & (tiles['RA'] < 300)] = +1

        # Assign a sublist to each tile equal to pass for tiles in the
        # first-year full depth field, or else equal to pass+8.  The full
        # depth field covers 15 deg <= dec <= 25 deg in the NGC,
        # padded by 3 deg for the first pass in each program.
        dec = tiles['DEC']
        passnum = tiles['PASS']
        first_pass = (passnum == 0) | (passnum == 4) | (passnum == 5)
        dec_min = np.full(numtiles, 15.)
        dec_max = np.full(numtiles, 25.)
        dec_min[first_pass] -= 3.
        dec_max[first_pass] += 3.
        tiles['SUBLIST'] = passnum
        tiles['SUBLIST'][
            (tiles['GAL_CAP'] < 0) | (dec < dec_min) | (dec > dec_max)] += 8

        # Initialize the LST bins we will use for scheduling each night.
        self.nLST = 144
        self.LSTedges = np.linspace(0., 360., self.nLST + 1)
        self.LSTbins = 0.5 * (self.LSTedges[1:] + self.LSTedges[:-1])
        self.LSTres = self.LSTedges[1]

        self.tiles = tiles
        self.numtiles = numtiles

        # Add HA, LSTMIN, LSTMAX columns.
        self.assignHA(MJDstart, MJDend)

        self.tiles.sort(('SUBLIST', 'DEC'))


    def assignHA(self, MJDstart, MJDend, compute=False):
        """Assigns optimal hour angles for the DESI tiles;
        can be re-run at any point during the survey to
        reoptimise the schedule.
        """

        if compute:
            raise NotImplementedError('assignHA(compute=True)')
        else:
            # Read in the pre-computed HA and begin/end LST range.
            info = astropy.table.Table.read(
                pkg_resources.resource_filename(
                    'desisurvey', 'data/tile-info.fits'), hdu=1)
            # Ignore most of the columns.
            info = info[['TILEID', 'HA', 'BEGINOBS', 'ENDOBS', 'OBSTIME']]
            # Join with our tiles table, matching on TILEID.
            self.tiles = astropy.table.join(
                self.tiles, info, keys='TILEID', join_type='left')
            if len(self.tiles) != self.numtiles:
                raise RuntimeError('Missing some tiles in tile-info.fits')
            # Rename new columns.
            self.tiles.rename_column('BEGINOBS', 'LSTMIN')
            self.tiles.rename_column('ENDOBS', 'LSTMAX')
            self.tiles.rename_column('OBSTIME', 'EXPLEN')


    def afternoonPlan(self, day_stats, date_string, tiles_observed):
        """Main decision making method

        Args:
            day_stats: row of tabulated ephmerides data for today
            date_string: string of the form YYYYMMDD
            tiles_observed: table with follwing columns: tileID, status

        Returns:
            string containg the filename for today's plan; it has the format
            obsplanYYYYMMDD.fits
        """
        nto = len(tiles_observed)

        # Copy the STATUS for previously observed tiles.
        if nto > 0:
            for status in set(tiles_observed['STATUS']):
                ii = (tiles_observed['STATUS'] == status)
                jj = np.in1d(self.tiles['TILEID'], tiles_observed['TILEID'][ii])
                self.tiles['STATUS'][jj] = status

        # Find all tiles with STATUS < 2
        finalTileList = self.tiles[self.tiles['STATUS'] < 2]

        # Assign tiles to LST bins
        planList0 = []
        lst15evening = mjd2lst(day_stats['MJDetwi'])
        lst15morning = mjd2lst(day_stats['MJDmtwi'])
        lst13evening = mjd2lst(day_stats['MJDe13twi'])
        lst13morning = mjd2lst(day_stats['MJDm13twi'])
        LSTmoonrise = mjd2lst(day_stats['MJDmoonrise'])
        LSTmoonset = mjd2lst(day_stats['MJDmoonset'])
        LSTbrightstart = mjd2lst(day_stats['MJD_bright_start'])
        LSTbrightend = mjd2lst(day_stats['MJD_bright_end'])

        # Calculate LST of each tile in the range [0, 360).
        finalTileLST = finalTileList['RA'] + finalTileList['HA']
        assert np.min(finalTileLST) > -360.
        finalTileLST = np.fmod(finalTileLST + 360., 360.)

        # Select tiles assigned to each program.  The original code tested
        # for bits in OBSCONDITIONS but this is equivalent and faster.
        dark_tile = finalTileList['PROGRAM'] == 'DARK'
        gray_tile = finalTileList['PROGRAM'] == 'GRAY'
        bright_tile = finalTileList['PROGRAM'] == 'BRIGHT'

        # Check that each tile is assigned to exactly one program.
        assert np.all(dark_tile.astype(int) + gray_tile + bright_tile == 1)

        # Assign each tile to an LST bin.
        finalTileLSTbin = np.digitize(finalTileLST, self.LSTedges) - 1
        assert np.all(finalTileLSTbin >= 0)

        # Assign the program for each LST bin tonight.
        def inLSTWindow(start, stop):
            if start <= stop:
                return (self.LSTbins > start) & (self.LSTbins < stop)
            else:
                return (self.LSTbins < stop) | (self.LSTbins > start)

        night13 = inLSTWindow(lst13evening, lst13morning)
        night15 = inLSTWindow(lst15evening, lst15morning)
        moon_up = inLSTWindow(LSTmoonrise, LSTmoonset)
        bright = inLSTWindow(LSTbrightstart, LSTbrightend)
        dark =  night15 & ~moon_up
        gray = night15 & moon_up & ~bright

        # Add the time between 13 and 15 degree twilight to the BRIGHT program.
        bright |= night13 & ~night15

        # Check that each bin is assigned to at most one program.
        assert np.max(dark.astype(int) + bright + gray) == 1

        # Loop over LST bins
        for i in range(self.nLST):
            scheduled = []
            # DARK time
            if dark[i]:
                # Find all DARK tiles in this LST bin with STATUS < 2.
                found = np.where(dark_tile & (finalTileLSTbin == i) &
                                 (finalTileList['STATUS'] < 2))[0]
                # Schedule the first 5.
                scheduled.extend(found[:5])
                # If fewer than 5 dark tiles fall within this window, pad with grey
                if len(scheduled) < 5:
                    found = np.where(gray_tile & (finalTileLSTbin == i) &
                                     (finalTileList['STATUS'] < 2))[0]
                    scheduled.extend(found[:5 - len(scheduled)])
                # If fewer than 5 dark or grey tiles fall within this window,
                # pad with bright tiles.
                if len(scheduled) < 5:
                    found = np.where(bright_tile & (finalTileLSTbin == i) &
                                     (finalTileList['STATUS'] < 2))[0]
                    scheduled.extend(found[:5 - len(scheduled)])
            # GRAY time
            if gray[i]:
                # Find all GRAY tiles in this LST bin with STATUS < 2.
                found = np.where(gray_tile & (finalTileLSTbin == i) &
                                 (finalTileList['STATUS'] < 2))[0]
                # Schedule the first 5.
                scheduled.extend(found[:5])
                # If fewer than 5 grey tiles fall within this window, pad with
                # bright tiles.
                if len(scheduled) < 5:
                    found = np.where(bright_tile & (finalTileLSTbin == i) &
                                     (finalTileList['STATUS'] < 2))[0]
                    scheduled.extend(found[:5 - len(scheduled)])
            # BRIGHT time
            if bright[i]:
                # Find all BRIGHT tiles in this LST bin with STATUS < 2.
                found = np.where(bright_tile & (finalTileLSTbin == i) &
                                 (finalTileList['STATUS'] < 2))[0]
                # Schedule the first 5.
                scheduled.extend(found[:5])
            # Assign priorites to each scheduled tile.
            finalTileList['PRIORITY'][scheduled] = 3 + np.arange(len(scheduled))
            planList0.extend(scheduled)

        self.log.info('Afternoon plan contains {0} tiles.'
                      .format(len(planList0)))
        table = finalTileList[planList0]
        table.meta['MOONFRAC'] = day_stats['MoonFrac']
        filename = 'obsplan{0}.fits'.format(date_string)
        table.write(filename, overwrite=True)

        tilesTODO = len(planList0)

        return filename
