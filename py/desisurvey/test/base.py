import unittest
import os
import datetime
import tempfile
import shutil

import astropy.table

import desimodel.io

import desisurvey.config
import desisurvey.ephem
import desisurvey.tiles


class Tester(unittest.TestCase):
    """Base class for package unit tests.

    Updates the configuration to:
     - change the default survey and ephemerides span to one month.
     - use a reduced set of tiles.
     - save outputs to a temporary directory.

    On exit, the temporary directory is removed, and any
    cached ephemerides or tiles are cleared.
    """
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory.
        cls.tmpdir = tempfile.mkdtemp()
        # Write output files to this temporary directory.
        config = desisurvey.config.Configuration()
        config.set_output_path(cls.tmpdir)
        # Run for 1 week for testing (but include some time in each program).
        cls.start = datetime.date(2019,12,31)
        cls.stop = datetime.date(2020,2,28)
        cls.start_save = desisurvey.ephem.START_DATE
        cls.stop_save = desisurvey.ephem.STOP_DATE
        desisurvey.ephem.START_DATE = cls.start
        desisurvey.ephem.STOP_DATE = cls.stop
        config.first_day.set_value(cls.start)
        config.last_day.set_value(cls.stop)
        # Use a subset of the tiles for faster testing. These cuts mirror
        # those used to trim the list of tiles in a test branch of desimodel,
        # so should give identical results with local and travis tests.
        tiles = astropy.table.Table(desimodel.io.load_tiles())
        subset = (35 < tiles['RA']) & (tiles['RA'] < 55) & \
                 (-10 < tiles['DEC']) & (tiles['DEC'] < 20)
        tiles_file = os.path.join(cls.tmpdir, 'tiles-subset.fits')
        tiles[subset].write(tiles_file)
        config.tiles_file.set_value(tiles_file)
        surveyinit = astropy.table.Table()
        surveyinit['tileID'] = tiles['TILEID']
        surveyinit['HA'] = tiles['TILEID']*0.0
        surveyinit['HA_DARK'] = tiles['TILEID']*0.0
        surveyinit['HA_GRAY'] = tiles['TILEID']*0.0
        surveyinit['HA_BRIGHT'] = tiles['TILEID']*0.0
        surveyinit.meta['EXTNAME'] = 'DESIGN'
        surveyinit[subset].write(os.path.join(cls.tmpdir, 'surveyinit.fits'))

    @classmethod
    def tearDownClass(cls):
        # Remove the directory after the test.
        shutil.rmtree(cls.tmpdir)
        # Reset our configuration.
        desisurvey.config.Configuration.reset()
        # Restore ephemerides date range.
        desisurvey.ephem.START_DATE = cls.start_save
        desisurvey.ephem.STOP_DATE = cls.stop_save
        # Clear caches.
        desisurvey.ephem._ephem = None
        desisurvey.tiles._cached_tiles = {}
