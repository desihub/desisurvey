import unittest
import os
import datetime
import tempfile
import shutil
from pkg_resources import resource_filename

import astropy.table
import astropy.io.ascii

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
        tiles = desisurvey.tiles.Tiles().read_tiles_table()
        subset = (35 < tiles['RA']) & (tiles['RA'] < 55) & \
                 (-10 < tiles['DEC']) & (tiles['DEC'] < 20)
        tiles['DESIGNHA'] = 0.0
        tiles['STATUS'] = 'unobs'
        tiles['PRIORITY'] = 1.0
        tiles_file = os.path.join(cls.tmpdir, 'tiles-subset.ecsv')
        tiles[subset].write(tiles_file)
        config.tiles_file.set_value(tiles_file)
        tiles = desisurvey.tiles.get_tiles(use_cache=False)

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

def read_horizons_moon_ephem():
    """Utility function for parsing data/horizons_2020_week1_moon.csv format

    Returns astropy.table.Table of Horizons moom ephemeris
    """
    import astropy.io.ascii

    filename = os.path.join('data', 'horizons_2020_week1_moon.csv')
    horizonsfile = resource_filename('desisurvey', filename)

    #- Horizons "csv" files have some non-standard header and footer data
    #- separated by $$SOE (start of ephemeris) and $$EOE (end of ephemeris)
    rows = list()
    with open(horizonsfile) as fx:
        #- Skip over header until "$$SOE" marker
        for line in fx:
            if line.startswith('$$SOE'):
                break
        #- Read data rows until "$$EOE" marker
        for line in fx:
            if line.startswith('$$EOE'):
                break
            else:
                rows.append(line)

    #- Parse CSV rows, providing column names
    names = ('date', 'jd', 'sun', 'moon', 'ra', 'dec',
             'az', 'alt', 'lst', 'frac', 'blank')
    moon_ephem = astropy.io.ascii.read(rows, names=names)
    return moon_ephem
