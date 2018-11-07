import tempfile, unittest, os, shutil, uuid, datetime
import numpy as np
from astropy.table import Table

import desimodel.io

import desisurvey.config
import desisurvey.ephem
from desisurvey.scripts import surveyinit

class TestScripts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory.
        cls.tmpdir = tempfile.mkdtemp()
        # Write output files to this temporary directory.
        config = desisurvey.config.Configuration()
        config.set_output_path(cls.tmpdir)
        # Run for 1 week for testing (but include some time in each program)
        start = datetime.date(2019,12,1)
        stop = datetime.date(2019,12,8)
        desisurvey.ephem.START_DATE = start
        desisurvey.ephem.STOP_DATE = stop
        config.first_day.set_value(start)
        config.last_day.set_value(stop)
        # Use just a subset of the tiles for faster testing
        tiles = Table(desimodel.io.load_tiles())
        subset = (35 < tiles['RA']) & (tiles['RA'] < 55) & \
                 (-10 < tiles['DEC']) & (tiles['DEC'] < 20)
        tiles_file = os.path.join(cls.tmpdir, 'tiles-subset.fits')
        tiles[subset].write(tiles_file)
        config.tiles_file.set_value(tiles_file)

    @classmethod
    def tearDownClass(cls):
        # Remove the directory after the test.
        shutil.rmtree(cls.tmpdir)
        # Reset our configuration.
        desisurvey.config.Configuration.reset()
        desisurvey.ephem._ephem = None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_scripts(self):
        cmd = 'surveyinit --max-cycles 5 --init zero'
        args = surveyinit.parse(cmd.split()[1:])
        surveyinit.main(args)
        '''
        cmd = 'surveyplan --create --rules rules-layers.yaml'
        args = surveyplan.parse(cmd.split()[1:])
        surveyplan.main(args)
        '''
        for filename in [
            'ephem_2019-12-01_2019-12-08.fits',
            'surveyinit.fits',
            ]:
            filepath = os.path.join(self.tmpdir, filename)
            self.assertTrue(os.path.exists(filepath), 'Missing {}'.format(filename))


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
