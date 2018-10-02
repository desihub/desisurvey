import tempfile, unittest, os, shutil, uuid, datetime
import numpy as np
from astropy.table import Table

import desimodel.io

import desisurvey.config
from desisurvey.scripts import surveyinit, surveyplan

class TestScripts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory.
        cls.tmpdir = tempfile.mkdtemp()
        # Write output files to this temporary directory.
        config = desisurvey.config.Configuration()
        config.set_output_path(cls.tmpdir)
        # Just run for 2 days for testing
        start = datetime.date(2019,12,1)
        stop = datetime.date(2019,12,3)
        config.first_day.set_value(start)
        config.last_day.set_value(stop)
        # Use weather from 2009, which has some lost time both days.
        config.weather.set_value('Y2009')
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

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_scripts(self):
        cmd = 'surveyinit --max-cycles 5 --nbins 20'
        args = surveyinit.parse(cmd.split()[1:])
        # args = surveyinit.parse(['--verbose', ])
        surveyinit.main(args)

        cmd = 'surveyplan --create --rules rules-layers.yaml'
        args = surveyplan.parse(cmd.split()[1:])
        surveyplan.main(args)

        for filename in [
            'ephem_2019-12-01_2019-12-03.fits',
            'plan.fits',
            'plan.fits',
            'scheduler.fits',
            'surveyinit.fits',
            ]:
            filepath = os.path.join(self.tmpdir, filename)
            self.assertTrue(os.path.exists(filepath), 'Missing {}'.format(filename))

if __name__ == '__main__':
    unittest.main()
