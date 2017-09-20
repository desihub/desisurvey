import tempfile, unittest, os, shutil, uuid, datetime
import numpy as np

import desisurvey.config
from desisurvey.scripts import surveyinit, surveyplan

class TestScripts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory.
        cls.tmpdir = tempfile.mkdtemp()
        # Write output files to this temporary directory.
        config = desisurvey.config.Configuration()
        start = datetime.date(2019,12,1)
        stop = datetime.date(2019,12,3)
        config.first_day._value = start
        config.last_day._value = stop
        config.set_output_path(cls.tmpdir)

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
            'plan_2019-12-01.fits',
            'scheduler.fits',
            'surveyinit.fits',
            ]:
            filepath = os.path.join(self.tmpdir, filename)
            self.assertTrue(os.path.exists(filepath), 'Missing {}'.format(filename))

if __name__ == '__main__':
    unittest.main()
