import unittest
import os
import uuid

import numpy as np
from astropy.time import Time
from astropy.table import Table

from desisurvey.ephemerides import Ephemerides
from desisurvey.afternoonplan import surveyPlan

class TestSurveyPlan(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import uuid
        cls.testdir = os.path.abspath('./test-{}'.format(uuid.uuid4()))
        cls.origdir = os.getcwd()
        os.mkdir(cls.testdir)
        os.chdir(cls.testdir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.origdir)
        if os.path.isdir(cls.testdir):
            import shutil
            shutil.rmtree(cls.testdir)

    def test_planning(self):
        start = Time('2019-09-01T00:00:00')
        end = Time('2019-10-01T00:00:00')
        ephem = Ephemerides(start, end, use_cache=False)._table
        sp = surveyPlan(start.mjd, end.mjd, ephem, tilesubset=None)

        tiles = sp.tiles
        dLST = tiles['LSTMAX'] - tiles['LSTMIN']
        wraparound = (dLST < -12)
        dLST[wraparound] += 24
        self.assertGreater(np.min(dLST), 0)
        self.assertLess(np.max(dLST), 2)
        self.assertTrue(np.all(tiles['EXPLEN'] > 500))

        #- Plan night 0; set the first 10 tiles as observed
        planfile0 = sp.afternoonPlan(ephem[0], '20190901', tiles_observed=[])
        plan0 = Table.read(planfile0)
        plan0['STATUS'][0:10] = 2

        #- Plan night 1
        planfile1 = sp.afternoonPlan(ephem[1], '20190902', tiles_observed=plan0)
        plan1 = Table.read(planfile1)

        #- Tiles observed on night 0 shouldn't appear in night 1 plan
        self.assertTrue(not np.any(np.in1d(plan0['TILEID'][0:10], plan1['TILEID'])))

        #- Some night 0 tiles that weren't observed should show up again
        self.assertTrue(np.any(np.in1d(plan0['TILEID'][10:], plan1['TILEID'])))

if __name__ == '__main__':
    unittest.main()
