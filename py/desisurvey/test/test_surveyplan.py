import unittest
import os
import datetime
import tempfile
import shutil

import numpy as np

from astropy.table import Table
from astropy.time import Time
import astropy.units as u

import desisurvey.config
from desisurvey.ephemerides import Ephemerides
from desisurvey.progress import Progress
from desisurvey.afternoonplan import surveyPlan

class TestSurveyPlan(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory.
        cls.tmpdir = tempfile.mkdtemp()
        # Write output files to this temporary directory.
        config = desisurvey.config.Configuration()
        config.set_output_path(cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        # Remove the directory after the test.
        shutil.rmtree(cls.tmpdir)
        # Reset our configuration.
        desisurvey.config.Configuration.reset()

    def test_planning(self):
        start = datetime.date(2019, 9, 1)
        stop = datetime.date(2019, 10, 1)
        ephem = Ephemerides(start, stop, use_cache=False, write_cache=False)
        sp = surveyPlan(ephem.start.mjd, ephem.stop.mjd, ephem)

        tiles = sp.tiles
        dLST = tiles['LSTMAX'] - tiles['LSTMIN']
        wraparound = (dLST < -180)
        dLST[wraparound] += 360
        self.assertGreater(np.min(dLST), 0)
        self.assertLess(np.max(dLST), 30)
        self.assertTrue(np.all(tiles['EXPLEN'] > 500))

        #- Plan night 0; set the first 10 tiles as observed
        progress = Progress()
        day0 = ephem.get_row(0)
        planfile0 = sp.afternoonPlan(day0, progress)
        tiles = progress._table['tileid'][:10]
        t0 = Time(58849., format='mjd')
        for i, tile_id in enumerate(tiles):
            progress.add_exposure(
                tile_id, t0 + i * u.hour, 1e3 * u.s, 1., 1.5, 1.1, 0, 0, 0)

        #- Plan night 1
        day1 = ephem.get_row(1)
        planfile1 = sp.afternoonPlan(day1, progress)
        plan1 = Table.read(planfile1)

        #- Tiles observed on night 0 shouldn't appear in night 1 plan
        self.assertTrue(not np.any(np.in1d(tiles, plan1['TILEID'])))

        #- Some night 0 tiles that weren't observed should show up again
        plan0 = Table.read(planfile0)
        self.assertTrue(np.any(np.in1d(plan0['TILEID'][10:], plan1['TILEID'])))

if __name__ == '__main__':
    unittest.main()
