import unittest
import os
import datetime
import tempfile
import shutil

import numpy as np


import desisurvey.config
import desisurvey.ephem
import desisurvey.tiles
from desisurvey.plan import Planner


class TestPlan(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory.
        cls.tmpdir = tempfile.mkdtemp()
        # Write output files to this temporary directory.
        config = desisurvey.config.Configuration()
        config.set_output_path(cls.tmpdir)
        # Run for 1 week for testing (but include some time in each program)
        cls.start = datetime.date(2019,12,1)
        cls.stop = datetime.date(2019,12,8)
        desisurvey.ephem.START_DATE = cls.start
        desisurvey.ephem.STOP_DATE = cls.stop
        config.first_day.set_value(cls.start)
        config.last_day.set_value(cls.stop)

    @classmethod
    def tearDownClass(cls):
        # Remove the directory after the test.
        shutil.rmtree(cls.tmpdir)
        # Reset our configuration.
        desisurvey.config.Configuration.reset()
        desisurvey.ephem._ephem = None

    def test_plan(self):
        tiles = desisurvey.tiles.get_tiles()
        completed = np.zeros(tiles.ntiles, bool)
        num_nights = (self.stop - self.start).days
        gen = np.random.RandomState(123)
        for cadence in 'daily', 'monthly':
            plan = Planner(fiberassign_cadence=cadence)
            plan.initialize(self.start)
            for i in range(num_nights):
                night = self.start + datetime.timedelta(i)
                # Save and restore our state.
                plan.save()
                plan2 = Planner(fiberassign_cadence=cadence, restore_date=plan.last_night)
                # Run afternoon plan using original and restored objects.
                avail, pri = plan.afternoon_plan(night, completed)
                avail2, pri2 = plan2.afternoon_plan(night, completed)
                # Check that the restored planner gives identical results.
                self.assertTrue(np.array_equal(avail, avail2))
                self.assertTrue(np.array_equal(pri, pri2))
                self.assertTrue(np.array_equal(plan.tile_countdown, plan2.tile_countdown))
                # Mark a random set of tiles completed after this night.
                completed[gen.choice(tiles.ntiles, tiles.ntiles // num_nights)] = True


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
