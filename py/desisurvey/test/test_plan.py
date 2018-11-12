import unittest
import datetime

import numpy as np

import desisurvey.tiles
from desisurvey.test.base import Tester
from desisurvey.plan import Planner


class TestPlan(Tester):

    def test_plan(self):
        tiles = desisurvey.tiles.get_tiles()
        completed = np.zeros(tiles.ntiles, bool)
        num_nights = (self.stop - self.start).days
        gen = np.random.RandomState(123)
        for cadence in 'daily', 'monthly':
            plan = Planner(fiberassign_cadence=cadence)
            plan2 = None
            for i in range(num_nights):
                night = self.start + datetime.timedelta(i)
                # Run afternoon plan using original and restored objects.
                avail, pri = plan.afternoon_plan(night, completed)
                if plan2 is not None:
                    # Check that the restored planner gives identical results.
                    avail2, pri2 = plan2.afternoon_plan(night, completed)
                    self.assertTrue(np.array_equal(avail, avail2))
                    self.assertTrue(np.array_equal(pri, pri2))
                    self.assertTrue(np.array_equal(plan.tile_countdown, plan2.tile_countdown))
                # Mark a random set of tiles completed after this night.
                completed[gen.choice(tiles.ntiles, tiles.ntiles // num_nights)] = True
                # Save and restore our state.
                plan.save('snapshot.fits')
                plan2 = Planner(fiberassign_cadence=cadence, restore='snapshot.fits')



def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
