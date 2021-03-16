import unittest
import datetime

import numpy as np

import desisurvey.tiles
import desisurvey.config
from desisurvey.test.base import Tester
from desisurvey.plan import Planner


class TestPlan(Tester):

    def test_plan(self):
        tiles = desisurvey.tiles.get_tiles()
        donefrac = np.zeros(tiles.ntiles, 'f4')
        num_nights = (self.stop - self.start).days
        gen = np.random.RandomState(123)
        config = desisurvey.config.Configuration()
        for cadence in 'daily', 'monthly':
            config.fiber_assignment_cadence.set_value(cadence)
            plan = Planner(simulate=True)
            plan2 = None
            for i in range(num_nights):
                night = self.start + datetime.timedelta(i)
                # Run afternoon plan using original and restored objects.
                avail, planned = plan.afternoon_plan(night)
                if plan2 is not None:
                    # Check that the restored planner gives identical results.
                    avail2, planned2 = plan2.afternoon_plan(night)
                    self.assertTrue(np.array_equal(avail, avail2))
                    self.assertTrue(np.array_equal(planned, planned2))
                    self.assertTrue(np.array_equal(plan.tile_countdown, plan2.tile_countdown))
                    self.assertTrue(np.array_equal(plan.donefrac, plan2.donefrac))
                    self.assertTrue(np.array_equal(plan.designha, plan2.designha))
                # Mark a random set of tiles completed after this night.
                malreadydone = donefrac == 1
                donefrac[gen.choice(tiles.ntiles, tiles.ntiles // num_nights)] = 1.
                plan.set_donefrac(tiles.tileID[~malreadydone], donefrac[~malreadydone])
                # Save and restore our state.
                plan.save('snapshot.fits')
                plan2 = Planner(restore='snapshot.fits', simulate=True)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
