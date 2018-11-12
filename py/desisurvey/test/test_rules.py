import unittest

import numpy as np

import desisurvey.tiles
from desisurvey.test.base import Tester
from desisurvey.rules import Rules


class TestRules(Tester):

    def test_rules(self):
        rules = Rules('rules-layers.yaml')
        tiles = desisurvey.tiles.get_tiles()
        completed = np.ones(tiles.ntiles, bool)
        rules.apply(completed)
        completed[:] = False
        rules.apply(completed)
        gen = np.random.RandomState(123)
        for i in range(10):
            completed[gen.choice(tiles.ntiles, tiles.ntiles // 10, replace=False)] = True
            rules.apply(completed)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
