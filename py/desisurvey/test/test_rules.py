import unittest

import numpy as np

import desisurvey.tiles
from desisurvey.test.base import Tester
from desisurvey.rules import Rules


class TestRules(Tester):

    def test_rules(self):
        rules = Rules('rules-layers.yaml')
        tiles = desisurvey.tiles.get_tiles()
        donefrac = np.ones(tiles.ntiles, 'f4')
        rules.apply(donefrac)
        donefrac[:] = 0
        rules.apply(donefrac)
        gen = np.random.RandomState(123)
        for i in range(10):
            donefrac[gen.choice(tiles.ntiles, tiles.ntiles // 10, replace=False)] = 1
            rules.apply(donefrac)
