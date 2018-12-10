import unittest
import os
import datetime
import tempfile
import shutil

import numpy as np

import desisurvey.ephem
from desisurvey.test.base import Tester
from desisurvey.forecast import Forecast


class TestForecast(Tester):

    def test_forecast(self):
        gen = np.random.RandomState(123)
        ephem = desisurvey.ephem.get_ephem()
        W = gen.uniform(size=ephem.num_nights)
        tiles = desisurvey.tiles.get_tiles()
        HA = gen.normal(scale=15, size=tiles.ntiles)
        for twilight in True, False:
            forecast = Forecast(use_twilight=twilight, design_hourangle=HA, weather=W)
            forecast.summary()


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
