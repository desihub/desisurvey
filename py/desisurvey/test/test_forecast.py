import unittest
import os
import datetime
import tempfile
import shutil

import numpy as np


import desisurvey.config
import desisurvey.ephem
from desisurvey.forecast import Forecast


class TestForecast(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory.
        cls.tmpdir = tempfile.mkdtemp()
        # Write output files to this temporary directory.
        config = desisurvey.config.Configuration()
        config.set_output_path(cls.tmpdir)
        # Run for 1 week for testing (but include some time in each program)
        start = datetime.date(2019,12,1)
        stop = datetime.date(2019,12,8)
        desisurvey.ephem.START_DATE = start
        desisurvey.ephem.STOP_DATE = stop
        config.first_day.set_value(start)
        config.last_day.set_value(stop)

    @classmethod
    def tearDownClass(cls):
        # Remove the directory after the test.
        shutil.rmtree(cls.tmpdir)
        # Reset our configuration.
        desisurvey.config.Configuration.reset()
        desisurvey.ephem._ephem = None

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
