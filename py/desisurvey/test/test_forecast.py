import unittest
import tempfile
import shutil

import numpy as np

from desisurvey.config import Configuration
from desisurvey.ephemerides import Ephemerides
from desisurvey.forecast import Forecast


class TestExpCalc(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory.
        cls.tmpdir = tempfile.mkdtemp()
        # Write output files to this temporary directory.
        config = Configuration()
        config.set_output_path(cls.tmpdir)
        # Calculate 5-year ephemerides (takes ~80s)
        ephem = Ephemerides()

    @classmethod
    def tearDownClass(cls):
        # Remove the directory after the test.
        shutil.rmtree(cls.tmpdir)
        # Reset our configuration.
        desisurvey.config.Configuration.reset()

    def test_forecast(self):
        F = Forecast()
        F.set_overheads(
            setup={'DARK': 120, 'GRAY': 120, 'BRIGHT': 120},
            split={'DARK':  60, 'GRAY':  60, 'BRIGHT':  60},
            dead ={'DARK':   0, 'GRAY':   0, 'BRIGHT':   0})
        self.assertAlmostEqual(
            F.summary()['DARK']['Exposure time margin (%)'], 24.55, places=2)
