import unittest
import tempfile
import shutil
import os

import numpy as np

import desisurvey.config

from ..plan import *


class TestPlan(unittest.TestCase):

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

    def test_initialize(self):
        """Create a temporary planner init file for the first 2 days in 2020"""
        start = desisurvey.utils.get_date('2020-01-01')
        stop = desisurvey.utils.get_date('2020-01-03')
        ephem = desisurvey.ephemerides.Ephemerides(start, stop)
        initialize(ephem)

    def test_time_index_conversion(self):
        """Test time <-> index round trips"""
        p = Planner(os.path.join(self.tmpdir, 'planner.fits'))
        for i in range(p.num_nights * p.num_times):
            t = p.time_of_index(i)
            self.assertEqual(i, p.index_of_time(t))
