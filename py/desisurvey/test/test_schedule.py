import unittest
import tempfile
import shutil
import os

import numpy as np

import desisurvey.config

from desisurvey.old.schedule import *


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
        """Create a temporary scheduler (requires healpy)"""
        start = desisurvey.utils.get_date('2020-01-01')
        stop = desisurvey.utils.get_date('2020-01-03')
        ephem = desisurvey.ephemerides.Ephemerides(start, stop)
        initialize(ephem)

    def test_lst_range(self):
        """Test that 0 <= LST < 360 deg"""
        p = Scheduler(os.path.join(self.tmpdir, 'scheduler.fits'))
        lst = p.etable['lst']
        self.assertTrue(np.min(lst) >= 0.)
        self.assertTrue(np.max(lst) < 360.)

    def test_time_index_conversion(self):
        """Test time <-> index round trips"""
        p = Scheduler(os.path.join(self.tmpdir, 'scheduler.fits'))
        for i in range(p.num_nights * p.num_times):
            t = p.time_of_index(i)
            self.assertEqual(i, p.index_of_time(t))

    def test_spatial_index_conversion(self):
        """Test tile_id -> index"""
        p = Scheduler(os.path.join(self.tmpdir, 'scheduler.fits'))
        for i in range(len(p.tiles)):
            tid = p.tiles['tileid'][i]
            pix = p.tiles['map'][i]
            self.assertEqual(p.index_of_tile(tid), pix)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
