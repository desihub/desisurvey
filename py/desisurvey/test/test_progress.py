import unittest
import tempfile
import shutil
import os

import numpy as np

import desisurvey.config

from ..progress import *


class TestProgress(unittest.TestCase):

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

    def test_ctor(self):
        """Create a new table from scratch"""
        p = Progress()
        self.assertTrue(p.max_exposures == len(p._table['mjd'][0]))
        p = p._table
        self.assertEqual(len(np.unique(p['tileid'])), len(p))
        self.assertTrue(np.all(np.unique(p['pass']) == np.arange(8, dtype=int)))
        self.assertTrue(np.all(p['status'] == 0))
        self.assertTrue(np.all((-80 < p['dec']) & (p['dec'] < 80)))
        self.assertTrue(np.all((0 <= p['ra']) & (p['ra'] < 360)))
        self.assertTrue(np.all(p['mjd'] == 0))
        self.assertTrue(np.all(p['exptime'] == 0))
        self.assertTrue(np.all(p['snrfrac'] == 0))
        self.assertTrue(np.all(p['airmass'] == 0))
        self.assertTrue(np.all(p['seeing'] == 0))
