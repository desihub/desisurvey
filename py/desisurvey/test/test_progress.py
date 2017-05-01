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
        self.assertEqual(p.completed(), 0.)
        self.assertEqual(type(p.get_tile(10)), astropy.table.Row)
        with self.assertRaises(ValueError):
            p.get_tile(-1)
        t = p._table
        self.assertEqual(len(np.unique(t['tileid'])), len(t))
        self.assertTrue(np.all(np.unique(t['pass']) == np.arange(8, dtype=int)))
        self.assertTrue(np.all(t['status'] == 0))
        self.assertTrue(np.all((-80 < t['dec']) & (t['dec'] < 80)))
        self.assertTrue(np.all((0 <= t['ra']) & (t['ra'] < 360)))
        self.assertTrue(np.all(t['mjd'] == 0))
        self.assertTrue(np.all(t['exptime'] == 0))
        self.assertTrue(np.all(t['snr2frac'] == 0))
        self.assertTrue(np.all(t['airmass'] == 0))
        self.assertTrue(np.all(t['seeing'] == 0))

    def test_add_exposures(self):
        """Add some exposures to a new table"""
        p = Progress()
        t = p._table
        tiles = t['tileid'][:10].data
        for i, tile_id in enumerate(tiles):
            p.add_exposure(tile_id, 58849. + i, 100., 0.5, 1.5, 1.1)
            self.assertTrue(p.get_tile(tile_id)['snr2frac'][0] == 0.5)
            self.assertTrue(np.all(p.get_tile(tile_id)['snr2frac'][1:] == 0.))
        self.assertEqual(p.completed(include_partial=True), 5.)
        self.assertEqual(p.completed(include_partial=False), 0.)

    def test_exposures_incrementing(self):
        """Successive exposures of the same tile must be time ordered"""
        p = Progress()
        t = p._table
        tile_id = t['tileid'][0]
        p.add_exposure(tile_id, 58849.0, 100., 0.5, 1.5, 1.1)
        p.add_exposure(tile_id, 58849.1, 100., 0.5, 1.5, 1.1)
        with self.assertRaises(ValueError):
            p.add_exposure(tile_id, 58849.0, 100., 0.5, 1.5, 1.1)

    def test_save_read(self):
        """Create, save and read a progress table"""
        p1 = Progress()
        tiles = p1._table['tileid'][:10].data
        for i, tile_id in enumerate(tiles):
            p1.add_exposure(tile_id, 58849. + i, 100., 0.5, 1.5, 1.1)
        p1.save('p1.fits')
        p2 = Progress('p1.fits')
        self.assertEqual(p2.completed(include_partial=True), 5.)
        self.assertEqual(p2.completed(include_partial=False), 0.)

    def test_version_check(self):
        """Cannot use progress with the wrong version"""
        p = Progress()
        p._table.meta['VERSION'] = -1
        p.save('progress.fits')
        with self.assertRaises(RuntimeError):
            Progress('progress.fits')

    def test_completed_truncates(self):
        """Completion value truncates at one"""
        p = Progress()
        tile_id = p._table['tileid'][0]
        p.add_exposure(tile_id, 58849.0, 100., 0.5, 1.5, 1.1)
        p.add_exposure(tile_id, 58849.1, 100., 0.5, 1.5, 1.1)
        p.add_exposure(tile_id, 58849.2, 100., 0.5, 1.5, 1.1)
        self.assertEqual(p.completed(include_partial=True), 1.)
        self.assertEqual(p.completed(include_partial=False), 1.)

    def test_completed_history(self):
        """Optional before_mjd gives completion history"""
        p = Progress()
        n = p.max_exposures
        tiles = p._table['tileid'][:n]
        mjds = 58849. + np.arange(n)
        for tile, mjd in zip(tiles, mjds):
            p.add_exposure(tile, mjd, 100., 0.2, 1.5, 1.1)
            p.add_exposure(tile, mjd + 0.1, 100., 0.3, 1.5, 1.1)
        self.assertEqual(p.completed(), 0.5 * n)
        for i, mjd in enumerate(mjds):
            self.assertEqual(p.completed(before_mjd=mjd), 0.5 * i)

    def test_max_exposures(self):
        """Cannot exceed max exposures for a single tile"""
        p = Progress()
        n = p.max_exposures + 1
        tile_id = p._table['tileid'][0]
        mjds = 58849. + np.arange(n)
        for mjd in mjds[:-1]:
            p.add_exposure(tile_id, mjd, 100., 0.2, 1.5, 1.1)
        with self.assertRaises(RuntimeError):
            p.add_exposure(tile_id, mjds[-1], 100., 0.2, 1.5, 1.1)

    def test_get_observed(self):
        """Get list of observed tiles"""
        p = Progress()
        tiles = p._table['tileid'][:10].data
        for i, tile_id in enumerate(tiles):
            p.add_exposure(tile_id, 58849. + i, 100., 0.5, 1.5, 1.1)
        self.assertTrue(
            np.all(p.get_observed(include_partial=True)['tileid'] == tiles))
        self.assertEqual(
            len(p.get_observed(include_partial=False)['tileid']), 0)

    def test_get_observed_copy(self):
        """Cannot modify internal table with get_observed() return value"""
        p = Progress()
        tiles = p._table['tileid'][:10].data
        for i, tile_id in enumerate(tiles):
            p.add_exposure(tile_id, 58849. + i, 100., 1.5, 1.5, 1.1)
        t = p.get_observed()
        t['tileid'][:10] = -1
        for i, tile_id in enumerate(tiles):
            self.assertEqual(t['tileid'][i], -1)
            self.assertEqual(p._table['tileid'][i], tile_id)

    def test_summary(self):
        """Summary contains one row per tile"""
        p = desisurvey.progress.Progress()
        self.assertEqual(len(p.get_summary('observed')), 0)
        self.assertEqual(len(p.get_summary('completed')), 0)
        self.assertEqual(len(p.get_summary('all')), p.num_tiles)
        airmass, seeing = 1.5, 1.1
        for i, t in enumerate(p._table['tileid'][:100]):
            p.add_exposure(t, 58000 + i, 1000., 0.25, airmass, seeing)
            p.add_exposure(t, 58000 + i + 0.5, 1000., 0.25, airmass, seeing)
        self.assertEqual(len(p.get_summary('observed')), 100)
        self.assertEqual(len(p.get_summary('completed')), 0)
        self.assertEqual(len(p.get_summary('all')), p.num_tiles)
        s = p.get_summary('observed')
        self.assertTrue(np.all(s['mjd_max'] > s['mjd_min']))
        self.assertTrue(np.all(s['airmass'] == airmass))
        self.assertTrue(np.all(s['seeing'] == seeing))
        self.assertTrue(np.all(s['exptime'] == 2000.))
        self.assertTrue(np.all(s['snr2frac'] == 0.5))
