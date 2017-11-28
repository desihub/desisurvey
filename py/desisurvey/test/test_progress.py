import unittest
import tempfile
import shutil
import os

import numpy as np

import astropy.units as u
from astropy.table import Table

import desisurvey.config

from desisurvey.progress import *


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
        self.assertEqual(p.first_mjd, 0.)
        self.assertEqual(p.last_mjd, 0.)
        self.assertEqual(p.last_tile, None)
        self.assertEqual(p.num_exp, 0)
        self.assertEqual(p.completed(), 0.)
        self.assertEqual(type(p.get_tile(260)), astropy.table.Row)
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
        self.assertTrue(np.all(t['transparency'] == 0))

    def test_add_exposures(self):
        """Add some exposures to a new table"""
        p = Progress()
        t = p._table
        tiles = t['tileid'][:10].data
        t0 = astropy.time.Time('2020-01-01 07:00')
        for i, tile_id in enumerate(tiles):
            p.add_exposure(
                tile_id, t0 + i * u.hour, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)
            self.assertTrue(p.get_tile(tile_id)['snr2frac'][0] == 0.5)
            last_tile = p.get_tile(tile_id)
            self.assertTrue(np.array_equal(
                last_tile.as_void(), p.last_tile.as_void()))
            self.assertTrue(np.all(last_tile['snr2frac'][1:] == 0.))
        self.assertEqual(p.completed(include_partial=True), 5.)
        self.assertEqual(p.completed(include_partial=False), 0.)
        self.assertTrue(p.first_mjd > 0)
        self.assertTrue(p.last_mjd > p.first_mjd)

    def test_restore_status(self):
        """Test that status is restored"""
        p = Progress()
        t = p._table
        tiles = t['tileid'][:10].data
        t0 = astropy.time.Time('2020-01-01 07:00')
        for i, tile_id in enumerate(tiles):
            p.add_exposure(
                tile_id, t0 + i * u.hour, 1e3 * u.s, 2.0, 1.5, 1.1, 1, 0, 0, 0)
        good_status = p._table['status'].copy()
        p._table['status'] = 0
        p2 = Progress(p._table)
        self.assertTrue(np.all(p2._table['status'] == good_status))

    def test_get_exposures(self):
        """Test get_exposures() method"""
        p = Progress()
        t = p._table
        tiles = t['tileid'][:10].data
        t0 = astropy.time.Time('2020-01-01 07:00')
        for i, tile_id in enumerate(tiles):
            p.add_exposure(
                tile_id, t0 + i * u.hour, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)
        explist = p.get_exposures()
        self.assertEqual(explist.meta['EXTNAME'], 'EXPOSURES')
        self.assertTrue(np.all(np.diff(explist['MJD']) > 0))
        explist = p.get_exposures(tile_fields='index', exp_fields='lst')
        self.assertTrue(np.all(np.diff(explist['LST']) > 0))
        self.assertTrue(np.min(explist['LST'] >= 0))
        self.assertTrue(np.max(explist['LST'] < 360))
        with self.assertRaises(ValueError):
            p.get_exposures(tile_fields='mjd')
        with self.assertRaises(ValueError):
            p.get_exposures(tile_fields='nonexistent')
        # with self.assertRaises(ValueError):
        #     p.get_exposures(exp_fields='pass')
        explist = p.get_exposures(exp_fields='mjd,night,program')
        for row in explist:
            self.assertEqual(desisurvey.utils.get_date(row['MJD']),
                             desisurvey.utils.get_date(row['NIGHT']))
            night = str(desisurvey.utils.get_date(row['MJD']))
            self.assertEqual(night, str(desisurvey.utils.get_date(night)))

        #- Test roundtrip to disk
        expfile = os.path.join(self.tmpdir, 'test-exposures.fits')
        explist.write(expfile)
        newexp = Table.read(expfile)

        self.assertEqual(newexp.meta['EXTNAME'], 'EXPOSURES')
        self.assertEqual(explist['PROGRAM'].dtype, newexp['PROGRAM'].dtype)
        self.assertEqual(explist['NIGHT'].dtype, newexp['NIGHT'].dtype)
        self.assertTrue(np.all(explist['PROGRAM'] == newexp['PROGRAM']))
        self.assertTrue(np.all(explist['NIGHT'] == newexp['NIGHT']))

    def test_exposures_incrementing(self):
        """Successive exposures of the same tile must be time ordered"""
        p = Progress()
        t = p._table
        tile_id = t['tileid'][0]
        t0 = astropy.time.Time('2020-01-01 07:00')
        t1 = t0 + 1 * u.hour
        p.add_exposure(tile_id, t0, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)
        p.add_exposure(tile_id, t1, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)
        self.assertEqual(p.first_mjd, t0.mjd)
        self.assertEqual(p.last_mjd, t1.mjd)
        self.assertEqual(p.num_exp, 2)
        with self.assertRaises(ValueError):
            p.add_exposure(tile_id, t0, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)

    def test_save_read(self):
        """Create, save and read a progress table"""
        p1 = Progress()
        tiles = p1._table['tileid'][:10].data
        t0 = astropy.time.Time('2020-01-01 07:00')
        for i, tile_id in enumerate(tiles):
            p1.add_exposure(
                tile_id, t0 + i * u.hour, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)
        p1.save('p1.fits')
        p2 = Progress('p1.fits')
        self.assertEqual(p2.completed(include_partial=True), 5.)
        self.assertEqual(p2.completed(include_partial=False), 0.)
        self.assertTrue(p2.first_mjd > 0)
        self.assertTrue(p2.last_mjd > p2.first_mjd)
        self.assertEqual(p2.last_tile['tileid'], tiles[-1])

    def test_table_ctor(self):
        """Construct progress from a table"""
        p1 = Progress()
        tiles = p1._table['tileid'][:10].data
        t0 = astropy.time.Time('2020-01-01 07:00')
        for i, tile_id in enumerate(tiles):
            p1.add_exposure(
                tile_id, t0 + i * u.hour, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)
        p2 = Progress(p1._table)
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
        t0 = astropy.time.Time('2020-01-01 07:00')
        p.add_exposure(
            tile_id, t0 + 1 * u.hour, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)
        p.add_exposure(
            tile_id, t0 + 2 * u.hour, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)
        p.add_exposure(
            tile_id, t0 + 3 * u.hour, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)
        self.assertEqual(p.completed(include_partial=True), 1.)
        self.assertEqual(p.completed(include_partial=False), 1.)

    def test_completed_only_passes(self):
        """Test only_passes option to completed()"""
        p = Progress()
        self.assertEqual(p.completed(only_passes=range(9)), 0.)
        self.assertEqual(p.completed(only_passes=(7, 1)), 0.)
        self.assertEqual(p.completed(only_passes=1), 0.)
        pass1 = np.where(p._table['pass'] == 1)[0]
        pass7 = np.where(p._table['pass'] == 7)[0]
        n1, n7 = len(pass1), len(pass7)
        self.assertEqual(p.completed(only_passes=1, as_tuple=True),
                         (0., n1, 0.))
        self.assertEqual(p.completed(only_passes=(7, 1), as_tuple=True),
                         (0., n1 + n7, 0.))
        n = 10
        tiles = p._table['tileid'][list(pass1[:n]) + list(pass7[:n])]
        t0 = astropy.time.Time('2020-01-01 07:00')
        for tile_id in tiles:
            p.add_exposure(tile_id, t0, 1e3 * u.s, 1.5, 1.5, 1.1, 1, 0, 0, 0)
            t0 += 0.1 * u.day
        self.assertEqual(p.completed(only_passes=(7, 1)), 2 * n)
        self.assertEqual(p.completed(only_passes=7), n)
        self.assertEqual(p.completed(only_passes=(1,)), n)
        self.assertEqual(p.completed(only_passes=(1, 2, 3)), n)
        self.assertEqual(p.completed(only_passes=(2, 3)), 0)
        self.assertEqual(p.completed(only_passes=1, as_tuple=True),
                         (n, n1, 100. * n / n1))
        self.assertEqual(p.completed(only_passes=(7, 1), as_tuple=True),
                         (2 * n, n1 + n7, 100. * 2 * n / (n1 + n7)))

    def test_max_exposures(self):
        """Cannot exceed max exposures for a single tile"""
        p = Progress()
        n = p.max_exposures + 1
        tile_id = p._table['tileid'][0]
        mjds = 58849. + np.arange(n)
        tt = astropy.time.Time('2020-01-01 07:00') + np.arange(n) * u.hour
        for t in tt[:-1]:
            p.add_exposure(tile_id, t, 1e3 * u.s, 0.2, 1.5, 1.1, 1, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            p.add_exposure(tile_id, tt[-1], 1e3 * u.s,
                           0.2, 1.5, 1.1, 1, 0, 0, 0)

    def test_summary(self):
        """Summary contains one row per tile"""
        p = desisurvey.progress.Progress()
        self.assertEqual(len(p.get_summary('observed')), 0)
        self.assertEqual(len(p.get_summary('completed')), 0)
        self.assertEqual(len(p.get_summary('all')), p.num_tiles)
        self.assertTrue(np.all(p.get_summary('all')['nexp'] == 0))
        n, airmass, seeing, transp = 100, 1.5, 1.1, 0.95
        t0 = astropy.time.Time('2020-01-01 07:00')
        for i, t in enumerate(p._table['tileid'][:n]):
            p.add_exposure(
                t, t0 + i * u.hour, 1e3 * u.s, 0.25, airmass, seeing,
                transp, 0, 0, 0)
            p.add_exposure(t, t0 + (i + 0.5) * u.hour, 1e3 * u.s, 0.25,
                           airmass, seeing, transp, 0, 0, 0)
        self.assertEqual(len(p.get_summary('observed')), 100)
        self.assertEqual(len(p.get_summary('completed')), 0)
        self.assertTrue(np.all(p.get_summary('observed')['nexp'] == 2))
        self.assertTrue(np.all(p.get_summary('completed')['nexp'] == 0))
        self.assertEqual(len(p.get_summary('all')), p.num_tiles)
        s = p.get_summary('observed')
        self.assertTrue(np.all(s['mjd_max'] > s['mjd_min']))
        self.assertTrue(np.all(s['airmass'] == airmass))
        self.assertTrue(np.all(s['seeing'] == seeing))
        self.assertTrue(np.all(s['transparency'] == transp))
        self.assertTrue(np.all(s['exptime'] == 2000.))
        self.assertTrue(np.all(s['snr2frac'] == 0.5))
        self.assertTrue(np.all(s['nexp'][:n] == 2))
        self.assertTrue(np.all(s['nexp'][n:] == 0))

    def test_copy_bad(self):
        """Copy with no range selects everything"""
        p1 = Progress()
        with self.assertRaises(ValueError):
            p1.copy_range(58849, 58849 - 1)

    def test_copy_all(self):
        """Copy with no range selects everything"""
        p1 = Progress()
        tiles = p1._table['tileid'][:10].data
        t0 = astropy.time.Time('2020-01-01 07:00')
        for i, tile_id in enumerate(tiles):
            p1.add_exposure(
                tile_id, t0 + i * u.hour, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)
        p2 = p1.copy_range()
        self.assertTrue(np.all(np.array(p1._table) == np.array(p2._table)))

    def test_copy_some(self):
        """Copy with range selects subset"""
        p1 = Progress()
        n = 10
        tiles = p1._table['tileid'][:n].data
        tt = astropy.time.Time('2020-01-01 07:00') + np.arange(n) * u.hour
        for t, tile_id in zip(tt, tiles):
            p1.add_exposure(tile_id, t, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)
        for t, tile_id in zip(tt, tiles):
            p1.add_exposure(
                tile_id, t + 100 * u.day, 1e3 * u.s, 0.5, 1.5, 1.1, 1, 0, 0, 0)
        self.assertEqual(p1.completed(), n)
        # Selects everything.
        mjd0 = tt[0].mjd
        p2 = p1.copy_range(mjd0, mjd0 + 200)
        self.assertTrue(np.all(np.array(p1._table) == np.array(p2._table)))
        p2 = p1.copy_range(mjd0, None)
        self.assertTrue(np.all(np.array(p1._table) == np.array(p2._table)))
        p2 = p1.copy_range(None, mjd0 + 200)
        self.assertTrue(np.all(np.array(p1._table) == np.array(p2._table)))
        # Selects half of the exposures.
        p2 = p1.copy_range(None, mjd0 + 100)
        self.assertEqual(p2.completed(), 0.5 * n)
        p2 = p1.copy_range(mjd0, mjd0 + 100)
        self.assertEqual(p2.completed(), 0.5 * n)
        p2 = p1.copy_range(mjd0 + 100, mjd0 + 200)
        self.assertEqual(p2.completed(), 0.5 * n)
        p2 = p1.copy_range(mjd0 + 100, None)
        self.assertEqual(p2.completed(), 0.5 * n)
        # Selects none of the exposures.
        p2 = p1.copy_range(None, mjd0)
        self.assertEqual(p2.completed(), 0.)
        p2 = p1.copy_range(mjd0 + 200, None)
        self.assertEqual(p2.completed(), 0.)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
