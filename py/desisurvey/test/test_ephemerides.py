import unittest
import os
import uuid
import datetime

import numpy as np

from astropy.time import Time
import astropy.units as u

from desisurvey.ephemerides import Ephemerides, get_grid
from desisurvey.config import Configuration


class TestEphemerides(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.origdir = os.getcwd()
        cls.testdir = os.path.abspath('./test-{}'.format(uuid.uuid4()))
        os.mkdir(cls.testdir)
        os.chdir(cls.testdir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.origdir)
        if os.path.isdir(cls.testdir):
            import shutil
            shutil.rmtree(cls.testdir)

    def test_getephem(self):
        """Tabulate one month of ephemerides"""
        start = datetime.date(2019, 9, 1)
        stop = datetime.date(2019, 10, 1)
        ephem = Ephemerides(start, stop, use_cache=False, write_cache=False)
        self.assertEqual(ephem.start.mjd, ephem.get_row(0)['noon'])
        self.assertEqual(ephem.start.mjd, ephem.get_night(start)['noon'])
        self.assertEqual(ephem.stop.mjd, ephem.get_row(-1)['noon'] + 1)
        self.assertEqual(ephem.start.mjd,
                         ephem.get_night(ephem.start)['noon'])
        self.assertEqual(ephem.num_nights,
                         int(round(ephem.stop.mjd - ephem.start.mjd)))

        etable = ephem._table
        self.assertEqual(len(etable), 30)
        self.assertTrue(np.all(etable['dusk'] > etable['noon']))
        self.assertTrue(np.all(etable['dawn'] > etable['dusk']))
        self.assertTrue(np.all(etable['dusk'] > etable['brightdusk']))
        self.assertTrue(np.all(etable['dawn'] < etable['brightdawn']))
        self.assertGreater(np.max(etable['moon_illum_frac']), 0.99)
        self.assertLessEqual(np.max(etable['moon_illum_frac']), 1.0)
        self.assertLess(np.min(etable['moon_illum_frac']), 0.01)
        self.assertGreaterEqual(np.min(etable['moon_illum_frac']), 0.00)
        self.assertTrue(np.all(etable['moonrise'] < etable['moonset']))

        for i in range(ephem.num_nights):

            x = ephem.get_row(i)
            date = Time(x['noon'], format='mjd').datetime.date()
            night = date.strftime('%Y%m%d')
            for key in [
                    'brightdusk', 'brightdawn',
                    'dusk', 'dawn',
                ]:
                #- AZ local time
                localtime = Time(x[key], format='mjd') - 7*u.hour
                #- YEARMMDD of sunset for that time
                yearmmdd = (localtime - 12*u.hour).to_datetime().strftime('%Y%m%d')
                msg = '{} != {} for {}={}'.format(night, yearmmdd, key, x[key])
                self.assertEqual(night, yearmmdd, msg)

    def test_full_moon(self):
        """Verify that the full moon break in Sep-2019 occurs on days 10-16"""
        start = datetime.date(2019, 9, 1)
        stop = datetime.date(2019, 9, 30)
        ephem = Ephemerides(start, stop, use_cache=False, write_cache=False)
        full = np.empty(ephem.num_nights, bool)
        for i in range(ephem.num_nights):
            night = start + datetime.timedelta(days=i)
            full[i] = ephem.is_full_moon(night)
        expected = np.zeros_like(full, bool)
        expected[9:16] = True
        self.assertTrue(np.all(full == expected))

    def test_get_grid(self):
        """Verify grid calculations"""
        for step_size in (1 * u.min, 0.3 * u.hour):
            for night_start in (-6 * u.hour, -6.4 * u.hour):
                g = get_grid(step_size, night_start)
                self.assertTrue(g[0] == night_start.to(u.day).value)
                self.assertAlmostEqual(g[1] - g[0], step_size.to(u.day).value)
                self.assertAlmostEqual(g[-1] - g[0],
                                (len(g) - 1) * step_size.to(u.day).value)


if __name__ == '__main__':
    unittest.main()
