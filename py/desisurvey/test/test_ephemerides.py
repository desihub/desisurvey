import unittest
import os
import uuid
import datetime

import numpy as np

from astropy.time import Time
from astropy import units

from desisurvey.ephemerides import Ephemerides
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
        self.assertEqual(ephem.start.mjd, ephem.get_row(0)['MJDstart'])
        self.assertEqual(ephem.start.mjd, ephem.get_night(start)['MJDstart'])
        self.assertEqual(ephem.stop.mjd, ephem.get_row(-1)['MJDstart'] + 1)
        self.assertEqual(ephem.start.mjd,
                         ephem.get_night(ephem.start)['MJDstart'])
        self.assertEqual(ephem.num_nights,
                         int(round(ephem.stop.mjd - ephem.start.mjd)))

        etable = ephem._table
        self.assertEqual(len(etable), 30)
        self.assertTrue(np.all(etable['MJDsunrise'] > etable['MJDsunset']))
        self.assertTrue(np.all(etable['dusk'] > etable['brightdusk']))
        self.assertTrue(np.all(etable['dawn'] < etable['brightdawn']))
        self.assertGreater(np.max(etable['MoonFrac']), 0.99)
        self.assertLessEqual(np.max(etable['MoonFrac']), 1.0)
        self.assertLess(np.min(etable['MoonFrac']), 0.01)
        self.assertGreaterEqual(np.min(etable['MoonFrac']), 0.00)
        self.assertTrue(np.all(etable['MJDmoonrise'] < etable['MJDmoonset']))

        for i in range(ephem.num_nights):

            x = ephem.get_row(i)
            date = Time(x['MJDstart'], format='mjd').datetime.date()
            night = date.strftime('%Y%m%d')
            for key in [
                    'MJDsunset', 'MJDsunrise',
                    'brightdusk', 'brightdawn',
                    'dusk', 'dawn',
                ]:
                #- AZ local time
                localtime = Time(x[key], format='mjd') - 7*units.hour
                #- YEARMMDD of sunset for that time
                yearmmdd = (localtime - 12*units.hour).to_datetime().strftime('%Y%m%d')
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

if __name__ == '__main__':
    unittest.main()
