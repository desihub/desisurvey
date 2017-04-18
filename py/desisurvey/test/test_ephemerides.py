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
        self.assertEqual(ephem.start.mjd, ephem.get_night(0)['MJDstart'])
        self.assertEqual(ephem.start.mjd, ephem.get_night(start)['MJDstart'])
        self.assertEqual(ephem.start.mjd, ephem.get_night(ephem.start)['MJDstart'])

        ephem = ephem._table
        self.assertEqual(len(ephem), 31)
        self.assertTrue(np.all(ephem['MJDsunrise'] > ephem['MJDsunset']))
        self.assertTrue(np.all(ephem['MJDetwi'] > ephem['MJDe13twi']))
        self.assertTrue(np.all(ephem['MJDmtwi'] < ephem['MJDm13twi']))
        self.assertGreater(np.max(ephem['MoonFrac']), 0.99)
        self.assertLessEqual(np.max(ephem['MoonFrac']), 1.0)
        self.assertLess(np.min(ephem['MoonFrac']), 0.01)
        self.assertGreaterEqual(np.min(ephem['MoonFrac']), 0.00)
        self.assertTrue(np.all(ephem['MJDmoonrise'] < ephem['MJDmoonset']))

        for x in ephem:

            date = Time(x['MJDstart'], format='mjd').datetime.date()
            night = date.strftime('%Y%m%d')
            for key in [
                    'MJDsunset', 'MJDsunrise',
                    'MJDe13twi', 'MJDm13twi',
                    'MJDetwi', 'MJDmtwi',
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
        full = np.empty(ephem.num_days, bool)
        for i in range(ephem.num_days):
            night = start + datetime.timedelta(days=i)
            full[i] = ephem.is_full_moon(night)
        expected = np.zeros_like(full, bool)
        expected[9:16] = True
        self.assertTrue(np.all(full == expected))

if __name__ == '__main__':
    unittest.main()
