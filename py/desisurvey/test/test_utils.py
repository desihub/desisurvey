import unittest
import datetime

import numpy as np

import astropy.time

from desisurvey import utils

class TestUtils(unittest.TestCase):

    def setUp(self):
        pass


    def test_get_date(self):
        """Test date conversions"""
        start = datetime.datetime(2019, 8, 23, 12)
        one_day = datetime.timedelta(days=1)
        for offset in range(500):
            day = start + one_day * offset
            # date -> date
            self.assertEqual(utils.get_date(day.date()), day.date())
            # datetime -> date
            self.assertEqual(utils.get_date(day), day.date())
            # astropy time -> datetime -> date
            self.assertEqual(utils.get_date(astropy.time.Time(day)), day.date())
            # YYYY-MM-DD -> datetime -> date
            self.assertEqual(utils.get_date(str(day.date())), day.date())


    def test_monsoon(self):
        """Monsoon based on (month, day) comparisons"""
        for year in range(2019, 2025):
            self.assertFalse(utils.is_monsoon(datetime.date(year, 7, 12)))
            self.assertTrue(utils.is_monsoon(datetime.date(year, 7, 13)))
            self.assertTrue(utils.is_monsoon(datetime.date(year, 8, 26)))
            self.assertFalse(utils.is_monsoon(datetime.date(year, 8, 27)))


    def test_local_noon(self):
        """The telescope is 7 hours behind of UTC during winter and summer."""
        for month in (1, 7):
            day = datetime.date(2019, month, 1)
            noon = utils.local_noon_on_date(day)
            self.assertEqual(noon.datetime.date(), day)
            self.assertEqual(noon.datetime.time(), datetime.time(hour=12 + 7))


    def test_sort2arr(self):
        a = [1,2,3]
        b = [3,1,2]
        c = utils.sort2arr(a,b)
        self.assertTrue(np.all(c == np.array([2,3,1])))

    def test_inLSTwindow(self):
        # inLSTwindow(lst, begin, end)
        self.assertTrue(utils.inLSTwindow(10, 5, 15))
        self.assertFalse(utils.inLSTwindow(100, 5, 15))

        self.assertTrue(utils.inLSTwindow(0, -5, 10))
        self.assertFalse(utils.inLSTwindow(-10, -5, 10))
        self.assertFalse(utils.inLSTwindow(15, -5, 10))

    def test_equ2gal_J2000(self):
        # Test against astropy.SkyCoords result.
        ra, dec = 15, 20
        l, b = utils.equ2gal_J2000(ra, dec)
        self.assertAlmostEqual(l, 125.67487462, 4)
        self.assertAlmostEqual(b, -42.82614243, 4)

    def test_angsep(self):
        self.assertAlmostEqual(utils.angsep(0,0,10,0), 10.0)
        self.assertAlmostEqual(utils.angsep(0,0,0,10), 10.0)
        self.assertAlmostEqual(utils.angsep(20,0,10,0), 10.0)
        self.assertAlmostEqual(utils.angsep(0,20,0,10), 10.0)
        self.assertAlmostEqual(utils.angsep(60,70,60,50), 20.0)

    def test_radec2altaz(self):

        LST = 168.86210588900758  # 2000-01-01 12:00:00 at KPNO according to astropy.time
        ra, dec, lst = LST, 60, LST
        alt, az = utils.radec2altaz(ra, dec, lst)
        self.assertAlmostEqual(alt, 61.96710605261274, 2) # Values from Astropy SkyCoords
        self.assertAlmostEqual(az, 0.0011510242215743817, 2)

        # Value close to zenith
        ra, dec, lst = LST, 31.965, LST
        alt_z, az_z = utils.radec2altaz(ra, dec, lst)
        alt_plus, az_minus = utils.radec2altaz(ra, dec+5.0, lst)
        alt_minus, az_minus = utils.radec2altaz(ra, dec-5.0, lst)
        self.assertAlmostEqual(alt_plus, alt_minus, 2)
        alt_plus, az_minus = utils.radec2altaz(ra+5.0, dec, lst)
        alt_minus, az_minus = utils.radec2altaz(ra-5.0, dec, lst)
        self.assertAlmostEqual(alt_plus, alt_minus)

if __name__ == '__main__':
    unittest.main()
