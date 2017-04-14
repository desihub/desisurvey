import unittest
import numpy as np
from desisurvey import utils

class TestUtils(unittest.TestCase):

    def setUp(self):
        pass

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
