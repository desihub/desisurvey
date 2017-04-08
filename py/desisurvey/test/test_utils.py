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
        # Test against astropy SkyCoord and AltAz
        LST = 91.74715323341904  # 2017-04-15 00:00:00 at KPNO according to astropy.time

        ra, dec, lst = LST, 60, LST
        alt, az = utils.radec2altaz(ra, dec, lst)
        self.assertAlmostEqual(alt, 61.96451022, 0)
        self.assertAlmostEqual(az, 0.406365, 0) # I don't get this value, it should be 0

        ra, dec, lst = 150, 20, LST
        alt, az = utils.radec2altaz(ra, dec, lst)
        self.assertAlmostEqual(alt, 36.66732297, 0)
        self.assertAlmostEqual(az, 87.93562759, 0)

        ra, dec, lst = 0, 45, LST
        alt, az = utils.radec2altaz(ra, dec, lst)
        self.assertAlmostEqual(alt, 21.03485612, 0)
        self.assertAlmostEqual(az, 310.87827568, 0)
        

if __name__ == '__main__':
    unittest.main()
