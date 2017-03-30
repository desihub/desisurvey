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

    @unittest.expectedFailure
    def test_equ2gal_J2000(self):
        # http://ned.ipac.caltech.edu/cgi-bin/calc?in_csys=Equatorial&in_equinox=J2000.0&obs_epoch=2000.0&lon=1&lat=20&pa=0.0&out_csys=Galactic&out_equinox=J2000.0
        ra, dec = 15, 20
        l, b = utils.equ2gal_J2000(ra, dec)
        self.assertAlmostEqual(l, 125.67488266, 4)
        self.assertAlmostEqual(b, -42.82605200, 4)    

    def test_angsep(self):
        self.assertAlmostEqual(utils.angsep(0,0,10,0), 10.0)
        self.assertAlmostEqual(utils.angsep(0,0,0,10), 10.0)
        self.assertAlmostEqual(utils.angsep(20,0,10,0), 10.0)
        self.assertAlmostEqual(utils.angsep(0,20,0,10), 10.0)
        self.assertAlmostEqual(utils.angsep(60,70,60,50), 20.0)

if __name__ == '__main__':
    unittest.main()
