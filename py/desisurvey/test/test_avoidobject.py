import unittest
from astropy.time import Time
from desisurvey.avoidobject import avoidObject,moonLoc

class TestavoidObject(unittest.TestCase):

    def setup(self):
        self.ra=60.0 #RA to test degrees
        self.dec=40.0 #DEC to test degrees
        self.time=Time('2020-01-10 06:00:00.00', scale='utc', format='isot') #Time to test (UTC)
    def test_output(self):
        """
        Test avoidObject output
        """
        avoid_object = avoidObject(self.time,self.ra,self.dec)
        self.assertIsInstance(avoid_object, bool)
class TestmoonLoc(unittest.TestCase):

    def setup(self):
        self.ra=60.0 #RA to test degrees
        self.dec=40.0 #DEC to test degrees
        self.time=Time('2020-01-10 06:00:00.00', scale='utc', format='isot') #Time to test (UTC)
    def test_output(self):
        """
        Test moonLoc output
        """
        moonloc = moonLoc(self.time,self.ra,self.dec)
        self.assertGreaterEqual(moonloc[0], -180.0)
        self.assertLess(moonloc[0], 180.0)
        self.assertGreaterEqual(moonloc[1], -90.0)
        self.assertLess(moonloc[1], 90.0)
        self.assertGreaterEqual(moonloc[2], 0.0)
        self.assertLess(moonloc[2], 360.0)
if __name__ == '__main__':
    unittest.main()
