import unittest
from astropy.time import Time
from desisurvey.avoidobject import avoidObject,moonLoc

class TestavoidObject(unittest.TestCase):

    def setUp(self):
        self.ra=60.0 #RA to test degrees
        self.dec=40.0 #DEC to test degrees
        self.time=Time('2020-01-10T06:00:00.00', scale='utc', format='isot') #Time to test (UTC)

    def test_output(self):
        """
        Test avoidObject output
        """
        avoid_object = avoidObject(self.time,self.ra,self.dec)
        self.assertIsInstance(avoid_object, bool)

class TestmoonLoc(unittest.TestCase):

    def setUp(self):
        self.ra=60.0 #RA to test degrees
        self.dec=40.0 #DEC to test degrees
        self.time=Time('2020-01-10T06:00:00.00', scale='utc', format='isot') #Time to test (UTC)

    def test_output(self):
        """
        Test moonLoc output
        """
        dist, alt, az = moonLoc(self.time,self.ra,self.dec)
        self.assertGreaterEqual(dist, -180.0)
        self.assertLess(dist, 180.0)

        self.assertGreaterEqual(alt, -90.0)
        self.assertLess(alt, 90.0)

        self.assertGreaterEqual(az, 0.0)
        self.assertLess(az, 360.0)

if __name__ == '__main__':
    unittest.main()
