import unittest
import os
import uuid

import numpy as np
from desisurvey.nightcal import getCalAll
from astropy.time import Time

class TestNightCal(unittest.TestCase):
    
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
            
    def test_getcal(self):
        start = Time('2019-09-01T00:00:00')
        end = Time('2019-10-01T00:00:00')
        ephem = getCalAll(start, end, use_cache=False)
        
        self.assertEqual(len(ephem), 31)
        self.assertTrue(np.all(ephem['MJDsunrise'] > ephem['MJDsunset']))
        self.assertTrue(np.all(ephem['MJDetwi'] > ephem['MJDe13twi']))
        self.assertTrue(np.all(ephem['MJDmtwi'] < ephem['MJDm13twi']))
        self.assertGreater(np.max(ephem['MoonFrac']), 0.99)
        self.assertLessEqual(np.max(ephem['MoonFrac']), 1.0)
        self.assertLess(np.min(ephem['MoonFrac']), 0.01)
        self.assertGreaterEqual(np.min(ephem['MoonFrac']), 0.00)
        self.assertTrue(np.all(ephem['MJDmoonrise'] < ephem['MJDmoonset']))
                
if __name__ == '__main__':
    unittest.main()
