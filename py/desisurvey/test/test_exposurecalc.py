import unittest

import numpy as np
from desisurvey.exposurecalc import expTimeEstimator, airMassCalculator, moonExposureTimeFactor

class TestExpCalc(unittest.TestCase):

    def setUp(self):
        pass

    def test_exptime(self):
        seeing = 1.1
        transparency = 1.0
        airmass = 1.2
        program = 'DARK'
        ebmv = 0.01
        sn2 = 10
        moonFrac = 0.05
        moonDist = 70
        moonAlt = 10
        for program in ['DARK', 'GRAY', 'BRIGHT']:
            t = expTimeEstimator(seeing, transparency, airmass, program, ebmv, sn2,
                                 moonFrac, moonDist, moonAlt)
            self.assertGreater(t, 0.0)

        program = 'DARK'

        #- Worse seeing = longer exposures
        seeing = 1.0
        t1 = expTimeEstimator(seeing, transparency, airmass, program, ebmv, sn2,
                              moonFrac, moonDist, moonAlt)
        seeing=1.2
        t2 = expTimeEstimator(seeing, transparency, airmass, program, ebmv, sn2,
                              moonFrac, moonDist, moonAlt)
        self.assertGreater(t2, t1)

        #- Worse higher airmass = longer exposures
        t1 = expTimeEstimator(seeing, transparency, airmass, program, ebmv, sn2,
                              moonFrac, moonDist, moonAlt)
        t2 = expTimeEstimator(seeing, transparency, airmass*1.2, program, ebmv, sn2,
                              moonFrac, moonDist, moonAlt)
        self.assertGreater(t2, t1)

    def test_airmass(self):
        ra, dec = 10.5, 31.9640  #- random RA, Mayall dec
        airmass = airMassCalculator(ra, dec, lst=ra, return_altaz=False)
        self.assertTrue(isinstance(airmass, (float, np.float)))
        self.assertAlmostEqual(airmass, 1.0, 4)

        airmass, alt, az = airMassCalculator(ra, dec, lst=ra, return_altaz=True)
        self.assertAlmostEqual(airmass, 1.0, 4)
        self.assertAlmostEqual(alt, 90.0, 4)

        n = 30
        airmass = airMassCalculator(ra+np.arange(n), dec, lst=ra, return_altaz=False)
        self.assertEqual(len(airmass), n)
        self.assertTrue(np.all(np.diff(airmass) > 0))

    def test_moon(self):
        #- Moon below horizon
        x = moonExposureTimeFactor(moonFrac=0.5, moonDist=60, moonAlt=-30)
        self.assertAlmostEqual(x, 1.0, 4)

        #- New moon above horizon still has some impact
        x = moonExposureTimeFactor(moonFrac=0.0, moonDist=60, moonAlt=30)
        self.assertGreater(x, 1.0)
        self.assertLess(x, 1.1)

        #- Partial moon takes more time
        x1 = moonExposureTimeFactor(moonFrac=0.1, moonDist=60, moonAlt=30)
        x2 = moonExposureTimeFactor(moonFrac=0.2, moonDist=60, moonAlt=30)
        self.assertGreater(x1, 1.0)
        self.assertGreater(x2, x1)

        #- Closer moon takes more time
        x1 = moonExposureTimeFactor(moonFrac=0.5, moonDist=60, moonAlt=30)
        x2 = moonExposureTimeFactor(moonFrac=0.5, moonDist=30, moonAlt=30)
        self.assertGreater(x2, x1)

if __name__ == '__main__':
    unittest.main()
