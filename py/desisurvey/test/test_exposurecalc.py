import unittest

import numpy as np

from desisurvey.exposurecalc import exposure_time, moon_exposure_factor


class TestExpCalc(unittest.TestCase):

    def setUp(self):
        pass

    def test_exptime(self):
        seeing = 1.1
        transparency = 1.0
        airmass = 1.2
        program = 'DARK'
        EBV = 0.01
        moon_frac = 0.05
        moon_sep = 70
        moon_alt = 10
        for program in ['DARK', 'GRAY', 'BRIGHT']:
            t = exposure_time(program, seeing, transparency, airmass, EBV,
                                 moon_frac, moon_sep, moon_alt)
            self.assertGreater(t, 0.0)

        program = 'DARK'

        #- Worse seeing = longer exposures
        seeing = 1.0
        t1 = exposure_time(program, seeing, transparency, airmass, EBV,
                           moon_frac, moon_sep, moon_alt)
        seeing=1.2
        t2 = exposure_time(program, seeing, transparency, airmass, EBV,
                           moon_frac, moon_sep, moon_alt)
        self.assertGreater(t2, t1)

        #- Worse higher airmass = longer exposures
        t1 = exposure_time(program, seeing, transparency, airmass, EBV,
                           moon_frac, moon_sep, moon_alt)
        t2 = exposure_time(program, seeing, transparency, airmass*1.2,
                           EBV, moon_frac, moon_sep, moon_alt)
        self.assertGreater(t2, t1)

    def test_moon(self):
        #- Moon below horizon
        x = moon_exposure_factor(moon_frac=0.5, moon_sep=60, moon_alt=-30)
        self.assertAlmostEqual(x, 1.0, 4)

        #- New moon above horizon still has some impact
        x = moon_exposure_factor(moon_frac=0.0, moon_sep=60, moon_alt=30)
        self.assertGreater(x, 1.0)
        self.assertLess(x, 1.1)

        #- Partial moon takes more time
        x1 = moon_exposure_factor(moon_frac=0.1, moon_sep=60, moon_alt=30)
        x2 = moon_exposure_factor(moon_frac=0.2, moon_sep=60, moon_alt=30)
        self.assertGreater(x1, 1.0)
        self.assertGreater(x2, x1)

        #- Closer moon takes more time
        x1 = moon_exposure_factor(moon_frac=0.5, moon_sep=60, moon_alt=30)
        x2 = moon_exposure_factor(moon_frac=0.5, moon_sep=30, moon_alt=30)
        self.assertGreater(x2, x1)


if __name__ == '__main__':
    unittest.main()
