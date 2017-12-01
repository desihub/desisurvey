import unittest

import numpy as np

import astropy.units as u

from desisurvey.etc import *


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
            self.assertGreater(t, 0.0 * u.s)

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

    def test_twilight(self):
        # Sun below the horizon
        x = twilight_exposure_factor(sun_alt=-20., sun_daz=0., airmass=1.2)
        self.assertAlmostEqual(x, 1.0, 4)

        # Exposure times increase as the dawn sun rises, for many daz values.
        for daz in (0, 90, 270, -10):
            efac = np.empty(7)
            for i, alt in enumerate(np.linspace(-18, -12, 7)):
                efac[i] = twilight_exposure_factor(
                    sun_alt=alt, sun_daz=daz, airmass=1.)
            self.assertTrue(np.all(efac >= 1.0))
            self.assertTrue(np.all(efac < 1.2))
            self.assertTrue(np.all(np.diff(efac) > 0))

    def test_moon(self):
        #- Moon below horizon
        x = moon_exposure_factor(
            moon_frac=0.5, moon_sep=60, moon_alt=-30, airmass=1.2)
        self.assertAlmostEqual(x, 1.0, 4)

        #- New moon above horizon still has some impact
        x = moon_exposure_factor(
            moon_frac=0.0, moon_sep=60, moon_alt=30, airmass=1.2)
        self.assertGreater(x, 1.0)
        self.assertLess(x, 1.1)

        #- Partial moon takes more time
        x1 = moon_exposure_factor(
            moon_frac=0.1, moon_sep=60, moon_alt=30, airmass=1.2)
        x2 = moon_exposure_factor(
            moon_frac=0.2, moon_sep=60, moon_alt=30, airmass=1.2)
        self.assertGreater(x1, 1.0)
        self.assertGreater(x2, x1)

        #- Closer moon takes more time
        x1 = moon_exposure_factor(
            moon_frac=0.5, moon_sep=60, moon_alt=30, airmass=1.2)
        x2 = moon_exposure_factor(
            moon_frac=0.5, moon_sep=30, moon_alt=30, airmass=1.2)
        self.assertGreater(x2, x1)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
