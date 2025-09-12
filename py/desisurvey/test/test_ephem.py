import unittest
import os.path

import numpy as np

from astropy.time import Time
from astropy.coordinates import ICRS, AltAz
import astropy.units as u
import astropy.io

from desisurvey.test.base import Tester, read_horizons_moon_ephem
from desisurvey.ephem import get_ephem, get_grid, get_object_interpolator
from desisurvey.utils import get_location


class TestEphemerides(Tester):

    @classmethod
    def setUpClass(cls):
        super(TestEphemerides, cls).setUpClass()
        # Read moon ephemerides for the first week of 2020.
        cls.table = read_horizons_moon_ephem()

    def test_get_ephem(self):
        """Test memory and disk caching"""
        # Create and save to memory only
        id1 = id(get_ephem(write_cache=False))
        # Load from memory
        id2 = id(get_ephem())
        self.assertEqual(id1, id2)
        # Save to disk.
        id3 = id(get_ephem())
        self.assertEqual(id1, id3)
        # Clear memory cache.
        _ephem = None
        # Read from disk.
        id4 = id(get_ephem())
        self.assertEqual(id1, id4)

    def test_ephem_table(self):
        """Test basic table structure"""
        ephem = get_ephem()
        self.assertEqual(ephem.num_nights, (ephem.stop_date - ephem.start_date).days)

        self.assertEqual(id(ephem._table), id(ephem.table))

        etable = ephem._table
        self.assertEqual(len(etable), 59)
        self.assertTrue(np.all(etable['dusk'] > etable['noon']))
        self.assertTrue(np.all(etable['dawn'] > etable['dusk']))
        self.assertTrue(np.all(etable['dusk'] > etable['brightdusk']))
        self.assertTrue(np.all(etable['dawn'] < etable['brightdawn']))
        self.assertGreater(np.max(etable['moon_illum_frac']), 0.99)
        self.assertLessEqual(np.max(etable['moon_illum_frac']), 1.0)
        self.assertLess(np.min(etable['moon_illum_frac']), 0.01)
        self.assertGreaterEqual(np.min(etable['moon_illum_frac']), 0.00)
        self.assertTrue(np.all(etable['moonrise'] < etable['moonset']))

        hrs1 = ephem.get_program_hours(ephem.start_date, ephem.stop_date, include_twilight=True).sum(axis=1)
        hrs2 = ephem.get_program_hours(ephem.start_date, ephem.stop_date, include_twilight=False).sum(axis=1)
        hrs3 = ephem.get_program_hours(ephem.start_date, ephem.stop_date, include_twilight=True, include_full_moon=True).sum(axis=1)
        self.assertEqual(hrs1[0], hrs2[0])
        self.assertEqual(hrs1[1], hrs2[1])
        self.assertGreater(hrs1[2], hrs2[2])
        self.assertLess(hrs1[0], hrs3[0])
        self.assertEqual(hrs1[1], hrs3[1])
        self.assertLess(hrs1[2], hrs3[2])

    def test_lst_hours(self):
        """Test consistent LST and hourly schedules"""
        ephem = get_ephem()
        gen = np.random.RandomState(123)
        weather = gen.uniform(size=ephem.num_nights)
        for twilight in True, False:
            for full_moon in True, False:
                for monsoon in True, False:
                    lst_hist, lst_bins = ephem.get_available_lst(
                        ephem.start_date, ephem.stop_date, weather=weather,
                        include_monsoon=monsoon, include_full_moon=full_moon, include_twilight=twilight)
                    hrs = ephem.get_program_hours(ephem.start_date, ephem.stop_date,
                        include_monsoon=monsoon, include_full_moon=full_moon, include_twilight=twilight)
                    hrs_sum = (hrs * weather).sum(axis=1)
                    lst_sum = lst_hist.sum(axis=1) * 0.99726956583 # sidereal / solar hours
                    self.assertTrue(np.allclose(hrs_sum, lst_sum))

    def test_get_grid(self):
        """Verify grid calculations"""
        for step_size in (1 * u.min, 0.3 * u.hour):
            for night_start in (-6 * u.hour, -6.4 * u.hour):
                g = get_grid(step_size, night_start)
                self.assertTrue(g[0] == night_start.to(u.day).value)
                self.assertAlmostEqual(g[1] - g[0], step_size.to(u.day).value)
                self.assertAlmostEqual(g[-1] - g[0],
                                (len(g) - 1) * step_size.to(u.day).value)

    def test_moon_phase(self):
        """Verfify moon illuminated fraction for first week of 2020"""
        ephem = get_ephem()
        for i, jd in enumerate(self.table['jd']):
            t = Time(jd, format='jd')
            frac = ephem.get_moon_illuminated_fraction(t.mjd)
            truth = 1e-2 * self.table['frac'][i]
            self.assertTrue(abs(frac - truth) < 0.01)

    def test_moon_radec(self):
        """Verify moon (ra,dec) for first week of 2020"""
        ephem = get_ephem()
        for i, jd in enumerate(self.table['jd']):
            t = Time(jd, format='jd')
            night = ephem.get_night(t)
            f_moon = get_object_interpolator(night, 'moon', altaz=False)
            dec, ra = f_moon(t.mjd)
            truth = ICRS(ra=self.table['ra'][i] * u.deg,
                         dec=self.table['dec'][i] * u.deg)
            calc = ICRS(ra=ra * u.deg, dec=dec * u.deg)
            sep = truth.separation(calc)
            self.assertTrue(abs(sep.to(u.deg).value) < 0.3)

    def test_moon_altaz(self):
        """Verify moon (alt,az) for first week of 2020"""
        ephem = get_ephem()
        location = get_location()
        for i, jd in enumerate(self.table['jd']):
            t = Time(jd, format='jd')
            night = ephem.get_night(t)
            f_moon = get_object_interpolator(night, 'moon', altaz=True)
            alt, az = f_moon(t.mjd)
            truth = AltAz(alt=self.table['alt'][i] * u.deg,
                          az=self.table['az'][i] * u.deg,
                          obstime=t, location=location, pressure=0)
            calc = AltAz(alt=alt * u.deg, az=az * u.deg,
                         obstime=t, location=location, pressure=0)
            sep = truth.separation(calc)
            self.assertTrue(abs(sep.to(u.deg).value) < 0.3)
