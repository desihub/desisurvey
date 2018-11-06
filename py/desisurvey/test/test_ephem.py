import unittest
import os
import datetime
import tempfile
import shutil

import numpy as np

from astropy.time import Time
from astropy.coordinates import ICRS, AltAz
import astropy.units as u
import astropy.io

import desisurvey.config
from desisurvey.ephem import Ephemerides, get_grid, get_object_interpolator
from desisurvey.utils import get_date, get_location, freeze_iers


class TestEphemerides(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory.
        cls.tmpdir = tempfile.mkdtemp()
        # Write output files to this temporary directory.
        config = desisurvey.config.Configuration()
        config.set_output_path(cls.tmpdir)
        # Configure a CSV reader for the Horizons output format.
        csv_reader = astropy.io.ascii.Csv()
        csv_reader.header.comment = r'[^ ]'
        csv_reader.data.start_line = 35
        csv_reader.data.end_line = 203
        # Read moon ephemerides for the first week of 2020.
        path = astropy.utils.data._find_pkg_data_path(
            os.path.join('data', 'horizons_2020_week1_moon.csv'),
            package='desisurvey')
        cls.table = csv_reader.read(path)
        # Horizons CSV file has a trailing comma on each line.
        cls.table.remove_column('col10')
        # Use more convenient column names.
        names = ('date', 'jd', 'sun', 'moon', 'ra', 'dec',
                 'az', 'alt', 'lst', 'frac')
        for old_name, new_name in zip(cls.table.colnames, names):
            cls.table[old_name].name = new_name

    @classmethod
    def tearDownClass(cls):
        # Remove the directory after the test.
        shutil.rmtree(cls.tmpdir)
        # Reset our configuration.
        desisurvey.config.Configuration.reset()

    def test_getephem(self):
        """Tabulate one month of ephemerides"""
        # Free IERS to avoid noisy warnings.
        freeze_iers()
        start = datetime.date(2019, 9, 1)
        stop = datetime.date(2019, 10, 1)
        ephem = Ephemerides(start, stop, use_cache=False, write_cache=False)
        self.assertEqual(ephem.num_nights, (ephem.stop_date - ephem.start_date).days)

        etable = ephem._table
        self.assertEqual(len(etable), 30)
        self.assertTrue(np.all(etable['dusk'] > etable['noon']))
        self.assertTrue(np.all(etable['dawn'] > etable['dusk']))
        self.assertTrue(np.all(etable['dusk'] > etable['brightdusk']))
        self.assertTrue(np.all(etable['dawn'] < etable['brightdawn']))
        self.assertGreater(np.max(etable['moon_illum_frac']), 0.99)
        self.assertLessEqual(np.max(etable['moon_illum_frac']), 1.0)
        self.assertLess(np.min(etable['moon_illum_frac']), 0.01)
        self.assertGreaterEqual(np.min(etable['moon_illum_frac']), 0.00)
        self.assertTrue(np.all(etable['moonrise'] < etable['moonset']))

        hrs1 = get_program_hours(ephem, include_twilight=True).sum(axis=1)
        hrs2 = get_program_hours(ephem, include_twilight=False).sum(axis=1)
        hrs3 = get_program_hours(ephem, include_twilight=True, include_full_moon=True).sum(axis=1)
        self.assertEqual(hrs1[0], hrs2[0])
        self.assertEqual(hrs1[1], hrs2[1])
        self.assertGreater(hrs1[2], hrs2[2])
        self.assertLess(hrs1[0], hrs3[0])
        self.assertEqual(hrs1[1], hrs3[1])
        self.assertLess(hrs1[2], hrs3[2])

        for i in range(ephem.num_nights):

            x = ephem.get_row(i)
            date = Time(x['noon'], format='mjd').datetime.date()
            night = date.strftime('%Y%m%d')
            for key in [
                    'brightdusk', 'brightdawn',
                    'dusk', 'dawn',
                ]:
                #- AZ local time
                localtime = Time(x[key], format='mjd') - 7*u.hour
                #- YEARMMDD of sunset for that time
                yearmmdd = (localtime - 12*u.hour).to_datetime().strftime('%Y%m%d')
                msg = '{} != {} for {}={}'.format(night, yearmmdd, key, x[key])
                self.assertEqual(night, yearmmdd, msg)

    def test_full_moon(self):
        """Verify that the full moon break in Sep-2019 occurs on days 10-16"""
        start = datetime.date(2019, 8, 1)
        stop = datetime.date(2019, 11, 1)
        ephem = Ephemerides(start, stop, use_cache=False, write_cache=False)
        for i in range(31, 60):
            night = start + datetime.timedelta(days=i)
            expected = (night >= datetime.date(2019, 9, 10)) and (night <= datetime.date(2019, 9, 16))
            self.assertTrue(ephem.is_full_moon(night) is expected)

    def test_full_moon_duration(self):
        """Verify full moon calculations for different durations"""
        # Free IERS to avoid noisy warnings.
        freeze_iers()
        start = datetime.date(2023, 6, 1)
        stop = datetime.date(2023, 9, 30)
        full_moon = datetime.date(2023, 8, 1)
        ephem = Ephemerides(start, stop, use_cache=False, write_cache=False)
        idx = ephem.get_night(full_moon, as_index=True)
        frac0 = ephem._table['moon_illum_frac'][idx - 14:idx + 15]
        for num_nights in range(1, 25):
            full = []
            for dt in range(-14, +15):
                night = full_moon + datetime.timedelta(days=dt)
                full.append(ephem.is_full_moon(night, num_nights=num_nights))
            self.assertTrue(np.count_nonzero(full) == num_nights)

    def test_monsoon(self):
        """Test nominal monsoon shutdown starts/stops on Mon/Fri each year"""
        config = desisurvey.config.Configuration()
        for key in config.monsoon.keys:
            node = getattr(config.monsoon, key)
            self.assertTrue(node.start().weekday() == 0)
            self.assertTrue(node.stop().weekday() == 4)
            self.assertTrue((node.stop() - node.start()).days == 18)

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
        ephem = Ephemerides(
            get_date('2019-12-31'), get_date('2020-02-02'),
            use_cache=False, write_cache=False)
        for i, jd in enumerate(self.table['jd']):
            t = Time(jd, format='jd')
            frac = ephem.get_moon_illuminated_fraction(t.mjd)
            truth = 1e-2 * self.table['frac'][i]
            self.assertTrue(abs(frac - truth) < 0.01)

    def test_moon_radec(self):
        """Verify moon (ra,dec) for first week of 2020"""
        ephem = Ephemerides(
            get_date('2019-12-31'), get_date('2020-02-02'),
            use_cache=False, write_cache=False)
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
        ephem = Ephemerides(
            get_date('2019-12-31'), get_date('2020-02-02'),
            use_cache=False, write_cache=False)
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


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
