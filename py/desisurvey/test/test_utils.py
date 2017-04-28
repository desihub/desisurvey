import unittest
import datetime
import os

import numpy as np

import pytz

import astropy.time
import astropy.coordinates
import astropy.io
import astropy.utils.data
import astropy.units as u

from desisurvey import utils, config


class TestUtils(unittest.TestCase):

    def setUp(self):
        # Configure a CSV reader for the Horizons output format.
        csv_reader = astropy.io.ascii.Csv()
        csv_reader.header.comment = r'[^ ]'
        csv_reader.data.start_line = 35
        csv_reader.data.end_line = 203
        # Read moon ephemerides for the first week of 2020.
        path = astropy.utils.data._find_pkg_data_path(
            os.path.join('data', 'horizons_2020_week1_moon.csv'),
            package='desisurvey')
        self.table = csv_reader.read(path)
        # Horizons CSV file has a trailing comma on each line.
        self.table.remove_column('col10')
        # Use more convenient column names.
        names = ('date', 'jd', 'sun', 'moon', 'ra', 'dec',
                 'az', 'alt', 'lst', 'frac')
        for old_name, new_name in zip(self.table.colnames, names):
            self.table[old_name].name = new_name

    def test_get_overhead(self):
        """Sanity checks on overhead time calculations"""
        c = config.Configuration()
        p0 = astropy.coordinates.SkyCoord(ra=300 * u.deg, dec=10 * u.deg)
        # Move with no readout and no slew only has focus overhead.
        self.assertEqual(utils.get_overhead_time(p0, p0, False),
                         c.focus_time())
        # Move with readout and no slew has focus and overhead in parallel.
        self.assertEqual(utils.get_overhead_time(p0, p0, True),
                         max(c.focus_time(), c.readout_time()))
        # Overhead with slew same when dRA == dDEC.
        for delta in (1, 45, 70):
            ra = p0.ra + [+delta, -delta, 0, 0] * u.deg
            dec = p0.dec + [0, 0, +delta, -delta] * u.deg
            p1 = astropy.coordinates.SkyCoord(ra=ra, dec=dec)
            dt = utils.get_overhead_time(p0, p1)
            self.assertTrue(dt.shape == (4,))
            self.assertEqual(dt[0], dt[1])
            self.assertEqual(dt[0], dt[2])
            self.assertEqual(dt[0], dt[3])

    def test_get_observer_to_sky(self):
        """Check (alt,az) -> (ra,dec) against JPL Horizons"""
        ra, dec = self.table['ra'], self.table['dec']
        alt, az = self.table['alt'], self.table['az']
        when = astropy.time.Time(self.table['jd'], format='jd')
        obs = utils.get_observer(when, alt * u.deg, az * u.deg)
        obs_sky = obs.transform_to(astropy.coordinates.ICRS)
        true_sky = astropy.coordinates.ICRS(ra=ra * u.deg, dec=dec * u.deg)
        sep = true_sky.separation(obs_sky).to(u.arcsec).value
        # Opening angle between true and observed (ra,dec) unit vectors
        # must be within 30 arcsec.
        self.assertTrue(np.max(np.fabs(sep)) < 30)

    def test_get_observer_from_sky(self):
        """Check (alt,az) -> (ra,dec) against JPL Horizons"""
        ra, dec = self.table['ra'], self.table['dec']
        alt, az = self.table['alt'], self.table['az']
        when = astropy.time.Time(self.table['jd'], format='jd')
        sky = astropy.coordinates.ICRS(ra=ra * u.deg, dec=dec * u.deg)
        true_altaz = utils.get_observer(when, alt * u.deg, az * u.deg)
        obs_altaz = sky.transform_to(utils.get_observer(when))
        sep = true_altaz.separation(obs_altaz).to(u.arcsec).value
        # Opening angle between true and observed (alt,az) unit vectors
        # must be within 30 arcsec.
        self.assertTrue(np.max(np.fabs(sep)) < 30)

    def test_get_observer_args(self):
        """Must provide both alt and az to get_observer()"""
        t = astropy.time.Time('2020-01-01')
        with self.assertRaises(ValueError):
            utils.get_observer(t, alt=0 * u.deg)
        with self.assertRaises(ValueError):
            utils.get_observer(t, az=0 * u.deg)

    def test_get_observer_units(self):
        """Alt,az must have angular units for get_observer()"""
        t = astropy.time.Time('2020-01-01')
        with self.assertRaises(TypeError):
            utils.get_observer(t, alt=1, az=1)
        with self.assertRaises(TypeError):
            utils.get_observer(t, alt=1 * u.m, az=1 * u.m)

    def test_get_observer_time(self):
        """Must pass arg convertible to astropy Time to get_observer()"""
        utils.get_observer('2020-01-01')
        utils.get_observer(datetime.datetime(2020, 1, 1, 0, 0))
        with self.assertRaises(ValueError):
            utils.get_observer(datetime.date(2020, 1, 1))
        with self.assertRaises(ValueError):
            utils.get_observer(50000.)

    def test_get_location(self):
        """Check for sensible coordinates"""
        loc = utils.get_location()
        self.assertTrue(np.fabs(loc.latitude.to(u.deg).value - 32.0) < 0.1)
        self.assertTrue(np.fabs(loc.longitude.to(u.deg).value + 111.6) < 0.1)
        self.assertTrue(np.fabs(loc.height.to(u.m).value - 2120) < 0.1)

    def test_get_location_cache(self):
        """Test location object caching"""
        self.assertEqual(id(utils.get_location()), id(utils.get_location()))

    def test_get_date(self):
        """Test date conversions"""
        # Start at local noon
        tz = pytz.timezone(config.Configuration().location.timezone())
        start = datetime.datetime(2019, 8, 23, 12)
        one_day = datetime.timedelta(days=1)
        one_hour = datetime.timedelta(hours=1)
        for day_offset in range(500):
            noon = start + one_day * day_offset
            local_noon = tz.localize(noon)
            answer = noon.date()
            # date -> date
            self.assertEqual(utils.get_date(answer), answer)
            # YYYY-MM-DD -> datetime -> date
            self.assertEqual(utils.get_date(str(answer)), answer)
            # Test specifications with time of day included.
            for hour_offset in (-1, 0, +1):
                time = noon + hour_offset * one_hour
                local_time = tz.localize(time)
                # unlocalized datetime -> date. UTC noon is 3am local,
                # so all hour offsets refer to the previous day.
                self.assertEqual(utils.get_date(time), answer - one_day)
                # The answer for localized datetimes depends on the hour offset.
                local_answer = answer - one_day if hour_offset < 0 else answer
                # localized datetime -> date
                self.assertEqual(utils.get_date(local_time), local_answer)
                # astropy time -> datetime -> date.
                t = astropy.time.Time(local_time)
                self.assertEqual(utils.get_date(t), local_answer)
                # MJD -> astropy time -> datetime -> date
                self.assertEqual(utils.get_date(t.mjd), local_answer)

    def test_monsoon(self):
        """Monsoon based on (month, day) comparisons"""
        for year in range(2019, 2025):
            self.assertFalse(utils.is_monsoon(datetime.date(year, 7, 12)))
            self.assertTrue(utils.is_monsoon(datetime.date(year, 7, 13)))
            self.assertTrue(utils.is_monsoon(datetime.date(year, 8, 26)))
            self.assertFalse(utils.is_monsoon(datetime.date(year, 8, 27)))

    def test_local_noon(self):
        """The telescope is 7 hours behind of UTC during winter and summer."""
        for month in (1, 7):
            day = datetime.date(2019, month, 1)
            noon = utils.local_noon_on_date(day)
            self.assertEqual(noon.datetime.date(), day)
            self.assertEqual(noon.datetime.time(), datetime.time(hour=12 + 7))

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

        LST = 168.86210588900758  # 2000-01-01 12:00:00 at KPNO according to astropy.time
        ra, dec, lst = LST, 60, LST
        alt, az = utils.radec2altaz(ra, dec, lst)
        self.assertAlmostEqual(alt, 61.96710605261274, 2) # Values from Astropy SkyCoords
        self.assertAlmostEqual(az, 0.0011510242215743817, 2)

        # Value close to zenith
        ra, dec, lst = LST, 31.965, LST
        alt_z, az_z = utils.radec2altaz(ra, dec, lst)
        alt_plus, az_minus = utils.radec2altaz(ra, dec+5.0, lst)
        alt_minus, az_minus = utils.radec2altaz(ra, dec-5.0, lst)
        self.assertAlmostEqual(alt_plus, alt_minus, 2)
        alt_plus, az_minus = utils.radec2altaz(ra+5.0, dec, lst)
        alt_minus, az_minus = utils.radec2altaz(ra-5.0, dec, lst)
        self.assertAlmostEqual(alt_plus, alt_minus)


if __name__ == '__main__':
    unittest.main()
