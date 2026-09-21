import unittest
import tempfile
import shutil
import datetime
import os

import numpy as np

import pytz

import astropy.time
import astropy.coordinates
import astropy.io
import astropy.utils.data
import astropy.units as u

from desisurvey.test.base import Tester, read_horizons_moon_ephem
import desisurvey.utils as utils
import desisurvey.config as config


class TestUtils(Tester):

    def setUp(self):
        # Read moon ephemerides for the first week of 2020.
        self.table = read_horizons_moon_ephem()

    def tearDown(self):
        pass

    def test_get_observer_to_sky(self):
        """Check (alt,az) -> (ra,dec) against JPL Horizons"""
        ra, dec = self.table['ra'], self.table['dec']
        alt, az = self.table['alt'], self.table['az']
        when = astropy.time.Time(self.table['jd'], format='jd')
        obs = utils.get_observer(when, alt * u.deg, az * u.deg)
        obs_sky = obs.transform_to(astropy.coordinates.ICRS())
        true_sky = astropy.coordinates.ICRS(ra=ra * u.deg, dec=dec * u.deg)
        sep = true_sky.separation(obs_sky).to(u.arcsec).value
        # Opening angle between true and observed (ra,dec) unit vectors
        # must be within 40 arcsec.
        self.assertTrue(np.max(np.fabs(sep)) < 40)

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
        # must be within 40 arcsec.
        self.assertTrue(np.max(np.fabs(sep)) < 40)

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
        with self.assertRaises(u.UnitsError):
            utils.get_observer(t, alt=1 * u.m, az=1 * u.m)

    def test_get_observer_time(self):
        """Must pass arg convertible to astropy Time to get_observer()"""
        utils.get_observer('2020-01-01')
        utils.get_observer(datetime.datetime(2020, 1, 1, 0, 0))
        with self.assertRaises(ValueError):
            utils.get_observer(datetime.date(2020, 1, 1))
        with self.assertRaises(ValueError):
            utils.get_observer(50000.)

    def test_zenith_airmass(self):
        """Airmass values monotically increase with zenith angle"""
        Z = np.arange(90) * np.pi / 180.
        cosZ = np.cos(Z)
        X = utils.cos_zenith_to_airmass(cosZ)
        self.assertEqual(Z.shape, X.shape)
        self.assertTrue(np.all(np.diff(X) > 0))

    def test_cosz_range(self):
        """cos(z) must be between -1 and +1"""
        ha = np.arange(-100, +400) * u.deg
        for dec in [-10, 20, 40] * u.deg:
            cosz = utils.cos_zenith(ha, dec)
            self.assertTrue(np.all(cosz >= -1))
            self.assertTrue(np.all(cosz <= +1))

    def test_cosz_one(self):
        """cos(z) == 1 when ha=0 and dec=lat"""
        ha = 0 * u.hourangle
        for dec in [-10, 20, 40] * u.deg:
            cosz = utils.cos_zenith(ha, dec, latitude=dec)
            self.assertTrue(np.allclose(cosz, 1.))

    def test_airmass_scalar(self):
        """Scalar input returns scalar output"""
        X = utils.cos_zenith_to_airmass(1.)
        self.assertEqual(X.shape, ())

    def test_airmass_clip(self):
        """cosZ values < 0 are clipped"""
        self.assertAlmostEqual(
            utils.cos_zenith_to_airmass(0),
            utils.cos_zenith_to_airmass(-1))

    def test_get_airmass_lowest(self):
        """The lowest airmass occurs when dec=latitude"""
        t = astropy.time.Time('2020-01-01')
        ra = np.arange(360) * u.deg
        dec = utils.get_location().lat
        Xinv = 1 / utils.get_airmass(t, ra, dec)
        self.assertTrue(np.max(Xinv) > 0.999)
        dec = dec + 30 * u.deg
        Xinv = 1 / utils.get_airmass(t, ra, dec)
        self.assertTrue(np.max(Xinv) < 0.9)

    def test_get_airmass_always_visible(self):
        """An object at DEC=70d is always visible"""
        t = astropy.time.Time('2020-01-01')
        ra = np.arange(360) * u.deg
        dec = 70 * u.deg
        Xinv = 1 / utils.get_airmass(t, ra, dec)
        self.assertTrue(np.all(0.2 < Xinv) and np.all(Xinv < 0.8))

    def test_get_location(self):
        """Check for sensible coordinates"""
        loc = utils.get_location()
        self.assertTrue(np.fabs(loc.lat.to(u.deg).value - 32.0) < 0.1)
        self.assertTrue(np.fabs(loc.lon.to(u.deg).value + 111.6) < 0.1)
        self.assertTrue(np.fabs(loc.height.to(u.m).value - 2120) < 0.1)

    def test_get_location_cache(self):
        """Test location object caching"""
        self.assertEqual(id(utils.get_location()), id(utils.get_location()))

    def test_monsoon(self):
        """Test nominal monsoon shutdown starts/stops on Mon/Fri each year"""
        config = desisurvey.config.Configuration()
        for key in config.monsoon.keys:
            node = getattr(config.monsoon, key)
            self.assertTrue(node.start().weekday() == 0)
            self.assertTrue(node.stop().weekday() == 4)
            self.assertTrue((node.stop() - node.start()).days == 18)

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

    def test_day_number(self):
        """Number of days since start of survey"""
        first = config.Configuration().first_day()
        self.assertEqual(0, utils.day_number(first))
        self.assertEqual(
            100, utils.day_number(first + datetime.timedelta(days=100)))

    def test_monsoon(self):
        """Sanity checks on monsoon calculations"""
        for year in range(2020, 2025):
            self.assertFalse(utils.is_monsoon(datetime.date(year, 1, 1)))
            self.assertFalse(utils.is_monsoon(datetime.date(year, 7, 11)))
            self.assertFalse(utils.is_monsoon(datetime.date(year, 8, 20)))
            self.assertFalse(utils.is_monsoon(datetime.date(year, 12, 31)))
        self.assertTrue(utils.is_monsoon(datetime.date(2020, 8, 1)))
        self.assertFalse(utils.is_monsoon(datetime.date(2021, 8, 1)))
        self.assertTrue(utils.is_monsoon(datetime.date(2022, 8, 1)))
        self.assertTrue(utils.is_monsoon(datetime.date(2023, 8, 1)))

    def test_local_noon(self):
        """The telescope is 7 hours behind of UTC during winter and summer."""
        for month in (1, 7):
            day = datetime.date(2019, month, 1)
            noon = utils.local_noon_on_date(day)
            self.assertEqual(noon.datetime.date(), day)
            self.assertEqual(noon.datetime.time(), datetime.time(hour=12 + 7))

    def test_separation_matrix(self):
        ra = [0, 45, 90, 180, 270]
        dec = [-90, -45, 0, 45, 90]
        sep0 = utils.separation_matrix(ra, dec, ra, dec)
        for n1 in range(1, 5):
            for n2 in range(1, 5):
                sep = utils.separation_matrix(
                    ra[:n1], dec[:n1], ra[:n2], dec[:n2])
                assert sep.shape == (n1, n2)
                assert np.allclose(sep0[:n1, :n2], sep)
        assert np.allclose(utils.separation_matrix([0], [0], [0], [0]), 0.)
        assert np.allclose(utils.separation_matrix([90], [0], [90], [0]), 0.)
        assert np.allclose(utils.separation_matrix([0], [0], [0], [90]), 90.)
        assert np.allclose(utils.separation_matrix([90], [0], [90], [90]), 90.)
        assert np.allclose(utils.separation_matrix([0], [0], [0], [-45]), 45.)
        assert np.allclose(utils.separation_matrix([330], [0], [30], [0]), 60.)

    def test_get_ha_limit_spec(self):
        cfg = config.Configuration()
        # Restore the default even if this test fails, since it is global.
        self.addCleanup(cfg.max_hour_angle_by_dec.set_value, '')
        # Disabled by default, and an empty value reads as no limit rather
        # than as an empty specification.
        cfg.max_hour_angle_by_dec.set_value('')
        assert utils.get_ha_limit_spec(cfg) is None
        spec = '-30:0.5, 0:3.0, 30:4.0'
        cfg.max_hour_angle_by_dec.set_value(spec)
        assert utils.get_ha_limit_spec(cfg) == spec
        # The singleton is used when no configuration is passed.
        assert utils.get_ha_limit_spec() == spec
        # A version of the package predating the parameter must read as no
        # limit rather than raise.
        class NoSuchParameter:
            pass
        assert utils.get_ha_limit_spec(NoSuchParameter()) is None

    def test_parse_ha_limit_spec(self):
        dec, ha = utils.parse_ha_limit_spec('-30:0.5, 0:3.0, 30:4.0')
        assert np.allclose(dec, [-30, 0, 30])
        # Limits are converted from hours to degrees.
        assert np.allclose(ha, [7.5, 45., 60.])
        # Nodes need not be given in order.
        dec2, ha2 = utils.parse_ha_limit_spec('30:4.0, -30:0.5, 0:3.0')
        assert np.allclose(dec, dec2)
        assert np.allclose(ha, ha2)
        # Extra whitespace and a trailing separator are tolerated.
        dec3, ha3 = utils.parse_ha_limit_spec('  -30 : 0.5 ,0:3.0,30:4.0,  ')
        assert np.allclose(dec, dec3)
        assert np.allclose(ha, ha3)
        # A malformed specification must raise rather than silently disable
        # the limit.
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('')
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('-30:0.5')
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('-30:0.5, junk')
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('-30:0.5, 0:x')
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('-30:0.5, 0:3.0:9')

    def test_parse_ha_limit_spec_rejects_unusable(self):
        # A specification that cannot be used must be rejected rather than
        # silently reinterpreted.
        # Declinations that cannot exist, e.g. a transposed node.
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('0.5:-30, 3.0:0')
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('-30:0.5, 120:3.0')
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('-30:0.5, nan:3.0')
        # A limit of zero deselects every tile at that declination.
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('-30:0, 0:3.0')
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('-30:-0.5, 0:3.0')
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('-30:0.5, 0:inf')
        # Limits are in hours, so a value that looks like degrees is a
        # mistake rather than a very loose limit.
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('-30:7.5, 0:45')
        # 12 hr is the largest meaningful limit, and is allowed.
        dec, ha = utils.parse_ha_limit_spec('-30:0.5, 0:12')
        assert np.allclose(ha, [7.5, 180.])
        # A repeated declination is resolved silently by the interpolation,
        # with the winner depending on the sort, so refuse it.
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('0:1.0, 0:3.0')
        with self.assertRaises(ValueError):
            utils.parse_ha_limit_spec('-30:0.5, 0:3.0, -30:2.0')

    def test_parse_ha_limit_spec_warns_below_floor(self):
        # desisurvey.tiles applies this limit after its half hour window
        # floor, so a smaller node overrides that floor.  Legitimate, but
        # worth a warning since it is easy to do by accident.
        with self.assertLogs(level='WARNING') as caught:
            dec, ha = utils.parse_ha_limit_spec('-30:0.25, 0:3.0')
        assert np.allclose(ha, [3.75, 45.])
        assert any('window floor' in m for m in caught.output)

    def test_max_ha_by_dec(self):
        spec = '-30:0.5, -20:1.5, 0:3.0, 40:4.0, 60:5.0'
        # Node values are reproduced exactly.
        assert np.allclose(utils.max_ha_by_dec([-30, -20, 0, 40, 60], spec),
                           [7.5, 22.5, 45., 60., 75.])
        # Interpolation is linear in declination, on the limit in hours.
        assert np.allclose(utils.max_ha_by_dec(-25, spec), 15.)
        assert np.allclose(utils.max_ha_by_dec(-10, spec), 33.75)
        # Outside the tabulated range the end values are held.
        assert np.allclose(utils.max_ha_by_dec([-90, -40], spec), 7.5)
        assert np.allclose(utils.max_ha_by_dec([70, 90], spec), 75.)
        # A ceiling can only tighten the limit, never loosen it.
        assert np.allclose(utils.max_ha_by_dec(60, spec, ceiling=75.), 75.)
        assert np.allclose(utils.max_ha_by_dec(60, spec, ceiling=50.), 50.)
        assert np.allclose(utils.max_ha_by_dec(-30, spec, ceiling=75.), 7.5)
        # Shape is preserved.
        dec = np.array([[-30., 0.], [40., 60.]])
        assert utils.max_ha_by_dec(dec, spec).shape == dec.shape
