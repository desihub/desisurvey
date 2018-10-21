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

from desisurvey import utils, config


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory.
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        # Remove the directory after the test.
        shutil.rmtree(cls.tmpdir)

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

    def tearDown(self):
        utils._iers_is_frozen = False

    def test_update_iers_bad_ext(self):
        """Test save_name extension check"""
        save_name = os.path.join(self.tmpdir, 'iers.fits')
        with self.assertRaises(ValueError):
            utils.update_iers(save_name)

    def test_update_iers(self):
        """Test updating the IERS table.  Requires a network connection."""
        save_name = os.path.join(self.tmpdir, 'iers.ecsv')
        utils.update_iers(save_name)
        # Second write should overwrite original file.
        utils.update_iers(save_name)
        utils.freeze_iers(save_name)

    def test_freeze_iers(self):
        """Test freezing from package data/"""
        utils.freeze_iers()
        future = astropy.time.Time('2024-01-01', location=utils.get_location())
        lst = future.sidereal_time('apparent')

    def test_freeze_iers_bad_ext(self):
        """Test freezing from package data/"""
        with self.assertRaises(ValueError):
            utils.freeze_iers('_non_existent_.fits')

    def test_freeze_iers_bad_name(self):
        """Test freezing from package data/"""
        with self.assertRaises(ValueError):
            utils.freeze_iers('_non_existent_.ecsv')

    def test_freeze_iers_bad_format(self):
        """Test freezing from valid file with wrong format"""
        with self.assertRaises(ValueError):
            utils.freeze_iers('config.yaml')

    def test_get_overhead(self):
        """Sanity checks on overhead time calculations"""
        c = config.Configuration()
        tro = c.readout_time()
        p0 = astropy.coordinates.ICRS(ra=300 * u.deg, dec=10 * u.deg)
        # Move with no readout and no slew only has focus overhead.
        self.assertEqual(utils.get_overhead_time(None, p0, tro),
                         c.focus_time())
        self.assertEqual(utils.get_overhead_time(p0, p0, 2 * tro),
                         c.focus_time())
        # Move with readout and no slew has focus and overhead in parallel.
        self.assertEqual(utils.get_overhead_time(None, p0),
                         max(c.focus_time(), c.readout_time()))
        self.assertEqual(utils.get_overhead_time(p0, p0, 0.2 * tro),
                         max(c.focus_time(), 0.8 * c.readout_time()))
        # Overhead with slew same when dRA == dDEC.
        for delta in (1, 45, 70):
            ra = p0.ra + [+delta, -delta, 0, 0] * u.deg
            dec = p0.dec + [0, 0, +delta, -delta] * u.deg
            p1 = astropy.coordinates.ICRS(ra=ra, dec=dec)
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
        dec = utils.get_location().latitude
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


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
