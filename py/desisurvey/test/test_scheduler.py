import unittest
import datetime

import numpy as np

import astropy.units as u

import desisurvey.plan
import desisurvey.etc
import desisurvey.config
import desisurvey.tiles
import desisurvey.utils
from desisurvey.test.base import Tester
from desisurvey.scripts import surveyinit
from desisurvey.scheduler import Scheduler

# Declination dependent hour angle limit used by the tests below.  The tiles
# in the test subset span -10 < DEC < 20, where these limits are well inside
# the flat max_hour_angle, so the cut has something to do.
HA_BY_DEC = '-30:0.5, -20:1.5, -10:2.3, 0:3.0, 10:3.3, 20:3.5, 30:4.0'


class TestScheduler(Tester):

    def enable_ha_by_dec(self, spec=HA_BY_DEC):
        """Turn on the declination dependent HA limit for one test.

        Both the configuration and the cached tiles are global, so restore
        them afterwards however the test exits.
        """
        config = desisurvey.config.Configuration()
        self.addCleanup(desisurvey.tiles.get_tiles, use_cache=False)
        self.addCleanup(config.max_hour_angle_by_dec.set_value, '')
        config.max_hour_angle_by_dec.set_value(spec)
        desisurvey.tiles.get_tiles(use_cache=False)
        return config

    def test_max_ha_is_scalar_when_disabled(self):
        # A stale per-tile array here would silently change how the |HA| cut
        # broadcasts, so pin down that the default stays a scalar.
        config = desisurvey.config.Configuration()
        planner = desisurvey.plan.Planner(simulate=True)
        scheduler = Scheduler(planner)
        self.assertEqual(np.ndim(scheduler.max_ha), 0)
        self.assertAlmostEqual(scheduler.max_ha,
                               config.max_hour_angle().to(u.deg).value)

    def test_max_ha_by_dec(self):
        config = self.enable_ha_by_dec()
        planner = desisurvey.plan.Planner(simulate=True)
        scheduler = Scheduler(planner)
        ceiling = config.max_hour_angle().to(u.deg).value
        self.assertEqual(scheduler.max_ha.shape, (scheduler.tiles.ntiles,))
        expected = desisurvey.utils.max_ha_by_dec(
            scheduler.tiles.tileDEC, HA_BY_DEC, ceiling=ceiling)
        self.assertTrue(np.allclose(scheduler.max_ha, expected))
        # The limit can only tighten the flat cut.
        self.assertTrue(np.all(scheduler.max_ha <= ceiling))

    def test_next_tile_respects_ha_limit(self):
        # The only conclusive check that the cut is applied: every tile the
        # scheduler actually selects must satisfy the per-tile limit, tested
        # on the scheduler's own hour angle (the midpoint of the estimated
        # exposure), which is the quantity the cut uses.  This cannot be
        # checked after the fact from an exposure list, where setup latency,
        # exposure time estimation error and cosmic split continuations all
        # let the realised hour angle drift past the limit legitimately.
        self.enable_ha_by_dec()
        cmd = 'surveyinit --max-cycles 5 --init zero'
        args = surveyinit.parse(cmd.split()[1:])
        surveyinit.main(args)
        config = desisurvey.config.Configuration()
        config.fiber_assignment_cadence.set_value('daily')
        planner = desisurvey.plan.Planner(simulate=True)
        planner.first_night = desisurvey.utils.get_date('2020-01-01')
        planner.last_night = desisurvey.utils.get_date('2025-01-01')
        scheduler = Scheduler(planner)
        self.assertEqual(scheduler.max_ha.shape, (scheduler.tiles.ntiles,))

        nselected = 0
        for i in range(3):
            night = self.start + datetime.timedelta(i)
            planner.afternoon_plan(night)
            scheduler.init_night(night)
            ETC = desisurvey.etc.ExposureTimeCalculator()
            dusk = scheduler.night_ephem['dusk']
            dawn = scheduler.night_ephem['dawn']
            for mjd in np.arange(dusk, dawn, 15. / (24. * 60.)):
                tileid = scheduler.next_tile(
                    mjd, ETC, seeing=1.1, transp=0.95, skylevel=1)[0]
                if tileid is None:
                    continue
                idx = scheduler.tiles.index(tileid)
                absha = np.abs(
                    ((scheduler.hourangle[idx] + 180) % 360) - 180)
                self.assertLess(absha, scheduler.max_ha[idx])
                nselected += 1
                scheduler.update_snr(tileid, 1.)
        # The assertion above is vacuous if nothing was ever scheduled.
        self.assertGreater(nselected, 0)

    def test_scheduler(self):
        cmd = 'surveyinit --max-cycles 5 --init zero'
        args = surveyinit.parse(cmd.split()[1:])
        surveyinit.main(args)
        config = desisurvey.config.Configuration()
        config.fiber_assignment_cadence.set_value('daily')
        planner = desisurvey.plan.Planner(simulate=True)
        planner.first_night = desisurvey.utils.get_date('2020-01-01')
        planner.last_night = desisurvey.utils.get_date('2025-01-01')
        scheduler = Scheduler(planner)
        num_nights = (self.stop - self.start).days
        for i in range(num_nights):
            night = self.start + datetime.timedelta(i)
            # Save and restore scheduler state.
            planner.save('snapshot.ecsv')
            planner2 = desisurvey.plan.Planner(restore='snapshot.ecsv',
                                               simulate=True)
            self.assertTrue(np.all(planner.donefrac == planner2.donefrac))
            self.assertTrue(np.all(planner.tile_status == planner2.tile_status))
            avail, planned = planner.afternoon_plan(night)
            avail2, planned2 = planner2.afternoon_plan(night)
            scheduler2 = Scheduler(planner2)
            self.assertTrue(np.all(scheduler.plan.obsend() ==
                                   scheduler2.plan.obsend()))
            self.assertTrue(np.all(scheduler.plan.obsend_by_program() ==
                                   scheduler2.plan.obsend_by_program()))
            self.assertTrue(np.all(avail == avail2))
            self.assertTrue(np.all(planned == planned2))
            # Run both schedulers in parallel.
            scheduler.init_night(night)
            scheduler2.init_night(night)
            # Loop over exposures during the night.
            dusk, dawn = scheduler.night_ephem['dusk'], scheduler.night_ephem['dawn']
            ETC = desisurvey.etc.ExposureTimeCalculator()
            for mjd in np.arange(dusk, dawn, 15. / (24. * 60.)):
                # TILEID,PROGRAM,SNR2FRAC,EXPFAC,AIRMASS,PROGRAM,PROGEND
                next = scheduler.next_tile(mjd, ETC, seeing=1.1, transp=0.95, skylevel=1)
                # Check that the restored scheduler gives the same results.
                next2 = scheduler2.next_tile(mjd, ETC, seeing=1.1, transp=0.95, skylevel=1)
                for field, field2 in zip(next, next2):
                    self.assertEqual(field, field2)
                tileid = next[0]
                if tileid is not None:
                    scheduler.update_snr(tileid, 1.)
                    scheduler2.update_snr(tileid, 1.)
