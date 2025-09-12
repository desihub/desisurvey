import unittest
import datetime

import numpy as np

import desisurvey.plan
import desisurvey.etc
import desisurvey.config
import desisurvey.utils
from desisurvey.test.base import Tester
from desisurvey.scripts import surveyinit
from desisurvey.scheduler import Scheduler


class TestScheduler(Tester):

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
