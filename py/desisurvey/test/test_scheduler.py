import unittest
import datetime

import numpy as np

import desisurvey.plan
import desisurvey.etc
import desisurvey.config
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
        planner = desisurvey.plan.Planner()
        scheduler = Scheduler()
        num_nights = (self.stop - self.start).days
        for i in range(num_nights):
            night = self.start + datetime.timedelta(i)
            # Save and restore scheduler state.
            scheduler.save('snapshot.fits')
            scheduler2 = Scheduler(restore='snapshot.fits')
            self.assertTrue(np.all(scheduler.snr2frac == scheduler2.snr2frac))
            self.assertTrue(np.all(scheduler.completed == scheduler2.completed))
            self.assertTrue(np.all(scheduler.completed_by_pass == scheduler2.completed_by_pass))
            avail, pri = planner.afternoon_plan(night, scheduler.completed)
            # Run both schedulers in parallel.
            scheduler.update_tiles(avail, pri)
            scheduler.init_night(night)
            scheduler2.update_tiles(avail, pri)
            scheduler2.init_night(night)
            # Loop over exposures during the night.
            dusk, dawn = scheduler.night_ephem['dusk'], scheduler.night_ephem['dawn']
            ETC = desisurvey.etc.ExposureTimeCalculator()
            for mjd in np.arange(dusk, dawn, 15. / (24. * 60.)):
                # TILEID,PASSNUM,SNR2FRAC,EXPFAC,AIRMASS,PROGRAM,PROGEND
                next = scheduler.next_tile(mjd, ETC, seeing=1.1, transp=0.95, skylevel=1)
                # Check that the restored scheduler gives the same results.
                next2 = scheduler2.next_tile(mjd, ETC, seeing=1.1, transp=0.95, skylevel=1)
                for field, field2 in zip(next, next2):
                    self.assertEqual(field, field2)
                tileid = next[0]                
                if tileid is not None:
                    scheduler.update_snr(tileid, 1.)
                    scheduler2.update_snr(tileid, 1.)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
