import unittest
import os
import datetime
import tempfile
import shutil

import numpy as np

from astropy.table import Table

import desimodel.io

import desisurvey.config
import desisurvey.ephem
import desisurvey.tiles
import desisurvey.plan
import desisurvey.etc
from desisurvey.scripts import surveyinit
from desisurvey.scheduler import Scheduler


class TestScheduler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory.
        cls.tmpdir = tempfile.mkdtemp()
        # Write output files to this temporary directory.
        config = desisurvey.config.Configuration()
        config.set_output_path(cls.tmpdir)
        # Run for 1 week for testing (but include some time in each program)
        cls.start = datetime.date(2019,12,1)
        cls.stop = datetime.date(2019,12,8)
        desisurvey.ephem.START_DATE = cls.start
        desisurvey.ephem.STOP_DATE = cls.stop
        config.first_day.set_value(cls.start)
        config.last_day.set_value(cls.stop)
        # Use just a subset of the tiles for faster testing
        tiles = Table(desimodel.io.load_tiles())
        subset = (35 < tiles['RA']) & (tiles['RA'] < 55) & \
                 (-10 < tiles['DEC']) & (tiles['DEC'] < 20)
        tiles_file = os.path.join(cls.tmpdir, 'tiles-subset.fits')
        tiles[subset].write(tiles_file)
        config.tiles_file.set_value(tiles_file)

    @classmethod
    def tearDownClass(cls):
        # Remove the directory after the test.
        shutil.rmtree(cls.tmpdir)
        # Reset our configuration.
        desisurvey.config.Configuration.reset()
        desisurvey.ephem._ephem = None

    def test_scheduler(self):
        cmd = 'surveyinit --max-cycles 5 --init zero'
        args = surveyinit.parse(cmd.split()[1:])
        surveyinit.main(args)
        planner = desisurvey.plan.Planner(fiberassign_cadence='daily')
        scheduler = Scheduler()
        num_nights = (self.stop - self.start).days
        for i in range(num_nights):
            night = self.start + datetime.timedelta(i)
            scheduler.update_tiles(*planner.afternoon_plan(night, scheduler.completed))
            scheduler.init_night(night)
            # Save and restore scheduler state.
            scheduler.save('snapshot.fits')
            scheduler2 = Scheduler(restore='snapshot.fits')
            scheduler2.init_night(night)
            # Loop over exposures during the night.
            dusk, dawn = scheduler.night_ephem['dusk'], scheduler.night_ephem['dawn']
            ETC = desisurvey.etc.ExposureTimeCalculator()
            for mjd in np.arange(dusk, dawn, 15. / (24. * 60.)):
                tileid, _, _, _, _, _, _ = scheduler.next_tile(mjd, ETC, seeing=1.1, transp=0.95)
                # Check that the restored scheduler gives the same results.
                tileid2, _, _, _, _, _, _ = scheduler.next_tile(mjd, ETC, seeing=1.1, transp=0.95)
                self.assertEqual(tileid, tileid2)
                if tileid is not None:
                    scheduler.update_snr(tileid, 1.)
                    scheduler2.update_snr(tileid, 1.)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
