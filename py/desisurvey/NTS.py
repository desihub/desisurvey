"""
  Next Tile Selector

  Based on our google sheet (
  these are the expected inputs

        skylevel: current sky level [counts s-1 cm-2 arcsec-2]   (from ETC, at the end of last exposure)
        seeing: current atmospheric seeing PSF FWHM [arcsec]     (from ETC, at the end of last exposure)
        transparency: current atmospheric transparency [0-1, where 0=total cloud cover]   (from ETC, at the end of last exposure)
        lastexp:  completion time in UTC of most recent exposure
            EFS: this is currently not used.

        obsplan: filename containing that nights observing plan, defaults to planYEARMMDD.fits
            EFS: this is currently not used.
        fiber_assign_dir: (output) directory for fiber assign files.
            EFS: this is currently not used for more than prepending to the tileid.
        program (optional): request a tile will be in that program,
                            otherwise get next field chooses program based on current conditions

        previoustiles (optional): list of tiles that have been observed that night (IS THIS RECORDED IN A FILE?)
                                  This should be handled internally if at all possible. THe NTS could also scan the
                                  fiber_assign_dir directory.

  These variables are in the
        RA _prior:  in degrees, used only for user over-ride, defaults to -99
        DEC_prior:  in degrees, used only for user over-ride, defaults to -99
            EFS: for slewing minimization; not used.

  If input values are missing (e.g. first exposure of the night), the NTS falls back to reasonable defaults for skylevel etc.


  The primary output of the NTS will be a dictionary with the name of the fiber assign file (full path)
  The naming convention is tile_<tileid>.fits

  In addition, the following keys/information is returned by the NTS:
        tileid: (int) DESI Tile ID
        s2n: (foat) Requested signal to noice (for ETC)
        foundtile (boolean): indicates whether field selector was successful.
        exptime: expected exposure time based on ETC information from previous exposure [seconds]
        maxtime: maximum allowable exposure time [seconds]

        Names are converted to FITS convention: TILEID, S2NREQ, EXTTIME, MAXTIME, FBRASSGN
"""

import os
import desisurvey
import desisurvey.rules
import desisurvey.plan
import desisurvey.scheduler
import desisurvey.etc
import datetime


class NTS():
    def __init__(self, obsplan, fiber_assign_dir, defaults={}, night=None):
        """The caller has checked that the obsplan file exists and that
        fiber_assign_dir is writable
        """
        self.obsplan = obsplan
        self.fiber_assign_dir = fiber_assign_dir
        self.default_seeing = defaults.get('seeing', 1.0)
        self.default_transparency = defaults.get('transparency', 0.9)
        self.default_skylevel = defaults.get('skylevel', 1000.0)
        self.default_program = defaults.get('program', 'DESI DARK')
        if night is None:
            self.night = datetime.date.today()
            print('Warning: no night selected, using current date!',
                  self.night)
        else:
            self.night = night
        self.rules = desisurvey.rules.Rules()
        # should look for rules file in obsplan dir?
        try:
            nightstr = self.night.isoformat()
            self.planner = desisurvey.plan.Planner(
                self.rules,
                restore='planner_afternoon_{}.fits'.format(nightstr))
            self.scheduler = desisurvey.scheduler.Scheduler(
                restore='scheduler_{}.fits'.format(nightstr))
        except:
            raise ValueError('Error restoring scheduler & planner files; '
                             'has afternoon planning been performed?')
        self.scheduler.update_tiles(self.planner.tile_available,
                                    self.planner.tile_priority)
        self.scheduler.init_night(self.night, use_twilight=True)
        self.ETC = desisurvey.etc.ExposureTimeCalculator()

    def next_tile(self, mjd=None, skylevel=None, transparency=None,
                  seeing=None, program=None, lastexp=None, fiber_assign=None,
                  previoustiles=None):
        """
        select the next tile
        """

        if fiber_assign is not None:
            raise ValueError('NTS: not sure what to do with '
                             'fiberassign != None')  # EFS

        # tileid, s2n, exptime, maxtime

        if mjd is None:
            from astropy import time
            mjd = time.Time.now().mjd
            print('Warning: no time specified, using current time, MJD: %f' %
                  mjd)
        seeing = self.default_seeing if seeing is None else seeing
        skylevel = self.default_skylevel if skylevel is None else skylevel
        transparency = (self.default_transparency if transparency is None
                        else transparency)

        if previoustiles is not None:
            ind, mask = self.scheduler.tiles.index(previoustiles,
                                                   return_mask=True)
            self.scheduler.in_night_pool[ind[mask]] = False
        # remove previous tiles from possible tiles to schedule
        # note: this will be remembered until NTS is restarted!  EFS
        result = self.scheduler.next_tile(
            mjd, self.ETC, seeing, transparency, skylevel, program=program)
        (tileid, passnum, snr2frac_start, exposure_factor, airmass,
         sched_program, mjd_program_end) = result
        if tileid is None:
            return {'tileid': None, 's2n': 0., 'esttime': 0., 'maxtime': 0.,
                    'fiber_assign': '', 'foundtile': False}

        # lastexp ignored, fiberassign ignored.  EFS
        texp_tot, texp_remaining, nexp_remaining = self.ETC.estimate_exposure(
            sched_program, snr2frac_start, exposure_factor, nexp_completed=0)

        # s2n: this is really what we should be passing back, but currently
        # scheduler thinks in terms of texp in 'nominal' conditions.  Want
        # guidance converting this to s2n...
        # what cosmic split related elements should I be thinking about here?

        s2n = 50.0 * texp_remaining/(
            self.ETC.TEXP_TOTAL[sched_program]*exposure_factor)  # EFS hack
        exptime = texp_remaining
        maxtime = self.ETC.MAX_EXPTIME
        self.scheduler.update_snr(
            tileid, snr2frac_start + min([exptime, maxtime])/texp_tot)
        self.scheduler.save('scheduler_{}.fits'.format(self.night.isoformat()))
        if program is None:
            maxtime = min([maxtime, mjd_program_end-maxtime])

        fiber_assign = os.path.join(self.fiber_assign_dir,
                                    'tile_%d.fits' % tileid)
        days_to_seconds = 60*60*24

        selection = {'tileid': tileid, 's2n': s2n,
                     'esttime': exptime*days_to_seconds,
                     'maxtime': maxtime*days_to_seconds,
                     'fiber_assign': fiber_assign,
                     'foundtile': True}

        return selection


def afternoon_plan(night=None, lastnight=None):
    if night is None:
        night = datetime.date.today().isoformat()
    rules = desisurvey.rules.Rules()
    # should look for rules file in obsplan dir?
    if lastnight is not None:
        planner = desisurvey.plan.Planner(
            rules, restore='planner_end_{}.fits' % lastnight)
        scheduler = desisurvey.scheduler.Scheduler(
            restore='scheduler_end_{}.fits')
    else:
        planner = desisurvey.plan.Planner(rules)
        scheduler = desisurvey.scheduler.Scheduler()
    # restore: maybe check directory, and restore if file present?  EFS
    # planner.save(), scheduler.save()
    # planner.restore(), scheduler.restore()
    import dateutil.parser
    planner.afternoon_plan(dateutil.parser.parse(night).date(),
                           scheduler.completed)
    # currently afternoon planning checks to see what tiles have been marked
    # as done, and what new tiles may now be fiberassigned.
    # currently moves tiles closer to fiber assignment each time it's called
    # (countdon decreases), depending on fiber_assign_cadence, etc.
    # eventually we want this to be ~totally different, so while this isn't
    # really the behavior we'd want on the mountain, I'm leaving it until
    # we have something much different.
    planner.save('planner_afternoon_{}.fits'.format(night))
    scheduler.save('scheduler_{}.fits'.format(night))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Perform afternoon planning.',
        epilog='EXAMPLE: %(prog)s --night 2020-01-01')
    parser.add_argument('--night', type=str,
                        help='night to plan, default: tonight',
                        default=None)
    parser.add_argument('--lastnight', type=str,
                        help='night to restore, default: start fresh.',
                        default=None)

    parser.parse_args()
    afternoon_plan(parser.night, parser.lastnight)
