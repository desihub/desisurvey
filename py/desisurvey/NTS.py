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
import desisurvey.utils
import desisurvey.config
import desiutil.log
from astropy.io import ascii
from astropy import coordinates
from astropy import units as u

try:
    import DOSlib.logger as Log
    logob = Log
except ImportError:
    logob = desiutil.log.get_logger()


class QueuedList():
    """Simple class to manage list of exposures already observed in a night.

    Parameters
    ----------
    fn : str
        file name where QueuedList is backed to disk.
    """
    def __init__(self, fn):
        self.fn = fn
        self.log = logob
        self.restore()

    def restore(self):
        if os.path.exists(self.fn):
            try:
                self.queued = ascii.read(self.fn, comment='#',
                                         names=['tileid'], format='no_header')
            except OSError:
                self.log.error('Could not read in queued file; '
                               'record of past exposures lost!')
            self.queued = list(self.queued['tileid'])
        else:
            self.queued = []

    def add(self, tileid):
        self.queued.append(tileid)
        try:
            open(self.fn, 'a').write(str(tileid)+'\n')
            # could work harder to make this atomic.
        except OSError:
            self.log.error('Could not write out queued file; '
                           'record of last exposure lost!')


def azinrange(az, low, high):
    """Return whether azimuth is between low and high, trying to respect the
    360 deg boundary.

    We transform high so that it is in the range [low, low+360].  We then
    transform az likewise, so that the test can be done as low <= az <= high.
    In this scheme, azinrange(0, 2, 1) = True, since low, high = [2, 1] is
    interpreted as all angles between 2 and 361 degrees.

    Parameters
    ----------
    az: azimuth (deg)
    low: lower bound on azimuth (deg)
    high: upper bound on azimuth (deg)

    Returns
    -------
    Array of same shape as az, indicating if az is between low and high.
    """

    if low > high:
        high = ((high - low) % 360) + low
    az = ((az - low) % 360) + low
    return (az >= low) & (az <= high)


class NTS():
    def __init__(self, obsplan='config.yaml', defaults={}, night=None):
        """Initialize a new instance of the Next Tile Selector.

        Parameters
        ----------
        obsplan : config.yaml to load

        defaults : dictionary giving default values of 'seeing',
            'transparency', 'sky_level', and 'program', for next tile
            selection.

        night : night for which to assign tiles, YYYMMDD, default tonight.

        Returns
        -------
        NTS object. Tiles can be generated via next_tile(...)
        """
        self.obsplan = obsplan
        self.log = desiutil.log.get_logger()
        # making a new NTS; clear out old configuration / tile information
        if night is None:
            self.night = desisurvey.utils.get_current_date()
            self.log.info('No night selected, '
                          'using current date: {}.'.format(self.night))
        else:
            self.night = night
        nts_dir, _ = os.path.split(obsplan)
        if len(os.path.split(nts_dir)[0]) > 0:
            raise ValueError('NTS expects to find config in '
                             '$DESISURVEY_OUTPUT/dir/config-file.yaml')
        fulldir = os.path.join(os.environ['DESISURVEY_OUTPUT'], nts_dir)
        obsplan = os.path.join(os.environ['DESISURVEY_OUTPUT'],
                               obsplan)
        if not os.path.exists(obsplan):
            self.log.error('Could not find obsplan configuration '
                           '{}!'.format(obsplan))
            raise ValueError('Could not find obsplan configuration!')
        desisurvey.config.Configuration.reset()
        config = desisurvey.config.Configuration(obsplan)
        _ = desisurvey.tiles.get_tiles(use_cache=False, write_cache=True)

        self.default_seeing = defaults.get('seeing', 1.0)
        self.default_transparency = defaults.get('transparency', 0.9)
        self.default_skylevel = defaults.get('skylevel', 1000.0)
        self.default_program = defaults.get('program', 'DARK')
        self.rules = desisurvey.rules.Rules(
            config.get_path(config.rules_file()))
        self.config = config
        try:
            self.planner = desisurvey.plan.Planner(
                self.rules,
                restore='{}/desi-status-{}.fits'.format(fulldir, nts_dir))
            self.scheduler = desisurvey.scheduler.Scheduler(
                restore='{}/desi-status-{}.fits'.format(fulldir, nts_dir))
            self.queuedlist = QueuedList(
                config.get_path('{}/queued-{}.dat'.format(fulldir, nts_dir)))
        except Exception as e:
            print(e)
            raise ValueError('Error restoring scheduler & planner files; '
                             'has afternoon planning been performed?')
        self.scheduler.update_tiles(self.planner.tile_available,
                                    self.planner.tile_priority)
        self.scheduler.init_night(self.night, use_twilight=True)
        self.ETC = desisurvey.etc.ExposureTimeCalculator()

    def next_tile(self, conditions=None, exposure=None, constraints=None,
                  speculative=False):
        """
        Select the next tile.

        Parameters
        ----------
        conditions : dict, dictionary containing conditions

        exposure : dict, dictionary containing information about exposures

        constraints : dict, dictionary containing constraints on where
            observations may be made

        speculative: bool, if True, NTS may propose this tile again on later
            calls to next_tile this night.

        Returns
        -------
        A dictionary representing the next tile, containing the following
        fields:
        tileid : int, the next tileID.
        s2n : float, the addition s2n needed on this tile
        esttime : float, expected time needed to achieve this s2n (seconds)
        maxtime : float, do not observe for longer than maxtime (seconds)
        fiber_assign : str, file name of fiber_assign file
        foundtile : bool, a valid tile was found
        azrange : [lowaz, highaz], azimuth of tile must be in this range
        """

        if conditions is None:
            conditions = {}
        if exposure is None:
            exposure = {}
        if constraints is None:
            constraints = {}

        mjd = exposure.get('mjd', None)
        seeing = conditions.get('seeing', None)
        skylevel = conditions.get('skylevel', None)
        transparency = conditions.get('transparency', None)
        if seeing is None:
            seeing = self.default_seeing
        if skylevel is None:
            skylevel = self.default_skylevel
        if transparency is None:
            transparency = self.default_transparency

        if mjd is None:
            from astropy import time
            now = time.Time.now()
            mjd = now.mjd
            self.log.info('No time specified, using current time, MJD: %f' %
                          mjd)

        self.queuedlist.restore()

        previoustiles = exposure.get('previoustiles', [])
        if previoustiles is None:
            previoustiles = []
        previoustiles = previoustiles + self.queuedlist.queued
        # remove previous tiles from possible tiles to schedule
        ind, mask = self.scheduler.tiles.index(previoustiles,
                                               return_mask=True)
        save_in_night_pool = self.scheduler.in_night_pool[ind[mask]].copy()
        self.scheduler.in_night_pool[ind[mask]] = False

        azrange = constraints.get('azrange', None)
        if azrange is not None:
            tra = self.scheduler.tiles.tileRA
            tdec = self.scheduler.tiles.tileDEC
            altazframe = desisurvey.utils.get_observer(now)
            coordrd = coordinates.ICRS(ra=tra*u.deg, dec=tdec*u.deg)
            coordaz = coordrd.transform_to(altazframe)
            az = coordaz.az.to(u.deg).value
            self.scheduler.in_night_pool &= azinrange(az, azrange[0],
                                                      azrange[1])

        program = exposure.get('program', None)

        result = self.scheduler.next_tile(
            mjd, self.ETC, seeing, transparency, skylevel, program=program)
        self.scheduler.in_night_pool[ind[mask]] = save_in_night_pool
        (tileid, passnum, snr2frac_start, exposure_factor, airmass,
         sched_program, mjd_program_end) = result
        if tileid is None:
            return {'tileid': None, 's2n': 0., 'esttime': 0., 'maxtime': 0.,
                    'fiber_assign': '', 'foundtile': False}

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
        # this forces an exposure to end at the end of the program, and
        # can lead to awkward behavior if an exposure starts just as a program
        # ends.  Ignoring at present.
        # maxtime = min([maxtime, mjd_program_end-mjd])

        tileidstr = '{:06d}'.format(tileid)

        fiber_assign_dir = desisurvey.plan.get_fiber_assign_dir(None)
        fiber_assign = os.path.join(fiber_assign_dir,
                                    tileidstr[:3],
                                    'fiberassign-%s.fits' % tileidstr)
        days_to_seconds = 60*60*24

        selection = {'tileid': tileid, 's2n': s2n,
                     'esttime': exptime*days_to_seconds,
                     'maxtime': maxtime*days_to_seconds,
                     'fiber_assign': fiber_assign,
                     'foundtile': True}
        if not speculative:
            self.queuedlist.add(tileid)
        self.log.info('Next selection: %r' % selection)
        return selection
