"""
  Next Tile Selector

  Based on our google sheet (
  these are the expected inputs

        skylevel: current sky level [counts s-1 cm-2 arcsec-2]   (from ETC, at the end of last exposure)
        seeing: current atmospheric seeing PSF FWHM [arcsec]     (from ETC, at the end of last exposure)
        transparency: current atmospheric transparency [0-1, where 0=total cloud cover]   (from ETC, at the end of last exposure)

        obsplan: filename containing that nights observing plan
        program (optional): request a tile will be in that program,
                            otherwise get next field chooses program based on current conditions

        previoustiles (optional): list of tiles that have been observed that night

  These variables are in the
        RA _prior:  in degrees, used only for user over-ride, defaults to -99
        DEC_prior:  in degrees, used only for user over-ride, defaults to -99
            EFS: for slewing minimization; not used.

  If input values are missing (e.g. first exposure of the night), the NTS falls back to reasonable defaults for skylevel etc.


  The primary output of the NTS will be a dictionary with the name of the fiber assign file (full path)
  The naming convention is fiberassign_<tileid>.fits

  In addition, the following keys/information is returned by the NTS:
        tileid: (int) DESI Tile ID
        s2n: (foat) Requested signal to noice (for ETC)
        foundtile (boolean): indicates whether field selector was successful.
        exptime: expected exposure time based on ETC information from previous exposure [seconds]
        maxtime: maximum allowable exposure time [seconds]

        Names are converted to FITS convention: TILEID, S2NREQ, EXTTIME, MAXTIME, FBRASSGN
"""

import os
import json
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
from astropy import time
import numpy as np


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
                self.queued = list(self.queued['tileid'])
            except Exception as e:
                self.log.error('Could not read in queued file; '
                               'record of past exposures lost!')
                self.log.error('Got exception trying to load queued file!')
                self.log.error(e)
                self.queued = []
        else:
            self.queued = []

    def add(self, tileid):
        if tileid < 0:
            self.log.info('Not adding unknown TILEID to queued file.')
            return
        self.queued.append(tileid)
        exists = os.path.exists(self.fn)
        try:
            fp = open(self.fn, 'a')
            fp.write(str(tileid)+'\n')
            fp.flush()
            # could work harder to make this atomic.
        except OSError:
            self.log.error('Could not write out queued file; '
                           'record of last exposure lost!')
        if not exists:
            os.chmod(self.fn, 0o666)


class RequestLog():
    """Simple class to log requests to NTS.

    Parameters
    ----------
    fn : str
        file name where RequestLog is stored.
    """

    def __init__(self, fn):
        self.fn = fn
        exists = os.path.exists(fn)
        _ = open(fn, 'a')
        if not exists:
            os.chmod(self.fn, 0o666)

    def logrequest(self, conditions, exposure, constraints, speculative):
        now = time.Time.now()
        mjd = now.mjd
        res = dict(requesttime=mjd, conditions=conditions, exposure=exposure,
                   constraints=constraints, speculative=speculative)
        try:
            s = json.dumps(res)
        except Exception as e:
            logob.error('Could not dump request json to log!')
            logob.error(str(e))
            s = 'Error, missing entry!'
        fp = open(self.fn, 'a')
        fp.write(s+'\n')
        fp.flush()

    def logresponse(self, tile):
        now = time.Time.now()
        mjd = now.mjd
        res = dict(requesttime=mjd, tile=tile)
        try:
            s = json.dumps(res)
        except Exception as e:
            logob.error('Could not dump response json to log!')
            logob.error(str(e))
            s = 'Error, missing entry!'
        fp = open(self.fn, 'a')
        fp.write(s+'\n')
        fp.flush()


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
    def __init__(self, obsplan=None, defaults={}, night=None,
                 nts_survey=None):
        """Initialize a new instance of the Next Tile Selector.

        Parameters
        ----------
        obsplan : config.yaml to load

        defaults : dictionary giving default values of 'seeing',
            'transparency', 'sky_level', and 'program', for next tile
            selection.

        night : night for which to assign tiles, YYYYMMDD, default tonight.

        nts_survey : human readable means for selecting nightly obsplan.
            ignored if obsplan is set.

        Returns
        -------
        NTS object. Tiles can be generated via next_tile(...)
        """
        self.log = logob
        # making a new NTS; clear out old configuration / tile information
        if night is None:
            self.night = desisurvey.utils.get_current_date()
            self.log.info('No night selected, '
                          'using current date: {}.'.format(self.night))
        else:
            self.night = desisurvey.utils.get_date(night)
        if obsplan is None:
            if nts_survey is None:
                nts_survey = 'sv1'
            nts_survey = nts_survey.lower()
            nts_dir = (desisurvey.utils.night_to_str(self.night) + '-' +
                       nts_survey)
            obsplan = os.path.join(nts_dir, 'config.yaml')
        self.obsplan = obsplan
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

        self.default_seeing = defaults.get('seeing', 1.1)
        self.default_transparency = defaults.get('transparency', 1.0)
        self.default_skylevel = defaults.get('skylevel', 1.0)
        self.default_program = defaults.get('program', 'DARK')
        self.rules = desisurvey.rules.Rules(
            config.get_path(config.rules_file()))
        self.config = config
        try:
            self.planner = desisurvey.plan.Planner(
                self.rules, restore=config.tiles_file(),
                log=self.log)
            self.scheduler = desisurvey.scheduler.Scheduler(
                self.planner, log=self.log)
            self.queuedlist = QueuedList(
                '{}/queued.dat'.format(fulldir))
            self.requestlog = RequestLog(
                '{}/requestlog.dat'.format(fulldir))
        except Exception as e:
            print(e)
            raise ValueError('Error restoring scheduler & planner files; '
                             'has afternoon planning been performed?')
        self.scheduler.init_night(self.night, use_twilight=True)
        for queuedtile in self.queuedlist.queued:
            self.scheduler.plan.add_pending_tile(queuedtile)
        self.ETC = desisurvey.etc.ExposureTimeCalculator()

    def next_tile(self, conditions=None, exposure=None, constraints=None,
                  speculative=False):
        """
        Select the next tile.

        Parameters
        ----------
        conditions : dict, dictionary containing conditions
            Recognized keywords include:
            skylevel : current sky level
            seeing : current seeing
            transparency : current transparency

        exposure : dict, dictionary containing information about exposures
            Recognized keywords include:
            mjd : time at which tile is to be observed
            previoustiles : list of tileids that should not be observed
            program : program of tile to select

        constraints : dict, dictionary containing constraints on where
            observations may be made.  Recognized constraints include:
            azrange : [lowaz, highaz], azimuth of tile must be in this range
            elrange : [lowel, highel], elevation of tile must be in this range


        speculative: bool, if True, NTS may propose this tile again on later
            calls to next_tile this night.

        Returns
        -------
        A dictionary representing the next tile, containing the following
        fields:
        fiberassign : int, the next tileID.
        s2n : float, the addition s2n needed on this tile
        esttime : float, expected total time needed to achieve this s2n (seconds)
        exptime : float, amount of time per (split) exposure
        count : int, number of exposures to make
        maxtime : float, do not observe for longer than maxtime (seconds)
        foundtile : bool, a valid tile was found
        conditions : DARK / GRAY / BRIGHT
        program : program of this tile
        exposure_factor : exptime scale factor applied for E(B-V) and airmass
        """

        if conditions is None:
            conditions = {}
        if exposure is None:
            exposure = {}
        if constraints is None:
            constraints = {}

        self.requestlog.logrequest(conditions, exposure, constraints,
                                   speculative)

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
        save_in_night_pool = self.scheduler.in_night_pool.copy()
        self.scheduler.in_night_pool[ind[mask]] = False

        azrange = constraints.get('azrange', None)
        elrange = constraints.get('elrange', None)
        if (azrange is not None) or (elrange is not None):
            tra = self.scheduler.tiles.tileRA
            tdec = self.scheduler.tiles.tileDEC
            altazframe = desisurvey.utils.get_observer(now)
            coordrd = coordinates.ICRS(ra=tra*u.deg, dec=tdec*u.deg)
            coordaz = coordrd.transform_to(altazframe)
            az = coordaz.az.to(u.deg).value
            el = coordaz.alt.to(u.deg).value
            if azrange is not None:
                self.scheduler.in_night_pool &= azinrange(az, azrange[0],
                                                          azrange[1])
            if elrange is not None:
                self.scheduler.in_night_pool &= (
                    (el >= elrange[0]) & (el <= elrange[1]))

        program = exposure.get('program', None)

        result = self.scheduler.next_tile(
            mjd, self.ETC, seeing, transparency, skylevel, program=program,
            verbose=True)
        self.scheduler.in_night_pool = save_in_night_pool
        (tileid, passnum, snr2frac_start, exposure_factor, airmass,
         sched_program, mjd_program_end) = result
        if mjd_program_end < mjd:
            self.log.warning(
                'Program ends before exposure starts; is it daytime?')
            mjd_program_end = mjd + 1

        badtile = {'ra': 0., 'dec': 90.,
                   'esttime': 0., 'exptime': 0.,
                   'count': 0, 'maxtime': 0., 'fiberassign': 0,
                   'foundtile': False,
                   'conditions': '', 'program': '', 'exposure_factor': 0,
                   'req_efftime': 0., 'sbprof': 'PSF', 'mintime': 0,
                   'cosmics_splittime': 1000}
        if tileid is None:
            self.requestlog.logresponse(badtile)
            return badtile

        self.scheduler.plan.add_pending_tile(tileid)

        idx = self.scheduler.tiles.index(int(tileid))
        tile_program = self.scheduler.tiles.tileprogram[idx]
        programconf = getattr(self.config.programs, tile_program, None)
        if programconf is None:
            self.log.error('Did not recognize program {}'.format(
                tile_program))
            return badtile

        svmode = getattr(self.config, 'svmode', None)
        svmode = svmode() if svmode is not None else False
        if svmode or (snr2frac_start > self.config.min_snr2_fraction()):
            # in svmode we always go for full visits
            # if this tile is already finished, it's a backup tile; go for a
            # full visit.
            snr2frac_start = 0
        snr2frac_start = np.clip(snr2frac_start, 0, 1)
        texp_tot, texp_remaining, nexp_remaining = self.ETC.estimate_exposure(
            tile_program, snr2frac_start, exposure_factor, nexp_completed=0)
        efftime = getattr(programconf, 'efftime', None)
        if efftime is not None:
            efftime = efftime()
        else:
            efftime = 1000*u.s
        efftime = float(efftime.to(u.s).value)*(1-snr2frac_start)

        sbprof = getattr(programconf, 'sbprof', None)
        if sbprof is not None:
            sbprof = sbprof()
        if not isinstance(sbprof, str):
            sbprof = 'PSF'

        boost_factor = (
            getattr(self.config.conditions, sched_program).boost_factor())
        texp_tot *= boost_factor
        texp_remaining *= boost_factor

        # avoid crossing program boundaries, don't observe longer than an hour.
        maxdwell = self.config.maxtime().to(u.day).value
        mintime = getattr(programconf, 'mintime', None)
        if mintime is not None:
            mintime = mintime()
        else:
            mintime = self.config.mintime()
        mintime = mintime.to(u.day).value
        texp_remaining = max([texp_remaining, mintime])
        texp_remaining = min([texp_remaining, maxdwell])

        lstnow = (self.scheduler.LST0 +
                  self.scheduler.dLST*(mjd - self.scheduler.MJD0))
        hanow = lstnow - self.scheduler.tiles.tileRA[idx]
        maxdha = self.scheduler.tiles.max_abs_ha[idx] - hanow
        maxdwell = min([maxdwell, maxdha/360])
        texp_remaining = min([texp_remaining, maxdwell])

        onemin = 1/60/24
        # end dark/gray programs at 15 deg dawn, sharp.
        if ((sched_program != 'BRIGHT') and
                (mjd_program_end > self.scheduler.night_ephem['dusk'])):
            texp_remaining = min([texp_remaining, mjd_program_end-mjd])

        exptime = texp_remaining
        splittime = self.config.cosmic_ray_split().to(u.day).value

        days_to_seconds = 60*60*24
        if ((mjd <= self.scheduler.night_ephem['dusk']-5*onemin) or
                (mjd >= self.scheduler.night_ephem['dawn']-5*onemin)):
            splittime = 301/days_to_seconds
            # in twilight, exposures should never be longer than 300 s
            # according to DJS.

        if exptime > splittime:
            count = int((exptime / splittime).astype('i4') + 1)
        else:
            count = 1
        mincount = getattr(programconf, 'min_exposures', None)
        if mincount is not None:
            mincount = mincount()
        else:
            mincount = 1
        count = np.max([mincount, count])
        splitexptime = exptime / count
        minexptime = getattr(programconf, 'minimum_exposure_time', None)
        if minexptime:
            minexptime = getattr(minexptime, sched_program)()
            minexptime = minexptime.to(u.s).value
            splitexptime = max([splitexptime, minexptime/days_to_seconds])

        selection = {
            'ra': float(self.scheduler.tiles.tileRA[idx]),
            'dec': float(self.scheduler.tiles.tileDEC[idx]),
            'esttime': float(exptime*days_to_seconds),
            'exptime': float(splitexptime*days_to_seconds),
            'count': int(count),
            'maxtime': float(maxdwell*days_to_seconds),
            'fiberassign': int(tileid),
            'foundtile': True,
            'conditions': str(sched_program),
            'program': str(tile_program),
            'exposure_factor': float(exposure_factor),
            'req_efftime': float(efftime),
            'sbprof': str(sbprof),
            'mintime': float(mintime*days_to_seconds),
            'cosmics_splittime': float(splittime*days_to_seconds)}
        if not speculative:
            self.queuedlist.add(tileid)
        self.log.info('Next selection: %r' % selection)
        self.requestlog.logresponse(selection)
        return selection
