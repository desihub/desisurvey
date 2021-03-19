import os
import pytz
import numpy as np
import desisurvey.NTS
import desisurvey.svstats
import desiutil.log
from desisurvey.scripts import collect_etc
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy import units as u


def mjd_to_azstr(mjd):
    utc = pytz.timezone('utc')
    tz = pytz.timezone('US/Arizona')
    tt = Time(mjd, format='mjd').to_datetime(timezone=utc)
    return tt.astimezone(tz).strftime('%H:%M')


def run_plan(nts_dir=None, verbose=False, survey=None):
    kpno = EarthLocation.of_site('kpno')
    if nts_dir is None:
        obsplan = None
        night = None
    else:
        night = desisurvey.utils.get_date(nts_dir)
        obsplan = os.path.join(nts_dir, 'config.yaml')
    nts = desisurvey.NTS.NTS(obsplan=obsplan, night=night, nts_survey=survey)
    t0 = nts.scheduler.night_ephem['brightdusk']
    nts_dir, _ = os.path.split(nts.obsplan)
    if not verbose:
        desiutil.log.get_logger().setLevel(desiutil.log.WARNING)
    previoustiles = []
    ephem = nts.scheduler.night_ephem
    night_labels = np.array(['noon', '12 deg dusk', '18 deg dusk',
                             '18 deg dawn', '12 deg dawn',
                             'moonrise', 'moonset'])
    night_names = np.array(['noon', 'brightdusk', 'dusk', 'dawn', 'brightdawn',
                            'moonrise', 'moonset'])
    night_times = np.array([ephem[name] for name in night_names])
    s = np.argsort(night_times)
    print(nts.scheduler.night)
    for name, tt in zip(night_labels[s], night_times[s]):
        print('%11s %s' % (name, mjd_to_azstr(tt)))

    print('local   lst   cond  tile    ra   dec    program fac  tot  split ' +
          '  N')
    while t0 < nts.scheduler.night_ephem['brightdawn']:
        expdict = dict(mjd=t0, previoustiles=previoustiles)
        res = nts.next_tile(exposure=expdict, speculative=True)
        if not res['foundtile']:
            print('no tiles!')
            t0 += 60/60/60/24
            continue
        previoustiles.append(res['fiberassign'])
        lst = Time(t0, format='mjd', location=kpno).sidereal_time('apparent')
        lst = lst.to(u.deg).value
        ind = np.flatnonzero(
            nts.scheduler.tiles.tileID == res['fiberassign'])[0]
        ra = nts.scheduler.tiles.tileRA[ind]
        dec = nts.scheduler.tiles.tileDEC[ind]
        nnight = nts.scheduler.plan.donefrac[ind]*4
        print(
            ('%s %5.1f %6s %d %5.1f %5.1f %10s %3.1f '
             '%4d %6s %3.1f') % (
            mjd_to_azstr(t0), lst, res['conditions'],
            res['fiberassign'], ra, dec, res['program'],
            res['exposure_factor'], res['esttime'].astype('i4'),
                 ('%dx%d' % (res['count'], res['exptime'])), nnight))
        t0 += (res['exptime']+180)*res['count']/60/60/24


def parse(options=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='run an example night plan',
        epilog='EXAMPLE: %(prog)s [YYYYMMDD/config.yaml]')
    parser.add_argument('nts_dir', nargs='?', default=None, type=str,
                        help='nts_dir to use; default YYYYMMDD')
    parser.add_argument('--survey', default='sv1', type=str,
                        help='survey to use; default sv1')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='verbose output')
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):
    run_plan(nts_dir=args.nts_dir, survey=args.survey, verbose=args.verbose)
