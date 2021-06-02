import os
import pytz
import numpy as np
import desisurvey.NTS
import desisurvey.svstats
import desiutil.log
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy import units as u


def mjd_to_azstr(mjd):
    utc = pytz.timezone('utc')
    tz = pytz.timezone('US/Arizona')
    tt = Time(mjd, format='mjd').to_datetime(timezone=utc)
    return tt.astimezone(tz).strftime('%H:%M')


def run_plan(night=None, nts_dir=None, verbose=False, survey=None,
             seeing=1.1, table=False, azrange=None):
    kpno = EarthLocation.of_site('kpno')
    if nts_dir is None:
        obsplan = None
    else:
        if night is None:
            raise ValueError('if nts-dir is set, must also set night')
        else:
            night = desisurvey.utils.get_date(night)
        obsplan = os.path.join(nts_dir, 'config.yaml')
    nts = desisurvey.NTS.NTS(obsplan=obsplan, night=night, nts_survey=survey)
    t0 = nts.scheduler.night_ephem['brightdusk']
    nts_dir, _ = os.path.split(nts.obsplan)
    if not verbose:
        desiutil.log.get_logger().setLevel('WARNING')
        nts.log.setLevel('WARNING')
    previoustiles = []
    ephem = nts.scheduler.night_ephem
    changes = nts.scheduler.night_changes
    programs = nts.scheduler.night_programs
    night_labels = np.array(['noon', '12 deg dusk', '15 deg dusk',
                             '15 deg dawn', '12 deg dawn',
                             'moonrise', 'moonset'])
    night_names = np.array(['noon', 'brightdusk', 'dusk', 'dawn', 'brightdawn',
                            'moonrise', 'moonset'])
    night_times = np.array([ephem[name] for name in night_names])
    s = np.argsort(night_times)
    print(nts.scheduler.night)
    for name, tt in zip(night_labels[s], night_times[s]):
        print('%11s %s' % (name, mjd_to_azstr(tt)))

    print('local   lst   cond  tile    ra   dec    program   x fac  tot  split ' +
          'refft')
    current_ra = None
    current_dec = None
    constraints = dict(azrange=azrange)
    while t0 < nts.scheduler.night_ephem['brightdawn']:
        cidx = np.interp(t0+300/86400, changes, np.arange(len(changes)))
        cidx = int(np.clip(cidx, 0, len(programs)-1))
        cond = programs[cidx]
        moon_up_factor = getattr(nts.config.conditions, cond).moon_up_factor()
        expdict = dict(mjd=t0, previoustiles=previoustiles)
        conddict = dict(skylevel=moon_up_factor, seeing=seeing,
                        sky_ra=current_ra, sky_dec=current_dec)
        res = nts.next_tile(exposure=expdict, conditions=conddict,
                            speculative=True, constraints=constraints)
        if not res['foundtile']:
            t0 += 10*60/60/60/24
            continue
        previoustiles.append(int(res['fiberassign']))
        lst = Time(t0, format='mjd', location=kpno).sidereal_time('apparent')
        lst = lst.to(u.deg).value
        ind = np.flatnonzero(
            nts.scheduler.tiles.tileID == res['fiberassign'])[0]
        ra = nts.scheduler.tiles.tileRA[ind]
        dec = nts.scheduler.tiles.tileDEC[ind]
        current_ra = ra
        current_dec = dec
        am = nts.scheduler.tiles.airmass_at_mjd(t0, mask=ind)
        refft = res['req_efftime']
        if not table:
            print(
                ('%s %5.1f %6s %5d %5.1f %5.1f %10s %3.1f %3.1f '
                 '%4d %6s %4d') % (
                     mjd_to_azstr(t0), lst, res['conditions'],
                     res['fiberassign'], ra, dec, res['program'],
                     am, res['exposure_factor'], int(res['esttime']),
                     ('%dx%d' % (res['count'], res['exptime'])), refft))
        else:
            print(f"|| {mjd_to_azstr(t0)} || {lst:5.1f} "
                  f"|| {res['fiberassign']} || "
                  f"{res['ra']} || {res['dec']}"
                  f"|| {res['program']:6s} || "
                  f"{{{{{{sequence=DESI, exptime=5400.0, guider_exptime=5.0, "
                  f"acquisition_exptime=15, "
                  f"focus_exptime=60.0, sky_exptime=60.0, flavor=science, "
                  f"useetc=True, program='{res['program']}', "
                  f"fiberassign={res['fiberassign']}, "
                  f"req_efftime={res['req_efftime']:.0f}, "
                  f"sbprof='{res['sbprof']}', maxtime=5400, "
                  f"esttime={int(res['esttime'])},"
                  f"}}}}}} ||")

        t0 += (res['exptime']+0*180)*res['count']/60/60/24


def parse(options=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='run an example night plan',
        epilog='EXAMPLE: %(prog)s [YYYYMMDD/config.yaml]')
    parser.add_argument('night', nargs='?', default=None, type=str,
                        help='nts_dir to use; default YYYYMMDD')
    parser.add_argument('--survey', default='main', type=str,
                        help='survey to use; default main')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='verbose output')
    parser.add_argument('--nts-dir', default=None,
                        help='planning directory to use')
    parser.add_argument('--seeing', default=1.1, help='set seeing for night.',
                        type=float)
    parser.add_argument('--table', default=False, action='store_true')
    parser.add_argument('--azrange', default=None, nargs=2, type=float,
                        help='Require tiles to land in given azrange.')
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):
    run_plan(night=args.night, nts_dir=args.nts_dir,
             survey=args.survey, verbose=args.verbose, seeing=args.seeing,
             table=args.table, azrange=args.azrange)
