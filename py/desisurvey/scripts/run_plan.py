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


def run_plan(nts_dir=None, verbose=False):
    kpno = EarthLocation.of_site('kpno')
    if nts_dir is None:
        obsplan = None
        night = None
    else:
        night = desisurvey.utils.get_date(nts_dir)
        obsplan = os.path.join(nts_dir, 'config.yaml')
    nts = desisurvey.NTS.NTS(obsplan=obsplan, night=night)
    t0 = nts.scheduler.night_ephem['brightdusk']
    nts_dir, _ = os.path.split(nts.obsplan)
    etcfn = os.path.join(nts_dir, 'etc-stats-{}.fits'.format(nts_dir))
    exps = fits.getdata(etcfn, 'EXPS')
    nincond = collect_etc.number_in_conditions(exps)
    donecond = desisurvey.svstats.donefrac_in_conditions(nincond)
    if not verbose:
        desiutil.log.get_logger().setLevel(desiutil.log.WARNING)
    previoustiles = []
    ephem = nts.scheduler.night_ephem
    changes = nts.scheduler.night_changes
    programs = nts.scheduler.night_programs
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
    dgbstr = 'd/g/b' if not nts.scheduler.tiles.nogray else 'd/b'

    print('local   lst   cond  tile    ra   dec    program fac  tot  split ' +
          dgbstr)
    while t0 < nts.scheduler.night_ephem['brightdawn']:
        cidx = np.interp(t0, changes, np.arange(len(changes)))
        cidx = int(np.clip(cidx, 0, len(programs)))
        cond = programs[cidx]
        moon_up_factor = getattr(nts.config.moon_up_factor, cond)()
        expdict = dict(mjd=t0, previoustiles=previoustiles)
        conddict = dict(skylevel=moon_up_factor)
        res = nts.next_tile(exposure=expdict, conditions=conddict,
                            speculative=True)
        if not res['foundtile']:
            print('no tiles!')
            t0 += 60/60/60/24
            continue
        previoustiles.append(res['fiberassign'])
        lst = Time(t0, format='mjd', location=kpno).sidereal_time('apparent')
        lst = lst.to(u.deg).value
        nsofar = (donecond[donecond['TILEID'] == res['fiberassign']])
        ind = np.flatnonzero(
            nts.scheduler.tiles.tileID == res['fiberassign'])[0]
        ra = nts.scheduler.tiles.tileRA[ind]
        dec = nts.scheduler.tiles.tileDEC[ind]
        if len(nsofar) > 0:
            nsofar = [nsofar['NNIGHT_DARK'], nsofar['NNIGHT_GRAY'],
                      nsofar['NNIGHT_BRIGHT']]
        else:
            nsofar = [0, 0, 0]
        if not nts.scheduler.tiles.nogray:
            dgbstr = '%3.1f/%3.1f/%3.1f' % nsofar
        else:
            nsofar = (nsofar[0], nsofar[2])
            dgbstr = '%3.1f/%3.1f' % nsofar
        print(
            ('%s %5.1f %6s %d %5.1f %5.1f %10s %3.1f '
             '%4d %6s %s') % (
            mjd_to_azstr(t0), lst, res['conditions'],
            res['fiberassign'], ra, dec, res['program'],
            res['exposure_factor'], res['esttime'].astype('i4'),
                 ('%dx%d' % (res['count'], res['exptime'])), dgbstr))
        t0 += (res['exptime']+180)*res['count']/60/60/24


def parse(options=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='run an example night plan',
        epilog='EXAMPLE: %(prog)s [YYYYMMDD/config.yaml]')
    parser.add_argument('nts_dir', nargs='?', default=None, type=str,
                        help='nts_dir to use; default YYYYMMDD')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='verbose output')
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):
    run_plan(nts_dir=args.nts_dir, verbose=args.verbose)
