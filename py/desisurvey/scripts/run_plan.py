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


def run_plan(obsplan=None):
    utc = pytz.timezone('utc')
    tz = pytz.timezone('US/Arizona')
    kpno = EarthLocation.of_site('kpno')
    nts = desisurvey.NTS.NTS(obsplan=obsplan)
    t0 = nts.scheduler.night_ephem['brightdusk']
    nts_dir, _ = os.path.split(nts.obsplan)
    etcfn = os.path.join(nts_dir, 'etc-stats-{}.fits'.format(nts_dir))
    exps = fits.getdata(etcfn, 'EXPS')
    nincond = collect_etc.number_in_conditions(exps)
    donecond = desisurvey.svstats.donefrac_in_conditions(nincond)
    desiutil.log.get_logger().setLevel(desiutil.log.WARNING)
    previoustiles = []
    print('local   lst   cond  tile    ra   dec    program fac  tot  split '
          'd/g/b')
    while t0 < nts.scheduler.night_ephem['brightdawn']:
        expdict = dict(mjd=t0, previoustiles=previoustiles)
        res = nts.next_tile(exposure=expdict, speculative=True)
        if not res['foundtile']:
            print('no tiles!')
            t0 += 60
            continue
        previoustiles.append(res['fiberassign'])
        lst = Time(t0, format='mjd', location=kpno).sidereal_time('apparent')
        lst = lst.to(u.deg).value
        tt = Time(t0, format='mjd').to_datetime(timezone=utc)
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
        print('%s %5.1f %6s %d %5.1f %5.1f %10s %3.1f %4d %6s %d/%d/%d' % (
            tt.astimezone(tz).strftime('%H:%M'), lst, res['conditions'],
            res['fiberassign'], ra, dec, res['program'],
            res['exposure_factor'], res['esttime'].astype('i4'),
            ('%dx%d' % (res['count'], res['exptime'])), *nsofar))
        t0 += (res['exptime']+180)*res['count']/60/60/24


def parse(options=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='run an example night plan',
        epilog='EXAMPLE: %(prog)s [YYYYMMDD/config.yaml]')
    parser.add_argument('obsplan', nargs='?', default=None, type=str,
                        help='obsplan to use; default YYYYMMDD/config.yaml')
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):
    run_plan(obsplan=args.obsplan)
