import os
import subprocess
import pytz
import numpy as np
import datetime
import ephem as pyephem
import desisurvey.NTS
import desisurvey.svstats
import desiutil.log
import desisurvey.utils
import desisurvey.tiles
import desisurvey.ephem
import desisurvey.config
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy import units as u
from functools import partial


def mjd_to_azstr(mjd):
    utc = pytz.timezone('utc')
    tz = pytz.timezone('US/Arizona')
    tt = Time(mjd, format='mjd').to_datetime(timezone=utc)
    return tt.astimezone(tz).strftime('%H:%M')


def worktile(tileid, logdir=None):
    if logdir is None:
        stdout = subprocess.DEVNULL
    else:
        stdout = open(os.path.join(logdir, f'log-{tileid}-main.log'), 'w')
    return subprocess.call(
        ['fba-main-onthefly.sh', str(tileid), 'n', 'manual'],
        stdout=stdout)


def workqa(tileid, logdir=None):
    if logdir is None:
        stdout = subprocess.DEVNULL
    else:
        stdout = open(os.path.join(logdir, f'log-{tileid}-qa.log'), 'w')
    return subprocess.call(
        ['fba-main-onthefly.sh', str(tileid), 'y', 'manual'],
        stdout=stdout, stderr=stdout)


def planplot(tileid, plan, title='Nightly plan'):
    tiles = desisurvey.tiles.get_tiles()
    idx = tiles.index(tileid)
    mtonight = np.zeros(tiles.ntiles, dtype='bool')
    mtonight[idx] = True
    from matplotlib import pyplot as p
    allprog = ['DARK', 'BRIGHT', 'BACKUP', 'DARK1B', 'BRIGHT1B']
    # existing programs
    prog2plot = [
        prog for prog in allprog if prog in tiles.program_mask.keys()
    ]
    # existing programs with enabled tiles
    prog2plot = [
        prog for prog in prog2plot if tiles.program_mask[prog].sum() > 0
    ]
    nrow = 3
    if 'DARK1B' in prog2plot:
        p.figure(figsize=(17, 11))
        ncol = 2
        ii = [1, 3, 5, 2, 4]
    else:
        p.figure(figsize=(8.5, 11))
        ncol = 1
        ii = [1, 2, 3]
    ii = [ii[j] for j in range(len(ii)) if allprog[j] in prog2plot]
    loff = -60
    tra = ((tiles.tileRA - loff) % 360) + loff
    for i, program in zip(ii, prog2plot):
        m = (tiles.program_mask[program]) & (tiles.in_desi != 0)
        p.subplot(nrow, ncol, i)
        p.title(program)
        munobs = plan.tile_status == 'unobs'
        p.scatter(tra[m & munobs], tiles.tileDEC[m & munobs],
                  alpha=1.0, color='green', s=5, label='unobs ({})'.format((m & munobs).sum()))
        mcomplete = plan.tile_status == 'done'
        p.scatter(tra[m & mcomplete], tiles.tileDEC[m & mcomplete],
                  alpha=0.1, color='gray', s=5, label='done ({})'.format((m & mcomplete).sum()))
        mpending = ~(munobs | mcomplete)
        p.scatter(tra[m & mpending], tiles.tileDEC[m & mpending],
                  alpha=1, color='orange', s=20, label='pending ({})'.format((m & mpending).sum()))
        p.plot(tra[idx], tiles.tileDEC[idx], 'k--')
        p.scatter(tra[m & mtonight], tiles.tileDEC[m & mtonight],
                  alpha=1, facecolors='none', edgecolors='red',
                  s=50, linewidth=3)
        p.xlim(loff+360, loff)
        p.legend(loc=1)
    p.suptitle(title)
    p.savefig('plan.png')
    p.show()


def make_tiles(tilelist, plan, nprocess=10, logdir=None):
    import glob
    hpdir = os.environ['FA_HOLDING_PEN']
    if len(hpdir) == 0:
        raise ValueError('FA_HOLDING_PEN must be set')
    allhpfiles = glob.glob(os.path.join(hpdir, '**'), recursive=True)
    if desisurvey.utils.yesno('Deleting %d files in %s, continue?' %
                              (len(allhpfiles), hpdir)):
        import shutil
        for fn in glob.glob(os.path.join(hpdir, '*')):
            shutil.rmtree(fn)
    tiles = desisurvey.tiles.get_tiles()
    tilelist = np.array(tilelist)
    idx = tiles.index(tilelist)
    m = plan.tile_status[idx] == 'unobs'
    obstiles = ' '.join([str(t) for t in tilelist[~m]])
    print('Skipping tiles with status != unobs: %s' % obstiles)
    tilelist = tilelist[m]
    from multiprocessing import Pool
    pool = Pool(nprocess)
    tilestrings = np.array([str(t) for t in tilelist])
    print('Starting to write fiberassign tiles for ' +
          ' '.join(tilestrings))
    retcode1 = pool.map(partial(worktile, logdir=logdir), tilelist)
    retcode1 = np.array(retcode1)
    if np.any(retcode1 != 0):
        print('Weird return code for ' +
              ' '.join(tilestrings[retcode1 != 0]))
    print('Starting to write QA...')
    retcode2 = pool.map(partial(workqa, logdir=logdir), tilelist)
    retcode2 = np.array(retcode2)
    if np.any(retcode2 != 0):
        print('Weird return code for ' +
              ' '.join(tilestrings[retcode2 != 0]))
    print('All done.')


def run_plan(night=None, nts_dir=None, verbose=False, survey=None,
             seeing=1.1, table=False, azrange=None, makebackuptiles=False):
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
    night = nts.night
    t0 = nts.scheduler.night_ephem['brightdusk']
    nts_dir, _ = os.path.split(nts.obsplan)
    if not verbose:
        desiutil.log.get_logger().setLevel('WARNING')
        nts.log.setLevel('WARNING')
    previoustiles = []
    ephem = nts.scheduler.night_ephem
    changes = nts.scheduler.night_changes
    programs = nts.scheduler.night_programs
    mayall = desisurvey.ephem.get_mayall(noar=True)
    mayall.date = desisurvey.utils.local_noon_on_date(night).datetime
    mayall.horizon = '-6:00'
    sun = pyephem.Sun()
    mjd6degdusk = Time(
        mayall.next_setting(sun, use_center=True).datetime()).mjd
    mayall.horizon = '-10:00'
    mjd10degdusk = Time(
        mayall.next_setting(sun, use_center=True).datetime()).mjd
    extratimes = {'6degdusk': mjd6degdusk, '10degdusk': mjd10degdusk}
    night_labels = np.array(['noon', '6 deg dusk', '10 deg dusk',
                             '12 deg dusk',
                             '15 deg dusk',
                             '15 deg dawn', '12 deg dawn',
                             'moonrise', 'moonset'])
    night_names = np.array(['noon', '6degdusk', '10degdusk',
                            'brightdusk', 'dusk',
                            'dawn', 'brightdawn',
                            'moonrise', 'moonset'])
    night_times = np.array([ephem[name]
                            if name not in extratimes else extratimes[name]
                            for name in night_names])
    s = np.argsort(night_times)
    print(nts.scheduler.night)
    for name, tt in zip(night_labels[s], night_times[s]):
        print('%11s %s' % (name, mjd_to_azstr(tt)))

    print('local   lst   cond   tile    ra   dec    program   x fac  tot  split ' +
          'refft')
    current_ra = None
    current_dec = None
    constraints = dict(azrange=azrange)
    tilelist = []
    while t0 < nts.scheduler.night_ephem['brightdawn']:
        cidx = np.interp(t0+300/86400, changes, np.arange(len(changes)))
        cidx = int(np.clip(cidx, 0, len(programs)-1))
        cond = programs[cidx]
        moon_up_factor = getattr(nts.config.conditions, cond).moon_up_factor()
        expdict = dict(mjd=t0, previoustiles=previoustiles)
        speed = {
            'speed_%s_nts' % k.lower():
            nts.ETC.weather_factor(seeing, 1., moon_up_factor, sbprof=prof)
            for k, prof in (('DARK', 'ELG'), ('BRIGHT', 'BGS'),
                            ('BACKUP', 'PSF'))}
        conddict = dict(skylevel=moon_up_factor, seeing=seeing,
                        sky_ra=current_ra, sky_dec=current_dec, **speed)
        res = nts.next_tile(exposure=expdict, conditions=conddict,
                            speculative=True, constraints=constraints)
        if not res['foundtile']:
            t0 += 10*60/60/60/24
            continue
        tilelist.append(int(res['fiberassign']))
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
                ('%s %5.1f %6s %6d %5.1f %5.1f %10s %3.1f %3.1f '
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

    planplot(tilelist, nts.planner,
             title='%4d%02d%02d' % (night.year, night.month, night.day))
    if makebackuptiles:
        backuplogdir = os.path.join(
            os.path.dirname(desisurvey.config.Configuration().file_name),
            'backup-tile-logs')
        if not os.path.exists(backuplogdir):
            os.mkdir(backuplogdir)
        make_tiles(tilelist, nts.planner, logdir=backuplogdir)
        check_permissions()


def check_permissions():
    import stat
    hpdir = os.environ.get('FA_HOLDING_PEN', None)
    log = desiutil.log.get_logger()
    bad = False
    if hpdir is None:
        log.warning('FA_HOLDING_PEN is not set!')
    else:
        dirs = os.listdir(hpdir)
        for dirname in dirs:
            tdirname = os.path.join(hpdir, dirname)
            mode = stat.S_IMODE(
                os.lstat(tdirname).st_mode)
            if mode != 0o2775:
                bad = True
                log.error(f'{tdirname} has bad permissions!')
    if bad:
        log.error('Some files in the holding pen have bad permissions!'
                  'Raise the alarm in #survey-ops!')


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
    parser.add_argument('--makebackuptiles', default=False,
                        action='store_true',
                        help='Design backup tiles and place in holding pen')
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):
    run_plan(night=args.night, nts_dir=args.nts_dir,
             survey=args.survey, verbose=args.verbose, seeing=args.seeing,
             table=args.table, azrange=args.azrange,
             makebackuptiles=args.makebackuptiles)
