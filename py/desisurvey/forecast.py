"""Simple forecast of survey progress and margin.
"""
from __future__ import print_function, division, absolute_import

import os
import collections
import datetime

import numpy as np

from astropy.table import Table
import astropy.units as u
from astropy.time import Time

import desimodel.io

import desisurvey.config
import desisurvey.ephem
import desisurvey.etc
import desisurvey.utils
import desisurvey.plan


class Forecast(object):
    """Compute a simple forecast of survey progress and margin.

    Based on config, ephemerides, tiles.

    Parameters
    ----------
    start_date : datetime.date
        Forecast for survey that starts on the evening of this date.
    stop_date : datetime.date
        Forecast for survey that stops on the morning of this date.
    use_twilight : bool
        Include twilight time in the forecast scheduled time?
    weather : array or None
        1D array of nightly weather factors (0-1) to use, or None to use
        :func:`desisurvey.plan.load_weather`. The array length must equal
        the number of nights in [start,stop). Values are fraction of the
        night with the dome open (0=never, 1=always). Use
        1 - :func:`desimodel.weather.dome_closed_fractions` to lookup
        suitable corrections based on historical weather data.
    design_hourangle : array or None
        1D array of design hour angles to use in degrees, or None to use
        :func:`desisurvey.plan.load_design_hourangle`.
    """
    def __init__(self, start_date=None, stop_date=None, use_twilight=False,
                 weather=None, design_hourangle=None):
        config = desisurvey.config.Configuration()
        if start_date is None:
            start_date = config.first_day()
        else:
            start_date = desisurvey.utils.get_date(start_date)
        if stop_date is None:
            stop_date = config.last_day()
        else:
            stop_date = desisurvey.utils.get_date(stop_date)
        self.num_nights = (stop_date - start_date).days
        if self.num_nights <= 0:
            raise ValueError('Expected start_date < stop_date.')

        self.use_twilight = use_twilight
        # Look up the tiles to observe.
        tiles = desisurvey.tiles.get_tiles()
        self.tiles = tiles
        if design_hourangle is None:
            self.design_hourangle = np.zeros(tiles.ntiles)
        else:
            if len(design_hourangle) != tiles.ntiles:
                raise ValueError('Array design_hourangle has wrong length.')
            self.design_hourangle = np.asarray(design_hourangle)
        # Get weather factors.
        if weather is None:
            self.weather = desisurvey.utils.get_average_dome_closed_fractions(
                start_date, stop_date)
            self.weather = 1-self.weather
        else:
            self.weather = np.asarray(weather)
        if self.weather.shape != (self.num_nights,):
            raise ValueError('Array weather has wrong shape.')
        # Get the design hour angles.
        if design_hourangle is None:
            self.design_hourangle = desisurvey.plan.load_design_hourangle()
        else:
            self.design_hourangle = np.asarray(design_hourangle)
        if self.design_hourangle.shape != (tiles.ntiles,):
            raise ValueError('Array design_hourangle has wrong shape.')
        # Compute airmass at design hour angles.
        self.airmass = tiles.airmass(self.design_hourangle)
        airmass_factor = desisurvey.etc.airmass_exposure_factor(self.airmass)
        # Load ephemerides.
        ephem = desisurvey.ephem.get_ephem()
        # Compute the expected available and scheduled hours per program.
        scheduled = ephem.get_program_hours(include_twilight=use_twilight)
        available = scheduled * self.weather
        self.cummulative_days = np.cumsum(available, axis=1) / 24.
        # Calculate program parameters.
        ntiles, tsched, openfrac, dust, airmass, nominal = [], [], [], [], [], []
        for program in tiles.programs:
            tile_sel = tiles.program_mask[program]
            ntiles.append(np.count_nonzero(tile_sel))
            progindx = tiles.program_index[program]
            scheduled_sum = scheduled[progindx].sum()
            tsched.append(scheduled_sum)
            openfrac.append(available[progindx].sum() / scheduled_sum)
            dust.append(tiles.dust_factor[tile_sel].mean())
            airmass.append(airmass_factor[tile_sel].mean())
            nominal.append(
                (getattr(config.programs, program).efftime)().to(u.s).value)
        # Build a table of all forecasting parameters.
        df = collections.OrderedDict()
        self.df = df
        df['Number of tiles'] = np.array(ntiles)
        df['Scheduled time (hr)'] = np.array(tsched)
        df['Dome open fraction'] = np.array(openfrac)
        self.set_overheads()
        df['Nominal exposure (s)'] = np.array(nominal)
        df['Dust factor'] = np.array(dust)
        df['Airmass factor'] = np.array(airmass)
        self.set_factors()

    def summary(self, width=7, prec=5, separator='-'):
        """Print a summary table of the forecast parameters.
        """
        # Find the longest key and calculate the row length.
        nprog = len(self.tiles.programs)
        maxlen = np.max([len(key) for key in self.df])
        rowlen = maxlen + (1 + width) * nprog
        # Build a format string for each row.
        header = ' ' * maxlen + ' {{:>{}s}}'.format(width) * nprog
        row = '{{:>{}s}}'.format(maxlen) + ' {{:{}.{}g}}'.format(width, prec) * nprog
        # Print the header.
        print(header.format(*self.tiles.programs))
        print(separator * rowlen)
        # Print each row.
        for key, values in self.df.items():
            print(row.format(key, *values))
        print(separator * rowlen)

    def set_overheads(self, update_margin=True,
                      setup={'DARK': 200, 'GRAY': 200, 'BRIGHT': 150, 'BACKUP': 150},
                      split={'DARK': 100, 'GRAY': 100, 'BRIGHT':  75, 'BACKUP': 150},
                      dead ={'DARK':  20, 'GRAY': 100, 'BRIGHT':  10, 'BACKUP': 10}):
        df = self.df
        df['Setup overhead / tile (s)'] = np.array([setup[p] for p in self.tiles.programs])
        df['Cosmic split overhead / tile (s)'] = np.array([split[p] for p in self.tiles.programs])
        df['Operations overhead / tile (s)'] = np.array([dead[p] for p in self.tiles.programs])
        df['Average available / tile (s)'] = (
            df['Scheduled time (hr)'] * df['Dome open fraction'] /
            # Avoid division by zero for a program with no tiles.
            np.maximum(1, df['Number of tiles']) * 3600 -
            df['Setup overhead / tile (s)'] -
            df['Cosmic split overhead / tile (s)'] -
            df['Operations overhead / tile (s)'])
        self.update()

    def set_factors(self, update_margin=True,
                       moon    = {'DARK': 1.00, 'GRAY': 1.5, 'BRIGHT': 3.6, 'BACKUP': 6},
                       weather = {'DARK': 1.22, 'GRAY': 1.20, 'BRIGHT': 1.16, 'BACKUP': 6}):
        df = self.df
        df['Moon factor'] = np.array([moon[p] for p in self.tiles.programs])
        df['Weather factor'] = np.array([weather[p] for p in self.tiles.programs])
        df['Average required / tile (s)'] = (
            df['Nominal exposure (s)'] *
            df['Dust factor'] *
            df['Airmass factor'] *
            df['Moon factor'] *
            df['Weather factor'])
        self.update()

    def update(self):
        df = self.df
        if 'Average available / tile (s)' not in df: return
        if 'Average required / tile (s)' not in df: return
        df['Exposure time margin (%)'] = 100 * (
            df['Average available / tile (s)'] /
            df['Average required / tile (s)'] - 1)
        self.program_progress = np.zeros((len(self.tiles.programs),
                                          self.num_nights))
        for program in self.tiles.programs:
            progidx = self.tiles.program_index[program]
            dtexp = (
                df['Average required / tile (s)'] +
                df['Setup overhead / tile (s)'] +
                df['Cosmic split overhead / tile (s)'] +
                df['Operations overhead / tile (s)']
                )[progidx] / 86400.
            # Calculate the mean time between exposures for this program.
            progress = self.cummulative_days[progidx] / dtexp
            # Compute progress assuming tiles are observed in pass order,
            # separated by exactly dtexp.
            ntiles_observed = 0
            ntiles = np.sum(self.tiles.program_mask[program])
            self.program_progress[progidx] = np.clip(
                progress - ntiles_observed, 0, ntiles)
            ntiles_observed += ntiles


def forecast_plots(tmain=None, exps=None, surveyopsdir=None,
                   include_backup=False, cfgfile=None, ratio=False,
                   nownight=None):
    from matplotlib import pyplot as p
    if surveyopsdir is None:
        surveyopsdir = os.environ['DESI_SURVEYOPS']
    if tmain is None:
        tmain = Table.read(os.path.join(
            surveyopsdir, 'ops', 'tiles-main.ecsv'))
    if exps is None:
        exps = Table.read(os.path.join(
            surveyopsdir, 'ops', 'exposures.ecsv'))
    cfg = desisurvey.config.Configuration(cfgfile)
    ephem = desisurvey.ephem.get_ephem()
    scheduled = ephem.get_program_hours()
    closefracs = desisurvey.utils.get_average_dome_closed_fractions(
        cfg.first_day(), cfg.last_day())
    programnames = ['DARK', 'GRAY', 'BRIGHT']
    weatheradjustedhours = [
        (1-closefracs) * sched / getattr(cfg.conditions, prog).moon_up_factor()
        for sched, prog in zip(scheduled, programnames)]
    weatheradjustedhours = np.sum(weatheradjustedhours, axis=0)
    if nownight is not None:
        # add a pseudo-exposure on a tile.  Ugly hack!
        from copy import deepcopy
        t1000ind = np.flatnonzero(exps['TILEID'] == 1000)[0]
        pseudoexp = deepcopy(exps[t1000ind])
        pseudoexp['NIGHT'] = nownight
        pseudoexp['EFFTIME'] = 0
        exps.add_row(pseudoexp)

    cz = desisurvey.utils.cos_zenith(tmain['DESIGNHA']*u.deg,
                                     tmain['DEC']*u.deg)
    am = desisurvey.utils.cos_zenith_to_airmass(cz)
    amfac = desisurvey.etc.airmass_exposure_factor(am)
    dustfac = desisurvey.etc.dust_exposure_factor(tmain['EBV_MED'])
    cost = amfac*dustfac
    costetime = cost*desisurvey.tiles.get_nominal_program_times(
        tmain['PROGRAM'])
    include = tmain['IN_DESI'] != 0
    if not include_backup:
        include &= tmain['PROGRAM'] != 'BACKUP'
    tileid_to_rownum = {t: i for i, t in enumerate(tmain['TILEID'])}
    lastnight = np.zeros(len(tmain), dtype='i4')
    for i in range(len(exps)):
        rownum = tileid_to_rownum.get(exps['TILEID'][i], -1)
        if rownum < 0:
            continue
        lastnight[rownum] = max(
            [lastnight[rownum], int(exps['NIGHT'][i])])
    lastday = cfg.last_day()
    lastnight[lastnight == 0] = (
        lastday.year*10000 + lastday.month*100 + lastday.day)
    lastmjd = Time(['-'.join([str(n)[:4], str(n)[4:6], str(n)[6:8]])
                    for n in lastnight]).mjd

    startdatetime = datetime.datetime.combine(
        cfg.first_day(), datetime.time())
    nightind = (lastmjd - Time(startdatetime).mjd).astype('i4')
    p.plot(
        np.cumsum(weatheradjustedhours)/np.sum(weatheradjustedhours)*100,
        label='adjusted % of time elapsed', color='tab:blue',
        linestyle='--')
    for i in range(5):
        p.axvline(365*(i+1), linestyle='--', color='gray')
        p.text(365*(i+1)-20, 0.75, f'{2021+i+1}-05-14', rotation=90,
               transform=p.gca().get_xaxis_transform(),
               bbox=dict(facecolor='white', pad=3, edgecolor='none',
                         alpha=0.5))
    countdone = tmain['DONEFRAC'].copy()
    countdone = np.clip(countdone, 0, 1)
    countdone[(tmain['STATUS'] == 'done') |
              (tmain['STATUS'] == 'obsend')] = 1
    s = np.argsort(nightind)
    doneetime = costetime*include*countdone
    dark = tmain['PROGRAM'] == 'DARK'
    bright = tmain['PROGRAM'] == 'BRIGHT'
    overall = dark | bright
    doneetimefracdict = dict()
    for name, mask in dict(dark=dark, bright=bright, overall=overall).items():
        doneetimefracdict[name] = np.cumsum((doneetime*mask)[s])
        doneetimefracdict[name] /= np.sum(costetime*include*mask)
    colors = dict(bright='tab:orange', dark='black', overall='tab:blue')
    maxind = np.max(np.flatnonzero(countdone[s] > 0))
    for label, mask in doneetimefracdict.items():
        p.plot(nightind[s[:maxind+1]], doneetimefracdict[label][:maxind+1]*100,
               label=f'% {label} done',
               color=colors[label])
    p.xlabel('nights since 2021-05-14')
    p.ylabel('Percentage complete')

    p.axvline(nightind[s[maxind]], linestyle='--', color='gray')
    nownightstr = str(lastnight[s[maxind]])
    nownightstr = nownightstr[:4]+'-'+nownightstr[4:6]+'-'+nownightstr[6:8]
    p.text(nightind[s[maxind]]+10, 0.75, f'{nownightstr}',
           rotation=90,
           transform=p.gca().get_xaxis_transform(),
           bbox=dict(facecolor='white', pad=3, edgecolor='none',
                     alpha=0.5))

    darkfrac = (
        np.sum(doneetime*include*dark) /
        np.sum(costetime*include*dark))
    brightfrac = (
        np.sum(doneetime*include*bright) /
        np.sum(costetime*include*bright))
    overallfrac = (
        np.sum(doneetime*include*overall) /
        np.sum(costetime*include*overall))
    lastnight = nightind[s[maxind]]
    elapsedfrac = (np.sum(weatheradjustedhours[:lastnight]) /
                   np.sum(weatheradjustedhours))
    p.text(0.95, 0.05,
           ('adj. time elapsed: {:5.2%}\n'
            'overall: {:5.2%}\n'
            'dark: {:5.2%}\n'
            'bright: {:5.2%}\n'
            'implied margin: {:5.2%}').format(
                elapsedfrac, overallfrac, darkfrac, brightfrac,
                overallfrac/elapsedfrac - 1),
           ha='right', bbox=dict(facecolor='white', pad=10, edgecolor='none'),
           transform=p.gca().transAxes)
    p.xlim(0, 5*365.25)
    p.ylim(0, 100)
    p.legend()
    if ratio:
        p.twinx()
        timefrac = np.interp(
            nightind[s[:maxind+1]],
            np.arange(len(weatheradjustedhours)),
            np.cumsum(weatheradjustedhours)/np.sum(weatheradjustedhours))
        p.plot(nightind[s[:maxind]],
               (doneetimefracdict['overall'][:maxind+1]/timefrac-1)*100,
               color='green', linestyle='--', label='overall margin')
        p.ylim(-10, 300)
        p.ylabel('overall margin')
        print((timefrac-1)*100)
    print('Dark months ahead: ', (darkfrac - elapsedfrac)*55)


def summarize_daterange(
        startdate, enddate, exps=None, surveyopsdir=None,
        tmain=None, cfgfile=None):
    # we have some date range; it corresponds to a certain amount
    # of nominal elapsed time.
    # we have the number of exposures we actually finished in
    # that time.
    # we compare those?
    if surveyopsdir is None:
        surveyopsdir = os.environ['DESI_SURVEYOPS']
    if exps is None:
        exps = Table.read(os.path.join(
            surveyopsdir, 'ops', 'exposures.ecsv'))
    exps = exps[(exps['NIGHT'] >= str(startdate)) &
                (exps['NIGHT'] < str(enddate))]
    if tmain is None:
        tmain = Table.read(os.path.join(
            surveyopsdir, 'ops', 'tiles-main.ecsv'))

    cfg = desisurvey.config.Configuration(cfgfile)
    ephem = desisurvey.ephem.get_ephem()
    scheduled = ephem.get_program_hours(start_date=cfg.first_day(),
                                        stop_date=cfg.last_day())
    closefracs = desisurvey.utils.get_average_dome_closed_fractions(
        cfg.first_day(), cfg.last_day())
    programnames = ['DARK', 'GRAY', 'BRIGHT']
    weatheradjustedhours = [
        (1-closefracs) * sched / getattr(cfg.conditions, prog).moon_up_factor()
        for sched, prog in zip(scheduled, programnames)]
    weatheradjustedhours = np.sum(weatheradjustedhours, axis=0)

    startdaynum = desisurvey.utils.day_number(
        desisurvey.utils.str_to_night(startdate))
    enddaynum = desisurvey.utils.day_number(
        desisurvey.utils.str_to_night(enddate))
    elapsedfrac = (np.sum(weatheradjustedhours[startdaynum:enddaynum]) /
                   np.sum(weatheradjustedhours))

    cz = desisurvey.utils.cos_zenith(tmain['DESIGNHA']*u.deg,
                                     tmain['DEC']*u.deg)
    am = desisurvey.utils.cos_zenith_to_airmass(cz)
    amfac = desisurvey.etc.airmass_exposure_factor(am)
    dustfac = desisurvey.etc.dust_exposure_factor(tmain['EBV_MED'])
    cost = amfac*dustfac
    costetime = cost*desisurvey.tiles.get_nominal_program_times(
        tmain['PROGRAM'])
    include = tmain['IN_DESI'] != 0
    tileid_to_rownum = {t: i for i, t in enumerate(tmain['TILEID'])}
    lastnight = np.zeros(len(tmain), dtype='i4')
    totprogramtime = {
        p: np.sum(costetime*include*(tmain['PROGRAM'] == p))
        for p in ['DARK', 'BRIGHT', 'BACKUP']}

    programtime = dict()
    programtile = dict()
    programetcefftime = dict()
    # 20210530 has invalid PROGRAM entries in the NTS exposures file
    expsprogram = np.array([
        tmain['PROGRAM'][tileid_to_rownum.get(t, -1)].lower()
        for t in exps['TILEID']])
    for program in ['DARK', 'BRIGHT', 'BACKUP']:
        goaltime = getattr(cfg.programs, program).efftime().to(u.s).value
        m = (expsprogram == program.lower()) & (exps['EFFTIME'] > 0)
        m2 = (expsprogram == program.lower()) & (exps['EFFTIME_ETC'] > 0)
        tileexpefftime = np.bincount(
            exps['TILEID'][m], weights=exps['EFFTIME'][m])
        tileexpefftime[tileexpefftime > 0.85*goaltime] = goaltime
        tileexpetcefftime = np.bincount(exps['TILEID'][m2],
                                        weights=exps['EFFTIME_ETC'][m2])
        utileid = np.flatnonzero(tileexpefftime)
        ntile = 0
        time = 0
        etcefftime = 0
        for tileid in utileid:
            r = tileid_to_rownum.get(tileid, -1)
            tcost = 1.6 if r == -1 else costetime[r]
            etcefftime += tcost*tileexpetcefftime[tileid]/goaltime
            if r < 0:
                continue
            time += costetime[r]*include[r]*tileexpefftime[tileid]/goaltime
            ntile += tileexpefftime[tileid]/goaltime
        programtime[program] = time
        programtile[program] = ntile
        programetcefftime[program] = etcefftime
    print(f'Elapsed fraction of five year survey, weather adjusted: '
          f'{100*elapsedfrac:5.2f}%')
    for p in programtime:
        print(f'{p:8s} {programtile[p]:7.1f} '
              f'{100*programtime[p]/totprogramtime[p]:5.2f}% '
              f'(naive sum(efftime_etc): '
              f'{100*programetcefftime[p]/totprogramtime[p]:5.2f}%)')

def surveysim_exps_to_exps_and_tmain(expsfn, tmain, maxnight=None):
    from astropy.io import fits
    exps = fits.getdata(expsfn, 'EXPOSURES')
    tdata = fits.getdata(expsfn, 'TILEDATA')

    expsout = np.zeros(len(exps),
                       dtype=[('NIGHT', 'U8'), ('TILEID', 'i4'),
                              ('EFFTIME', 'f4')])
    expsout['TILEID'] = exps['TILEID']
    nightstr = []
    for mjd in exps['MJD']:
        dt = Time(mjd, format='mjd').datetime
        nightstr.append('%04d%02d%02d' % (dt.year, dt.month, dt.day))
    expsout['NIGHT'] = nightstr
    if maxnight is not None:
        m = expsout['NIGHT'] <= maxnight
        expsout = expsout[m]
        exps = exps[m]

    tmainout = tmain.copy()
    donefrac = np.bincount(exps['TILEID'], weights=exps['DSNR2FRAC'],
                           minlength=max(tdata['TILEID'])+1)
    tmainout['DONEFRAC'] = donefrac[tmainout['TILEID']]
    mnotstarted = tmainout['DONEFRAC'] == 0
    mdone = tmainout['DONEFRAC'] > 0.85
    tmainout['STATUS'] = 'unobs'
    tmainout['STATUS'][~mnotstarted] = 'obsstart'
    tmainout['STATUS'][mdone] = 'done'

    return expsout, tmainout
