"""Script wrapper for initializing survey planning and scheduling.

This is normally run once, at the start of the survey, and saves its results
to a FITS file surveyinit.fits.  With the default parameters, the running time
is about 25 minutes.

This script will calculate the ephemerides and expected weather (dome open
fraction) for 2019-2025, then calculate design hour angles for the
nominal survey dates.  The results are saved in a file (normally
``surveyinit.fits``) and then do not need to be recalculated ever again.

To run this script from the command line, use the ``surveyinit`` entry point
that is created when this package is installed, and should be in your shell
command search path.
"""
from __future__ import print_function, division, absolute_import

import os
import argparse

import numpy as np

import astropy.io.fits as fits
import astropy.table

import desiutil.log

import desimodel.weather

import desisurvey.utils
import desisurvey.ephem
import desisurvey.tiles
import desisurvey.optimize


def parse(options=None):
    """Parse command-line options for running survey planning.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--verbose', action='store_true',
        help='display log messages with severity >= info')
    parser.add_argument(
        '--debug', action='store_true',
        help='display log messages with severity >= debug (implies verbose)')
    parser.add_argument(
        '--include-twilight', action='store_true',
        help='Include twilight in available LST')
    parser.add_argument(
        '--recalc', action='store_true',
        help='recalculate even when previous calculations are available')
    parser.add_argument(
        '--recalc-ephem', action='store_true',
        help='recalculate ephemerides tabulation')
    parser.add_argument(
        '--recalc-lst', action='store_true',
        help='recalculate LST optimization')
    parser.add_argument(
        '--nbins', type=int, default=192, metavar='N',
        help='number of LST bins to use')
    parser.add_argument(
        '--init', choices=('zero', 'flat'), default='flat',
        help='method to assign initial HA to each tile')
    parser.add_argument(
        '--adjust', default=1.0, metavar='DEG',
        help='tile HA adjustment (deg) per iteration, anneals each cycle')
    parser.add_argument(
        '--smooth', default=0.05, metavar='S',
        help='amount to smooth HA assignments, anneals each cycle')
    parser.add_argument(
        '--anneal', default=0.95, metavar='A',
        help='decrease adjust, smooth by this factor after each cycle')
    parser.add_argument(
        '--max-rmse', default=0.02, metavar='MAX',
        help='continue cycles until root mean square error < MAX')
    parser.add_argument(
        '--epsilon', default=0.01, metavar='EPS',
        help='continue cycles until fractional score improvement < EPS')
    parser.add_argument(
        '--max-cycles', type=int, default=100,
        help='maximum number of annealing cycles for each program')
    parser.add_argument(
        '--dark-stretch', type=float, default=1.2, metavar='S',
        help='stretch DARK exposure times by this factor')
    parser.add_argument(
        '--gray-stretch', type=float, default=1.3, metavar='S',
        help='stretch GRAY exposure times by this factor')
    parser.add_argument(
        '--bright-stretch', type=float, default=1.5, metavar='S',
        help='stretch BRIGHT exposure times by this factor')
    parser.add_argument(
        '--savelst', default='lst_optimized.fits', metavar='NAME',
        help='name of FITS output where LST distributions are saved')
    parser.add_argument(
        '--savetiles', default=None, metavar='NAME',
        help=('name of ecsv output where updated tile file is saved, '
              'default: same as tiles file, but in DESISURVEY_OUTPUT.'))
    parser.add_argument(
        '--output-path', default=None, metavar='PATH',
        help='output path to use instead of config.output_path')
    parser.add_argument(
        '--tiles-file', default=None, metavar='TILES',
        help='name of tiles file to use instead of config.tiles_file')
    parser.add_argument(
        '--config-file', default='config.yaml', metavar='CONFIG',
        help='input configuration file')
    parser.add_argument(
        '--completed', default=None,
        help='filename with information on already completed tiles')
    parser.add_argument(
        '--include-weather', default=True, type=bool,
        help='Use past weather to discount available LST when planning.')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def calculate_initial_plan(args):
    """Calculate the initial survey plan.

    Use :func:`desisurvey.plan.load_weather` and
    :func:`desisurvey.plan.load_design_hourangles` to retrieve
    these data from the saved plan.

    Parameters
    ----------
    args : object
        Object with attributes for parsed command-line arguments.
    """
    log = desiutil.log.get_logger()
    config = desisurvey.config.Configuration()
    tiles = desisurvey.tiles.get_tiles()
    ephem = desisurvey.ephem.get_ephem()

    # Initialize the output file to write.
    hdus = fits.HDUList()
    hdr = fits.Header()

    # Calculate average weather factors for each day covered by
    # the ephemerides.
    first = desisurvey.ephem.START_DATE
    last = desisurvey.ephem.STOP_DATE
    years = np.arange(2007, 2018)
    fractions = []
    for year in years:
        fractions.append(
            desimodel.weather.dome_closed_fractions(first, last, replay='Y{}'.format(year)))
    from scipy.ndimage import gaussian_filter
    fractions = gaussian_filter(fractions, 7, mode='wrap')
    weather = 1 - np.mean(fractions, axis=0)
    # Save the weather fractions as the primary HDU.
    hdr['FIRST'] = first.isoformat()
    hdr['YEARS'] = ','.join(['{}'.format(yr) for yr in years])
    start = config.first_day()
    stop = config.last_day()
    assert start >= first and stop <= last
    hdr['START'] = start.isoformat()
    hdr['STOP'] = stop.isoformat()
    hdr['TWILIGHT'] = args.include_twilight
    hdus.append(fits.ImageHDU(weather, header=hdr, name='WEATHER'))

    # Calculate the distribution of available LST in each program
    # during the nominal survey [start, stop).
    ilo, ihi = (start - first).days, (stop - first).days
    if args.include_weather:
        tweather = weather[ilo:ihi]
    else:
        tweather = None
    lst_hist, lst_bins = ephem.get_available_lst(
        nbins=args.nbins, weather=tweather,
        include_twilight=args.include_twilight)
    lst_centers = 0.5*(lst_bins[:-1]+lst_bins[1:])

    if tiles.nogray:
        assert not np.any(tiles.tileobsconditions == 'GRAY')
        new_lst_hist = lst_hist.copy()
        new_lst_hist[0, :] = lst_hist[0, :] + lst_hist[1, :]
        new_lst_hist[1, :] = lst_hist[2, :]
        lst_hist = new_lst_hist[0:2, :].copy()

    # Initialize the output results table.
    if tiles.nogray:
        conditions = ['DARK', 'BRIGHT']
    else:
        conditions = ['DARK', 'GRAY', 'BRIGHT']
    design = astropy.table.Table()
    design['INIT'] = np.zeros(tiles.ntiles)
    design['HA'] = np.zeros(tiles.ntiles)
    design['TEXP'] = np.zeros(tiles.ntiles)
    design['TILEID'] = tiles.tileID
    design['RA'] = tiles.tileRA
    design['DEC'] = tiles.tileDEC
    design['PROGRAM'] = tiles.tileprogram
    tiletab = tiles.read_tiles_table()

    # Optimize each program separately.
    stretches = dict(
        DARK=args.dark_stretch,
        GRAY=args.gray_stretch,
        BRIGHT=args.bright_stretch)

    if tiles.nogray:
        stretches.pop('GRAY')

    tile_is_assignable = np.zeros(tiles.ntiles, dtype='bool')
    for condition in conditions:
        tile_is_assignable |= tiles.allowed_in_conditions(condition)
    if ~np.all(tile_is_assignable):
        log.info('Warning: some tiles are not observable in '
                 'gray/dark/bright.  These will not be observable by the NTS '
                 'by default.')
        badprogram = np.unique(tiles.tileprogram[~tile_is_assignable])
        log.info('Problematic programs: {}'.format(' '.join(badprogram)))

    for index, condition in enumerate(conditions):
        sel = tiles.allowed_in_conditions(condition) & tiles.in_desi
        sel = sel & (tiletab['DONEFRAC'] < 1)
        if not np.any(sel):
            log.info('Skipping {} program with no tiles.'.format(condition))
            continue
        # Initialize an LST summary table.
        table = astropy.table.Table(meta={'ORIGIN': lst_bins[0]})
        table['LST'] = lst_centers
        table['AVAIL'] = lst_hist[index]
        # Initailize an optimizer for this program.
        opt = desisurvey.optimize.Optimizer(
            condition, lst_bins, lst_hist[index], init=args.init, center=None,
            stretch=stretches[condition], completed=args.completed,
            subset=tiles.tileID[sel])
        table['INIT'] = opt.plan_hist.copy()
        design['INIT'][sel] = opt.ha_initial
        # Initialize annealing cycles.
        ncycles = 0
        binsize = 360. / args.nbins
        frac = args.adjust / binsize
        smoothing = args.smooth
        # Loop over annealing cycles.
        while ncycles < args.max_cycles:
            start_score = opt.eval_score(opt.plan_hist)
            for i in range(opt.ntiles):
                opt.improve(frac)
            if smoothing > 0:
                opt.smooth(alpha=smoothing)
            stop_score = opt.eval_score(opt.plan_hist)
            delta = (stop_score - start_score) / start_score
            RMSE = opt.RMSE_history[-1]
            loss = opt.loss_history[-1]
            log.info(
                '[{:03d}] dHA={:5.3f}deg '.format(ncycles + 1, frac * binsize) +
                'RMSE={:6.2f}% LOSS={:5.2f}% delta(score)={:+5.1f}%'
                .format(1e2*RMSE, 1e2*loss, 1e2*delta))
            # Both conditions must be satisfied to terminate.
            if RMSE < args.max_rmse and delta > -args.epsilon:
                break
            # Anneal parameters for next cycle.
            frac *= args.anneal
            smoothing *= args.anneal
            ncycles += 1
        plan_sum = opt.plan_hist.sum()
        avail_sum = opt.lst_hist_sum
        margin = (avail_sum - plan_sum) / plan_sum
        log.info('{} plan uses {:.1f}h with {:.1f}h avail ({:.1f}% margin).'
                 .format(condition, plan_sum, avail_sum, 1e2 * margin))
        # Save planned LST usage.
        table['PLAN'] = opt.plan_hist
        table['HA0'] = opt.plan_hist_ha0
        hdus.append(fits.BinTableHDU(table, name=condition))

        # Calculate exposure times in (solar) seconds.
        texp, _ = opt.get_exptime(opt.ha)
        texp *= 24. * 3600. / 360. * 0.99726956583
        # Save results for this program.
        design['HA'][sel] = opt.ha
        design['TEXP'][sel] = texp

    hdus.append(fits.BinTableHDU(design, name='DESIGN'))
    fullname = config.get_path(args.savelst)
    hdus.writeto(fullname, overwrite=True)
    log.info('Saved initial plan to "{}".'.format(fullname))

    # add a DESIGNHA column or overwrite one to an existing tile file.
    tiletab['DESIGNHA'] = np.zeros(len(tiletab), dtype='f4')
    tiletab['DESIGNHA'].format = '%7.3f'
    tiletab['DESIGNHA'].unit = tiletab['RA'].unit
    tiletab['DESIGNHA'].description = 'Design hour angles'
    _, mt, md = np.intersect1d(tiletab['TILEID'], design['TILEID'],
                               return_indices=True)
    tiletab['DESIGNHA'][mt] = design['HA'][md]
    # drop unnecessary columns
    dropcolumns = ['AIRMASS', 'STAR_DENSITY', 'EXPOSEFAC', 'OBSCONDITIONS',
                   'IMAGEFRAC_G', 'IMAGEFRAC_R', 'IMAGEFRAC_Z',
                   'IMAGEFRAC_GR', 'IMAGEFRAC_GRZ', 'IN_IMAGING']
    for col in dropcolumns:
        if col in tiletab.dtype.names:
            tiletab.remove_column(col)
    fullname = args.savetiles
    tiletab.write(fullname, overwrite=True, format='ascii.ecsv')



def main(args):
    """Command-line driver for initializing the survey plan.
    """
    # Set up the logger
    if args.debug:
        log = desiutil.log.get_logger(desiutil.log.DEBUG)
        args.verbose = True
    elif args.verbose:
        log = desiutil.log.get_logger(desiutil.log.INFO)
    else:
        log = desiutil.log.get_logger(desiutil.log.WARNING)

    # Set the output path if requested.
    config = desisurvey.config.Configuration(file_name=args.config_file)
    if args.output_path is not None:
        config.set_output_path(args.output_path)
    if args.tiles_file is not None:
        config.tiles_file.set_value(args.tiles_file)

    # Tabulate emphemerides if necessary.
    ephem = desisurvey.ephem.get_ephem(
        use_cache=not (args.recalc or args.recalc_ephem))

    # Calculate design hour angles if necessary.
    fullnamelst = config.get_path(args.savelst)
    if args.savetiles is not None:
        fullnametiles = args.savetiles
    else:
        tiles = desisurvey.tiles.get_tiles()
        fullnametiles = os.path.basename(tiles.tiles_file)
    fullnametiles = config.get_path(fullnametiles)
    args.savetiles = fullnametiles
    if ((args.recalc or args.recalc_lst) or not
            (os.path.exists(fullnamelst) and os.path.exists(fullnametiles))):
        calculate_initial_plan(args)
    else:
        log.info('Initial plan has already been created.')
