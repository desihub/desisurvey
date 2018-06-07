"""Script wrapper for initializing survey planning and scheduling.

This is normally run once, at the start of the survey, and saves its results
to a FITS file surveyinit.fits.  With the default parameters, the running time
is about 25 minutes.

This script will create the ephemerides and scheduler files, if necessary,
which takes about 15 additional minutes. These only need to be created once.

To run this script from the command line, use the ``surveyinit`` entry point
that is created when this package is installed, and should be in your shell
command search path.
"""
from __future__ import print_function, division, absolute_import

import os
import argparse

import numpy as np

import astropy.table

import desiutil.log

import desimodel.io

import desisurvey.utils
import desisurvey.ephemerides
import desisurvey.schedule
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
        '--recalc', action='store_true',
        help='recalculate even when previous calculations are available')
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
        '--dark-stretch', type=float, default=1.1, metavar='S',
        help='stretch DARK exposure times by this factor')
    parser.add_argument(
        '--gray-stretch', type=float, default=1.2, metavar='S',
        help='stretch GRAY exposure times by this factor')
    parser.add_argument(
        '--bright-stretch', type=float, default=1.5, metavar='S',
        help='stretch BRIGHT exposure times by this factor')
    parser.add_argument(
        '--save', default='surveyinit.fits', metavar='NAME',
        help='name of FITS output file where results are saved')
    parser.add_argument(
        '--output-path', default=None, metavar='PATH',
        help='output path where output files should be written')
    parser.add_argument(
        '--config-file', default='config.yaml', metavar='CONFIG',
        help='input configuration file')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def calculate_initial_plan(args, scheduler, fullname):
    """Calculate initial hour-angle assignments for all tiles.
    """
    log = desiutil.log.get_logger()
    config = desisurvey.config.Configuration()

    # Initialize the output results table.
    tiles = astropy.table.Table(desimodel.io.load_tiles(
        onlydesi=True, extra=False, tilesfile=config.tiles_file()))
    out = tiles[['TILEID', 'RA', 'DEC', 'PASS']]
    out['HA'] = np.zeros(len(out))
    out['OBSTIME'] = np.zeros(len(out))

    # Optimize each program separately.
    stretches = dict(
        DARK=args.dark_stretch,
        GRAY=args.gray_stretch,
        BRIGHT=args.bright_stretch)
    for program in 'DARK', 'GRAY', 'BRIGHT':
        sel = tiles['PROGRAM'] == program
        if np.count_nonzero(sel) > 0:
            opt = desisurvey.optimize.Optimizer(
                scheduler, program, init=args.init, center=None, nbins=args.nbins,
                subset=tiles['TILEID'][sel], stretch=stretches[program])
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
                     .format(program, plan_sum, avail_sum, 1e2 * margin))

            # Calculate exposure times in seconds.
            texp, _ = opt.get_exptime(opt.ha)
            texp *= 24. * 3600. / 360.
            # Save results for this program.
            out['HA'][sel] = opt.ha
            out['OBSTIME'][sel] = texp

    log.info('Saving results to {0}'.format(fullname))
    out.write(fullname, overwrite=True)


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

    # Tabulate emphemerides if necessary.
    ephem = desisurvey.ephemerides.Ephemerides(use_cache=not args.recalc)

    if args.recalc or not os.path.exists(config.get_path('scheduler.fits')):
        # Tabulate data used the the scheduler.
        desisurvey.schedule.initialize(ephem)

    # Load scheduler with precomputed tables needed by the optimizer.
    scheduler = desisurvey.schedule.Scheduler()

    # Can we use existing HA assignments?
    fullname = config.get_path(args.save)
    if args.recalc or not os.path.exists(fullname):
        calculate_initial_plan(args, scheduler, fullname)
