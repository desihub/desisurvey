"""Script wrapper for initializing survey planning and scheduling.

To run this script from the command line, use the ``surveyinit`` entry point
that is created when this package is installed, and should be in your shell
command search path.
"""
from __future__ import print_function, division, absolute_import

import argparse
import cProfile

import numpy as np

import astropy.table

import desiutil.log

import desimodel.io

import desisurvey.utils
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
        '--nbins', type=int, default=90, metavar='N',
        help='number of LST bins to use')
    parser.add_argument(
        '--initial-frac', default=0.5, metavar='F',
        help='fraction of an LST bin for initial HA adjustments')
    parser.add_argument(
        '--anneal-rate', default=0.9, metavar='R',
        help='decrease fraction by this factor after each annealing cycle')
    parser.add_argument(
        '--epsilon', default=0.01, metavar='EPS',
        help='stop cycles when fractional MSE improvement < EPS')
    parser.add_argument(
        '--max-cycles', type=int, default=100,
        help='maximum number of annealing cycles for each program')
    parser.add_argument(
        '--save', default='surveyinit.fits', metavar='NAME',
        help='name of FITS output file where results are saved')
    parser.add_argument(
        '--output-path', default=None, metavar='PATH',
        help='output path where output files should be written')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


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

    # Freeze IERS table for consistent results.
    desisurvey.utils.freeze_iers()

    # Set the output path if requested.
    config = desisurvey.config.Configuration()
    if args.output_path is not None:
        config.set_output_path(args.output_path)

    # Load scheduler with precomputed tables needed by the optimizer.
    scheduler = desisurvey.schedule.Scheduler()

    # Initialize the output results table.
    tiles = astropy.table.Table(desimodel.io.load_tiles(
        onlydesi=True, extra=False))
    out = tiles[['TILEID', 'RA', 'DEC', 'PASS']]
    out['HA'] = np.zeros(len(out))
    out['OBSTIME'] = np.zeros(len(out))

    stretches = dict(DARK=1.0, GRAY=1.0, BRIGHT=1.25)

    # Optimize each program separately.
    for program in 'DARK', 'GRAY', 'BRIGHT':
        sel = tiles['PROGRAM'] == program
        opt = desisurvey.optimize.Optimizer(
            scheduler, program, init='zero', nbins=args.nbins,
            subset=tiles['TILEID'][sel], stretch=stretches[program])
        # Loop over annealing cycles.
        num_cycles = 0
        frac = args.initial_frac
        while num_cycles < args.max_cycles:
            start_MSE = opt.MSE_history[-1]
            for i in range(opt.ntiles):
                opt.improve(frac)
            stop_MSE = opt.MSE_history[-1]
            delta_MSE = (stop_MSE - start_MSE) / start_MSE
            loss = opt.loss_history[-1]
            log.info(
                '[{0:03d}] frac={1:.4f} MSE={2:7.1f} ({3:4.1f}%) LOSS={4:4.1f}%'
                     .format(num_cycles + 1, frac, stop_MSE, 1e2 * delta_MSE,
                             1e2 * loss))
            if delta_MSE > -args.epsilon:
                break
            frac *= args.anneal_rate
            num_cycles += 1

        # Calculate exposure times in seconds.
        texp, _ = opt.get_exptime(opt.ha)
        texp *= 24. * 3600. / 360.
        # Save results for this program.
        out['HA'][sel] = opt.ha
        out['OBSTIME'][sel] = texp

    fullname = config.get_path(args.save)
    log.info('Saving results to {0}'.format(fullname))
    out.write(fullname, overwrite=True)
