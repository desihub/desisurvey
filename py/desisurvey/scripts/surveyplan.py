"""Script wrapper for running survey planning.

To run this script from the command line, use the ``surveyplan`` entry point
that is created when this package is installed, and should be in your shell
command search path.
"""
from __future__ import print_function, division, absolute_import

import argparse
import os.path
import datetime
import sys

import astropy.time
import astropy.table

import desiutil.log

import desisurvey.ephemerides
import desisurvey.schedule
import desisurvey.plan
import desisurvey.utils
import desisurvey.config
import desisurvey.rules


def parse(options=None):
    """Parse command-line options for running survey planning.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action='store_true',
        help='display log messages with severity >= info')
    parser.add_argument('--debug', action='store_true',
        help='display log messages with severity >= debug (implies verbose)')
    parser.add_argument(
        '--create', action='store_true', help='create an initial plan')
    parser.add_argument(
        '--rules', metavar='YAML', default='rules.yaml',
        help='name of YAML file with observing priority rules')
    '''
    parser.add_argument(
        '--duration', type=int, metavar='DAYS', default=None,
        help='duration of plan in days (or plan rest of the survey)')
    parser.add_argument(
        '--nopts', type=int, metavar='N', default=5000,
        help='number of hour-angle optimization iterations to perform')
    parser.add_argument(
        '--plots', action='store_true',
        help='save diagnostic plots of the plan optimzation for each program')
    '''
    parser.add_argument(
        '--output-path', default=None, metavar='PATH',
        help='output path where output files should be written')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def main(args):
    """Command-line driver for updating the survey plan.
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

    # Load ephemerides.
    ephem = desisurvey.ephemerides.Ephemerides()

    # Initialize scheduler.
    if not os.path.exists(config.get_path('scheduler.fits')):
        # Tabulate data used by the scheduler if necessary.
        desisurvey.schedule.initialize(ephem)
    scheduler = desisurvey.schedule.Scheduler()

    # Read priority rules.
    rules = desisurvey.rules.Rules(args.rules)
    return

    if args.create:
        # Create a new plan and empty progress record.
        plan = desisurvey.plan.create()
        progress = desisurvey.progress.Progress()
        # Start the survey from scratch.
        start = scheduler.start_date
    else:
        # Load an existing plan and progress record.
        if not os.path.exists(config.get_path('plan.fits')):
            log.error('No plan.fits found in output path.')
            return -1
        if not os.path.exists(config.get_path('progress.fits')):
            log.error('No progress.fits found in output path.')
            return -1
        plan = astropy.table.Table.read(config.get_path('plan.fits'))
        progress = desisurvey.progress.Progress('progress.fits')
        # Start the new plan from the last observing date.
        with open(config.get_path('last_date.txt'), 'r') as f:
            start = desisurvey.utils.get_date(f.read().rstrip())

    # Save a backup of the progress so far.
    progress.save('progress_{0}.fits'.format(start))

    # Reached end of the survey?
    if start >= config.last_day():
        log.info('Reached survey end date!')
        sys.exit(9)

    # Calculate the end date of the plan.
    if args.duration is not None:
        stop = start + datetime.timedelta(days=args.duration)
    else:
        stop = config.last_day()

    log.info('Planning observations for {0} to {1}.'
             .format(start, stop))

    # Save plots?
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        plots = config.get_path('plan_{0}'.format(start))
    else:
        plots = None

    # Update the plan.
    plan = desisurvey.plan.update(
        plan, progress, scheduler, start, stop,
        nopts=(args.nopts,), plot_basename=plots)

    # All done?
    if plan is None:
        log.info('All tiles observed!')
        # Return a shell exit code to allow scripts to detect this condition.
        sys.exit(9)

    # Save the plan and a backup.
    plan.write(config.get_path('plan.fits'), overwrite=True)
    plan.write(config.get_path('plan_{0}.fits'.format(start)), overwrite=True)
