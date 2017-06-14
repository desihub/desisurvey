"""Script wrapper for running survey planning.

To run this script from the command line, use the ``surveyplan`` entry point
that is created when this package is installed, and should be in your shell
command search path.
"""
from __future__ import print_function, division, absolute_import

import argparse

import desiutil.log

import desisurvey.ephemerides
import desisurvey.schedule
import desisurvey.plan
import desisurvey.utils
import desisurvey.config


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
        '--duration', type=int, metavar='DAYS', default=None,
        help='duration of plan in days (or plan rest of the survey)')
    parser.add_argument(
        '--output-path', default=None, metavar='PATH',
        help='output path where output files should be written')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def main(args):
    """Command-line driver for running survey simulations.
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
