"""Script wrapper for creating a movie of survey progress.

To run this script from the command line, use the ``surveymovie`` entry point
that is created when this package is installed, and should be in your shell
command search path.
"""
from __future__ import print_function, division, absolute_import

import argparse

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.animation

import astropy.time
import astropy.units as u

import desiutil.log

import desisurvey.ephemerides
import desisurvey.utils
import desisurvey.config
import desisurvey.progress


def parse(options=None):
    """Parse command-line options for running survey planning.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action='store_true',
        help='display log messages with severity >= info')
    parser.add_argument('--debug', action='store_true',
        help='display log messages with severity >= debug (implies verbose)')
    parser.add_argument('--progress', default='progress.fits', metavar='FITS',
        help='name of FITS file with progress record')
    parser.add_argument(
        '--start', type=str, default=None, metavar='DATE',
        help='movie starts on the evening of this day, formatted as YYYY-MM-DD')
    parser.add_argument(
        '--stop', type=str, default=None, metavar='DATE',
        help='movie stops on the morning of this day, formatted as YYYY-MM-DD')
    parser.add_argument(
        '--output-path', default=None, metavar='PATH',
        help='output path where output files should be written')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    # Validate start/stop date args and covert to datetime objects.
    # Unspecified values are taken from our config.
    config = desisurvey.config.Configuration()
    if args.start is None:
        args.start = config.first_day()
    else:
        try:
            args.start = desisurvey.utils.get_date(args.start)
        except ValueError as e:
            raise ValueError('Invalid start: {0}'.format(e))
    if args.stop is None:
        args.stop = config.last_day()
    else:
        try:
            args.stop = desisurvey.utils.get_date(args.stop)
        except ValueError as e:
            raise ValueError('Invalid stop: {0}'.format(e))
    if args.start >= args.stop:
        raise ValueError('Expected start < stop.')

    return args


def wrap(angle, offset=-60):
    """Wrap values in the range [0, 360] to [offset, offset+360].
    """
    return np.fmod(angle - offset + 360, 360) + offset


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

    # Load progress.
    progress = desisurvey.progress.Progress(args.progress)
    tiles = progress._table
    ra = wrap(tiles['ra'])
    dec = tiles['dec']

    # Get a list of exposures in [start, stop].
    exposures = progress.get_exposures(
        args.start, args.stop, tile_fields='tileid,index,ra,dec,pass',
        exp_fields='mjd,exptime,snr2cum')
    tile_index = exposures['index']
    snr2 = exposures['snr2cum']
    num_exp = len(tile_index)
    log.info('Found {0} exposures from {1} to {2}.'
             .format(num_exp, args.start, args.stop))

    # Calculate each exposure's LST window.
    exp_midpt = astropy.time.Time(
        exposures['mjd'] + exposures['exptime'] / 86400., format='mjd',
        location=desisurvey.utils.get_location())
    lst_midpt = exp_midpt.sidereal_time('apparent').to(u.deg).value
    lst_len = exposures['exptime'] / 240. # convert from seconds to degrees.
    lst = np.empty((num_exp, 2))
    lst[:, 0] = wrap(lst_midpt - 0.5 * lst_len)
    lst[:, 1] = wrap(lst_midpt + 0.5 * lst_len)

    # power of 2 so that inches are exact.
    dpi = 32.
    # Use 1080p for compatibility with youtube.
    width, height = 1920, 1080

    # Initialize figure and axes.
    figure = plt.figure(
        frameon=False,figsize=(width / dpi, height / dpi), dpi=dpi)
    grid = matplotlib.gridspec.GridSpec(3, 3)
    grid.update(left=0, right=1, bottom=0, top=0.97, hspace=0, wspace=0)
    axes = []
    scatters = []
    passnum = 0
    for row in range(3):
        for col in range(3):
            # Create the axes for this pass.
            ax = plt.subplot(grid[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            axes.append(ax)
            # Top-right corner is reserved for a plot.
            if row == 0 and col == 2:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                plt.plot([0, 1], [0, 1], 'k:')
                continue
            ax.set_xlim(-55, 293)
            ax.set_ylim(-20, 77)
            # Draw the tile outlines for this pass.
            sel = (tiles['pass'] == passnum)
            ntiles = np.count_nonzero(sel)
            fc = np.zeros((ntiles, 4))
            ec = np.zeros((ntiles, 4))
            ec[:, 3] = 0.25
            s = np.full(ntiles, 85.)
            scatters.append(ax.scatter(
                ra[sel], dec[sel], s=s, facecolors=fc, edgecolors=ec, lw=1))
            passnum += 1

    # Add axis above plot to display programs during the night.
    programs = plt.axes([0, 0.97, 0.66667, 0.03], facecolor='y')
    programs.set_xticks([])
    programs.set_yticks([])

    # Add text label in the top-right corner.
    text = plt.annotate(
        'Label goes here...', xy=(0.995, 0.997), xytext=(0.995, 0.997),
        textcoords='figure fraction', fontsize=64,
        horizontalalignment='right', verticalalignment='top')

    plt.savefig('movie.png')
