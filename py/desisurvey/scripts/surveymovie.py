"""Script wrapper for creating a movie of survey progress.

To run this script from the command line, use the ``surveymovie`` entry point
that is created when this package is installed, and should be in your shell
command search path.

The optional matplotlib python package must be installed to use this script.

The external program ffmpeg must be installed to use this script.
At nersc, try ``module add ffmpeg``.
"""
from __future__ import print_function, division, absolute_import

import argparse
import os.path

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.animation
import matplotlib.colors
import matplotlib.animation

import astropy.time
import astropy.io.fits
import astropy.units as u

import desiutil.log

import desisurvey.ephem
import desisurvey.utils
import desisurvey.config
import desisurvey.tiles
import desisurvey.plots


def parse(options=None):
    """Parse command-line options for running survey planning.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action='store_true',
        help='display log messages with severity >= info')
    parser.add_argument('--debug', action='store_true',
        help='display log messages with severity >= debug (implies verbose)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
        help='interval for logging periodic info messages')
    parser.add_argument(
        '--exposures', default='exposures_surveysim.fits', metavar='FITS',
        help='name of FITS file with list of exposures taken')
    parser.add_argument(
        '--start', type=str, default=None, metavar='DATE',
        help='movie starts on the evening of this day, formatted as YYYY-MM-DD')
    parser.add_argument(
        '--stop', type=str, default=None, metavar='DATE',
        help='movie stops on the morning of this day, formatted as YYYY-MM-DD')
    parser.add_argument(
        '--expid', type=int, default=None, metavar='ID',
        help='index of single exposure to display')
    parser.add_argument(
        '--nightly', action='store_true',
        help='output one summary frame per night')
    # The scores option needs to be re-implemented after the refactor.
    ##parser.add_argument(
    ##    '--scores', action='store_true', help='display scheduler scores')
    parser.add_argument(
        '--save', type=str, default='surveymovie', metavar='NAME',
        help='base name (without extension) of output file to write')
    parser.add_argument(
        '--fps', type=float, default=10., metavar='FPS',
        help='frames per second to render')
    parser.add_argument(
        '--label', type=str, default='DESI', metavar='TEXT',
        help='label to display on each frame')
    parser.add_argument(
        '--output-path', default=None, metavar='PATH',
        help='path that desisurvey files are read from')
    parser.add_argument(
        '--tiles-file', default=None, metavar='TILES',
        help='name of tiles file to use instead of config.tiles_file')
    parser.add_argument(
        '--config-file', default='config.yaml', metavar='CONFIG',
        help='input configuration file')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    # The scores option needs to be re-implemented after the refactor.
    args.scores = False

    if args.nightly and args.scores:
        log.warn('Cannot display scores in nightly summary.')
        args.scores = False

    # Validate start/stop date args and covert to datetime objects.
    # Unspecified values are taken from our config.
    config = desisurvey.config.Configuration(args.config_file)
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


class Animator(object):
    """Manage animation of survey progress.
    """
    def __init__(self, exposures_path, start, stop, label, show_scores):
        self.log = desiutil.log.get_logger()
        self.config = desisurvey.config.Configuration()
        self.tiles = desisurvey.tiles.get_tiles()
        self.ephem = desisurvey.ephem.get_ephem()
        # Load exposures and associated tile data.
        self.exposures = astropy.table.Table.read(exposures_path, hdu='EXPOSURES')
        self.tiledata = astropy.table.Table.read(exposures_path, hdu='TILEDATA')
        self.label = label
        self.show_scores = show_scores
        self.ra = wrap(self.tiles.tileRA)
        self.dec = self.tiles.tileDEC
        self.passnum = self.tiles.passnum
        self.tileid = self.tiles.tileID
        # We require a standard set of passes.
        if not np.array_equal(self.tiles.passes, np.arange(8)):
            raise RuntimeError('Expected passes 0-7.')
        self.prognames = ['GRAY', 'DARK', 'DARK', 'DARK', 'DARK',
                          'BRIGHT', 'BRIGHT', 'BRIGHT']
        if self.tiles.nogray:
            self.prognames[0] = 'DARK'
        self.tiles_per_pass = self.tiles.pass_ntiles
        if self.tiles.nogray:
            self.program_names = ['DARK', 'BRIGHT']
        else:
            self.program_names = ['DARK', 'GRAY', 'BRIGHT']
        self.psels = [self.tiles.program_mask[x] for x in self.program_names]
        self.start_date = self.config.first_day()
        self.survey_weeks = int(np.ceil((self.config.last_day() - self.start_date).days / 7))

        # Add some computed columns to the exposures table.
        self.exposures['EXPID'] = np.arange(len(self.exposures))
        self.exposures['INDEX'] = self.tiles.index(self.exposures['TILEID'])
        self.exposures['PASS'] = self.tiles.passnum[self.exposures['INDEX']]
        self.exposures['STATUS'] = np.ones(len(self.exposures), np.int32)
        self.exposures['STATUS'][self.exposures['SNR2FRAC'] == 0] = 0
        self.exposures['STATUS'][
            self.exposures['SNR2FRAC'] >= self.config.min_snr2_fraction()] = 2

        # Convert tables to recarrays for much faster indexing.
        self.exposures = self.exposures.as_array()
        self.tiledata = self.tiledata.as_array()

        # Restrict the list of exposures to [start, stop].
        date_start = desisurvey.utils.get_date(start)
        date_stop = desisurvey.utils.get_date(stop)
        mjd_start = desisurvey.utils.local_noon_on_date(date_start).mjd
        mjd_stop = desisurvey.utils.local_noon_on_date(date_stop).mjd
        in_range = (self.exposures['MJD'] >= mjd_start) & (self.exposures['MJD'] < mjd_stop)
        self.exposures = self.exposures[in_range]
        self.num_exp = len(self.exposures)
        # Count nights with at least one exposure.
        day0 = desisurvey.utils.local_noon_on_date(date_start).mjd
        day_number = np.floor(self.exposures['MJD'] - day0)
        self.num_nights = len(np.unique(day_number))

        # Calculate each exposure's LST window.
        exp_midpt = astropy.time.Time(
            self.exposures['MJD'] + self.exposures['EXPTIME'] / 86400.,
            format='mjd', location=desisurvey.utils.get_location())
        lst_midpt = exp_midpt.sidereal_time('apparent').to(u.deg).value
        # convert from seconds to degrees.
        lst_len = self.exposures['EXPTIME'] / 240.
        self.lst = np.empty((self.num_exp, 2))
        self.lst[:, 0] = wrap(lst_midpt - 0.5 * lst_len)
        self.lst[:, 1] = wrap(lst_midpt + 0.5 * lst_len)

    def init_figure(self, nightly, width=1920, height=1080, dpi=32):
        """Initialize matplot artists for drawing each frame.
        """
        self.dpi = float(dpi)
        # Initialize figure and axes.
        self.figure = plt.figure(
            frameon=False,figsize=(width / self.dpi, height / self.dpi),
            dpi=self.dpi)
        grid = matplotlib.gridspec.GridSpec(3, 3)
        grid.update(left=0, right=1, bottom=0, top=0.97, hspace=0, wspace=0)
        axes = []
        self.labels = []
        self.iplots = []
        self.scatters = []
        self.lstlines = []
        self.avoids = []
        self.avoid_names = list(self.config.avoid_bodies.keys)
        navoids = len(self.avoid_names)
        self.moon_index = self.avoid_names.index('moon')
        self.xy_avoid = np.zeros((navoids, 2))
        self.f_obj = [None] * navoids
        bgcolor = matplotlib.colors.to_rgba('lightblue')
        avoidcolor = matplotlib.colors.to_rgba('red')
        self.defaultcolor = np.array([[1., 1., 1., 1.]])
        self.completecolor = np.array([0., 0.5, 0., 1.])
        self.availcolor = np.array([1., 1., 1., 1.])
        self.unavailcolor = np.array([0.65, 0.65, 0.65, 1.])
        self.nowcolor = np.array([0., 0.7, 0., 1.])
        pcolors = desisurvey.plots.program_color
        passnum = 0
        for row in range(3):
            for col in range(3):
                # Create the axes for this pass.
                ax = plt.subplot(grid[row, col], facecolor=bgcolor)
                ax.set_xticks([])
                ax.set_yticks([])
                axes.append(ax)
                # Top-right corner is reserved for integrated progress plots.
                if row == 0 and col == 2:
                    ax.set_xlim(0, self.survey_weeks)
                    ax.set_ylim(0, 1)
                    ax.plot([0, self.survey_weeks], [0., 1.], 'w-')
                    for pname in self.program_names:
                        pc = pcolors[pname]
                        xprog = 0.5 + np.arange(self.survey_weeks)
                        # Initialize values to INF so they are not plotted
                        # until some value is assigned later.  Any week with
                        # no observations will then result in a gap.
                        yprog = np.full(self.survey_weeks, np.inf)
                        self.iplots.append(ax.plot(
                            xprog, yprog, lw=2, ls='-',
                            color=pcolors[pname])[0])
                    continue
                ax.set_xlim(-55, 293)
                ax.set_ylim(-20, 77)
                # Draw label for this plot.
                pname = self.prognames[passnum]
                pc = pcolors[pname]
                self.labels.append(ax.annotate(
                    '{0}-{1} 100.0%'.format(pname, passnum),
                    xy=(0.05, 0.95), xytext=(0.05, 0.95),
                    xycoords='axes fraction', fontsize=48, family='monospace',
                    color=pc, horizontalalignment='left',
                    verticalalignment='top'))
                # Draw the tile outlines for this pass.
                sel = (self.passnum == passnum)
                ntiles = np.count_nonzero(sel)
                fc = np.empty((ntiles, 4))
                fc[:] = self.defaultcolor
                s = np.full(ntiles, 90.)
                self.scatters.append(ax.scatter(
                    self.ra[sel], self.dec[sel], s=s, facecolors=fc,
                    edgecolors='none', lw=1))
                # Initialize positions of moon and planets.
                fc = np.zeros((navoids, 4))
                ec = np.zeros((navoids, 4))
                ec[:] = avoidcolor
                s = np.full(navoids, 500.)
                s[self.moon_index] = 2500.
                self.avoids.append(ax.scatter(
                    self.xy_avoid[:, 0], self.xy_avoid[:, 1], facecolors=fc,
                    edgecolors=ec, s=s, lw=2))
                if not nightly:
                    # Draw LST lines for the current exposure.
                    line1 = ax.axvline(0., lw=2, ls=':', color=self.nowcolor)
                    line2 = ax.axvline(0., lw=2, ls=':', color=self.nowcolor)
                    self.lstlines.append((line1, line2))
                passnum += 1

        if self.show_scores:
            # Initialize scheduler score colormap.
            self.scorecmap = matplotlib.cm.get_cmap('hot_r')

        # Add axis above plot to display programs during the night.
        paxes = plt.axes([0, 0.97, 0.66667, 0.03], facecolor='y')
        paxes.set_xticks([])
        paxes.set_yticks([])
        edges = 0.5 + np.linspace(-6., +7., 13 * 60) / 24.
        self.dmjd = 0.5 * (edges[1:] + edges[:-1])
        paxes.set_xlim(edges[0], edges[-1])
        paxes.set_ylim(0., 1.)
        self.pdata = np.zeros(len(self.dmjd), int)
        # Prepare a custom colormap.
        colors = [bgcolor, pcolors['DARK'], pcolors['GRAY'], pcolors['BRIGHT']]
        pcmap = matplotlib.colors.ListedColormap(colors, 'programs')
        self.programs = paxes.imshow(
            self.pdata.reshape(1, -1), interpolation='none', aspect='auto',
            extent=(edges[0], edges[-1], 0., 1.),
            vmin=-0.5, vmax=3.5, cmap=pcmap)
        if not nightly:
            self.pline1 = paxes.axvline(0., lw=4, ls='-', color=self.nowcolor)
            self.pline2 = paxes.axvline(0., lw=4, ls='-', color=self.nowcolor)

        # Add text label in the top-right corner.
        self.text = plt.annotate(
            'YYYY-MM-DD #000000', xy=(0.995, 0.997), xytext=(0.995, 0.997),
            xycoords='figure fraction', fontsize=64, family='monospace',
            color='k', horizontalalignment='right', verticalalignment='top')

        # List all animated artists.
        self.artists = (
            self.scatters + self.avoids + self.labels + self.iplots +
            [self.programs, self.text])
        if not nightly:
            self.artists += [self.pline1, self.pline2]
            for l1, l2 in self.lstlines:
                self.artists += [l1, l2]

        # Initialize internal tracking vars.
        self.last_date = None
        self.scores = None
        self.iexp0 = None
        self.status = None

    def init_date(self, date, ephem):
        """Initialize before drawing frames for a new night.

        Parameters
        ----------
        date : datetime.date
            Date on which this night's observing starts.
        night : astropy.table.Column
            Ephemerides data for this night.
        """
        # Update the observing program for this night.
        dark, gray, bright = self.ephem.tabulate_program(ephem['noon'] + self.dmjd)
        self.pdata[:] = 0
        self.pdata[dark] = 1
        self.pdata[gray] = 2
        self.pdata[bright] = 3
        self.programs.set_data(self.pdata.reshape(1, -1))
        if self.show_scores:
            # Load new scheduler scores for this night.
            scores_name = self.config.get_path('scores_{0}.fits'.format(date))
            if os.path.exists(scores_name):
                hdus = astropy.io.fits.open(scores_name, memmap=False)
                self.scores = hdus[0].data
                hdus.close()
                # Save index of first exposure on this date.
                noon = desisurvey.utils.local_noon_on_date(date)
                self.iexp0 = np.argmax(self.exposures['MJD'] > noon.mjd)
            else:
                self.warn('Missing scores file: {0}.'.format(scores_name))
        # Get interpolator for moon, planet positions during this night.
        for i, name in enumerate(self.avoid_names):
            self.f_obj[i] = desisurvey.ephem.get_object_interpolator(
                ephem, name)
        if self.last_date is not None:
            week_num = int(np.floor((date - self.start_date).days / 7.))
            # Update progress graphs for each program.
            for psel, iplot in zip(self.psels, self.iplots):
                nprog = np.count_nonzero(psel)
                ndone = np.count_nonzero(self.status[psel] == 2)
                yprog = iplot.get_ydata()
                yprog[week_num] = 1.0 * ndone / nprog
                iplot.set_ydata(yprog)
        # Lookup which tiles are available and planned for tonight.
        day_number = desisurvey.utils.day_number(date)
        avail = self.tiledata['AVAIL']
        self.available = (avail >= -1) & (avail <= day_number)
        planned = self.tiledata['PLANNED']
        self.planned = (planned >= -1) & (planned <= day_number)
        self.last_date = date

    def draw_exposure(self, iexp, nightly):
        """Draw the frame for a single exposure.

        Calls :meth:`init_date` if this is the first exposure of the night
        that we have seen.

        Parameters
        ----------
        iexp : int
            Index of the exposure to draw.

        Returns
        -------
        bool
            True if a new frame was drawn.
        """
        info = self.exposures[iexp]
        mjd = info['MJD']
        date = desisurvey.utils.get_date(mjd)
        night = self.ephem.get_night(date)
        # Initialize status if necessary.
        if self.status is None:
            # Find the most recent SNR2FRAC for each tile before this exposure.
            snr2frac = np.zeros(self.tiles.ntiles)
            for j in range(iexp):
                snr2frac[self.exposures['INDEX'][j]] = self.exposures['SNR2FRAC'][j]
            # Determine which tiles are unobserved (0), partial (1) or done (2)
            # before this exposure.
            self.status = np.ones(self.tiles.ntiles)
            self.status[snr2frac == 0] = 0
            self.status[snr2frac >= self.config.min_snr2_fraction()] = 2
        # Update the status for the current exposure.
        self.status[info['INDEX']] = info['STATUS']
        if date != self.last_date:
            # Initialize for this night.
            self.init_date(date, night)
        elif nightly:
            return False
        # Update the top-right label.
        label = '{} {} #{:06d}'.format(self.label, date, info['EXPID'])
        if not nightly:
            label += ' ({:.1f}",{:.2f})'.format(
                info['SEEING'], info['TRANSP'])
        self.text.set_text(label)
        if not nightly:
            # Update current time in program.
            dt1 = mjd - night['noon']
            dt2 = dt1 + info['EXPTIME'] / 86400.
            self.pline1.set_xdata([dt1, dt1])
            self.pline2.set_xdata([dt2, dt2])
        if self.show_scores:
            # Update scores display for this exposure.
            score = self.scores[iexp - self.iexp0]
            max_score = np.max(score)
        for passnum, scatter in enumerate(self.scatters):
            sel = (self.passnum == passnum)
            done = self.status[sel] == 2
            avail = self.available[sel]
            inplan = self.planned[sel]
            if self.show_scores:
                fc = self.scorecmap(score[sel] / max_score)
            else:
                fc = scatter.get_facecolors()
                fc[avail] = self.availcolor
            sizes = scatter.get_sizes()
            sizes[~inplan] = 20.
            sizes[~done & inplan] = 90.
            sizes[done] = 30.
            fc[done] = self.completecolor
            fc[~avail] = self.unavailcolor
            if not nightly and (info['PASS'] == passnum):
                # Highlight the tile being observed now.
                jdx = np.where(self.tileid[sel] == info['TILEID'])[0][0]
                fc[jdx] = self.nowcolor
                scatter.get_sizes()[jdx] = 600.
            scatter.set_facecolors(fc)
            # Update percent complete label.
            pct = (100. * np.count_nonzero(self.status[sel] == 2) /
                   self.tiles_per_pass[passnum])
            self.labels[passnum].set_text('{0}-{1} {2:5.1f}%'.format(
                self.prognames[passnum], passnum, pct))
        if not nightly:
            # Update LST lines.
            x1, x2 = self.lst[iexp]
            for passnum, (line1, line2) in enumerate(self.lstlines):
                ls = '-' if info['PASS'] == passnum else '--'
                line1.set_linestyle(ls)
                line2.set_linestyle(ls)
                line1.set_xdata([x1, x1])
                line2.set_xdata([x2, x2])
        # Fill the moon with a shade of gray corresponding to its illuminated
        # fraction during this exposure.
        moon_frac = self.ephem.get_moon_illuminated_fraction(mjd)
        # Update moon and planet locations.
        for i, f in enumerate(self.f_obj):
            # Calculate this object's (dec,ra) path during the night.
            obj_dec, obj_ra = f(mjd)
            self.xy_avoid[i] = wrap(obj_ra), obj_dec
        for scatter in self.avoids:
            scatter.set_offsets(self.xy_avoid)
            scatter.get_facecolors()[self.moon_index] = [
                moon_frac, moon_frac, moon_frac, 1.]
        return True


def main(args):
    """Command-line driver to visualize survey scheduling and progress.
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
    if args.tiles_file is not None:
        config.tiles_file.set_value(args.tiles_file)

    # Look for the exposures file in the output path by default.
    args.exposures = config.get_path(args.exposures)

    # Initialize.
    animator = Animator(
        args.exposures, args.start, args.stop, args.label, args.scores)
    log.info('Found {0} exposures from {1} to {2} ({3} nights).'
             .format(animator.num_exp, args.start, args.stop,
                     animator.num_nights))
    animator.init_figure(args.nightly)

    if args.expid is not None:
        expid = animator.exposures['EXPID']
        assert np.all(expid == expid[0] + np.arange(len(expid)))
        if (args.expid < expid[0]) or (args.expid > expid[-1]):
            raise RuntimeError('Requested exposure ID {0} not available.'
                               .format(args.expid))
        animator.draw_exposure(args.expid - expid[0], args.nightly)
        save_name = args.save + '.png'
        plt.savefig(save_name)
        log.info('Saved {0}.'.format(save_name))
    else:
        nframes = animator.num_nights if args.nightly else animator.num_exp
        iexp = [0]
        def init():
            return animator.artists
        def update(iframe):
            if (iframe + 1) % args.log_interval == 0:
                log.info('Drawing frame {0}/{1}.'
                         .format(iframe + 1, nframes))
            if args.nightly:
                while not animator.draw_exposure(iexp[0], nightly=True):
                    iexp[0] += 1
            else:
                animator.draw_exposure(iexp=iframe, nightly=False)
            return animator.artists
        log.info('Movie will be {:.1f} mins long at {:.1f} frames/sec.'
                 .format(nframes / (60 * args.fps), args.fps))
        animation = matplotlib.animation.FuncAnimation(
            animator.figure, update, init_func=init, blit=True, frames=nframes)
        writer = matplotlib.animation.writers['ffmpeg'](
            bitrate=2400, fps=args.fps, metadata=dict(artist='surveymovie'))
        save_name = args.save + '.mp4'
        animation.save(save_name, writer=writer, dpi=animator.dpi)
        log.info('Saved {0}.'.format(save_name))
