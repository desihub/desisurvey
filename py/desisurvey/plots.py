"""Utility functions for plotting DESI survey progress and planning.
"""
from __future__ import print_function, division

import datetime

import numpy as np

import astropy.table
import astropy.units as u

import desiutil.plots

import desisurvey.ephemerides
import desisurvey.config
import desisurvey.utils


# Color associated with each program in the functions below.
program_color = {'DARK': 'black', 'GRAY': 'gray', 'BRIGHT': 'orange'}


def plot_sky_passes(ra, dec, passnum, z, clip_lo=None, clip_hi=None,
                    label='label', save=None):
    """Plot sky maps for each pass of a per-tile scalar quantity.

    The matplotlib package must be installed to use this function.

    Parameters
    ----------
    ra : array
        Array of RA values to use in degrees.
    dec : array
        Array of DEC values to use in degrees.
    pass : array
        Array of integer pass values to use.
    z : array
        Array of per-tile values to plot.
    clip_lo : float or string or None
        See :meth:`desiutil.plot.prepare_data`
    clip_hi : float or string or None
        See :meth:`desiutil.plot.prepare_data`
    label : string
        Brief description of per-tile value ``z`` to use for axis labels.
    save : string or None
        Name of file where plot should be saved.  Format is inferred from
        the extension.

    Returns
    -------
    tuple
        Tuple (figure, axes) returned by ``plt.subplots()``.
    """
    import matplotlib.pyplot as plt

    z = desiutil.plots.prepare_data(
        z, clip_lo=clip_lo, clip_hi=clip_hi, save_limits=True)
    vmin, vmax = z.vmin, z.vmax
    hopts = dict(bins=50, range=(vmin, vmax), histtype='step')

    fig, ax = plt.subplots(3, 3, figsize=(15, 10))

    # Mapping of subplots to passes.
    pass_map = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    # Mapping of passes to programs.
    pass_program = ['DARK'] * 4 + ['GRAY'] + ['BRIGHT'] * 3
    max_count = 0.
    for p in range(8):
        color = program_color[pass_program[p]]
        basemap = desiutil.plots.init_sky(ax=ax[pass_map[p]],
                                          ra_labels=[-120, 0, 120],
                                          dec_labels=None,
                                          galactic_plane_color=color)
        # Select the tiles in this pass.
        sel = np.where(passnum == p)[0]
        z_sel = desiutil.plots.prepare_data(
            z[sel], clip_lo=vmin, clip_hi=vmax, save_limits=True)
        # Plot the sky map for this pass.
        desiutil.plots.plot_sky_circles(
            ra_center=ra[sel], dec_center=dec[sel], data=z_sel,
            colorbar=True, basemap=basemap, edgecolor='none', label=label)
        # Plot the histogram of values for this pass.
        counts, _, _ = ax[0, 2].hist(z[sel], color=color, **hopts)
        max_count = max(counts.max(), max_count)

    # Decorate the histogram subplot.
    ax[0, 2].set_xlim(vmin, vmax)
    ax[0, 2].set_ylim(0, 1.05 * max_count)
    ax[0, 2].set_xlabel(label)
    ax[0, 2].get_yaxis().set_ticks([])

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    # Make extra room for the histogram x-axis label.
    hist_rect = ax[0, 2].get_position(original=True)
    hist_rect.y0 = 0.70
    ax[0, 2].set_position(hist_rect)

    if save:
        plt.savefig(save)

    return fig, ax


def plot_observed(progress, include='observed', start_date=None, stop_date=None,
                  what='exptime', print_summary=False, save=None):
    """Plot a summary of observed tiles.

    Reports progress tracked by :class:`desisurvey.progress.Progress.` using
    :func:`plot_sky_passes` to display a summary of observed tiles in each pass.

    Parameters
    ----------
    progress : desisurvey.progress.Progress
        Progress tracker to use.
    include : 'all', 'observed', or 'completed'
        Specify which tiles to include in the summary. The 'observed'
        selection will include tiles that have been observed at least
        once but have not yet reached their SNR**2 goal.
    start_date : date or None
        Plot observations starting on the night of this date, or starting
        with the first observation if None. Must be convertible to a
        date using :func:`desisurvey.utils.get_date`.
    stop_date : date or None
        Plot observations ending on the morning of this date, or ending with
        the last observation if None. Must be convertible to a date using
        :func:`desisurvey.utils.get_date`.
    what : string
        What quantity to plot for each planned tile. Must be a
        column name in the summary table returned by
        :meth:`desisurvey.progress.Progress.get_summary`.
    print_summary : bool
        Print a summary of observed tiles.
    save : string or None
        Name of file where plot should be saved.

    Returns
    -------
    tuple
        Tuple (figure, axes) returned by ``plt.subplots()``.
    """
    if start_date:
        start_date = desisurvey.utils.get_date(start_date)
        mjd_min = desisurvey.local_noon_on_date(start_date).mjd
    else:
        start_date = desisurvey.utils.get_date(progress.first_mjd)
        mjd_min = None
    if stop_date:
        stop_date = desisurvey.utils.get_date(stop_date)
        mjd_max = desisurvey.local_noon_on_date(stop_date).mjd
    else:
        stop_date = desisurvey.utils.get_date(progress.last_mjd)
        mjd_max = None
    date_label = '{0} to {1}'.format(start_date, stop_date)

    if mjd_min or mjd_max:
        # Copy progress within these MJD limits.
        progress = progress.copy_range(mjd_min, mjd_max)

    # Create the summary table to use.
    t = progress.get_summary(include)
    if what not in t.colnames:
        raise ValueError('Valid names are: {0}'.format(','.join(t.colnames)))

    if print_summary:
        print('Observing summary for {0}:'.format(date_label))
        for pass_num in range(8):
            stats = progress.completed(only_passes=pass_num, as_tuple=True)
            print('Completed {0:6.1f} / {1:4d} ({2:5.1f}%) tiles for pass {p}.'
                  .format(*stats, p=pass_num))

    label = '{0} ({1})'.format(what, date_label)
    return desisurvey.plots.plot_sky_passes(
        t['ra'], t['dec'], t['pass'], t[what], label=label, save=save)


def plot_program(ephem, start_date=None, stop_date=None, style='localtime',
                 include_monsoon=False, include_full_moon=False,
                 night_start=-6.5, night_stop=7.5, num_points=500,
                 bg_color='lightblue', save=None):
    """Plot an overview of the DARK/GRAY/BRIGHT program.

    The matplotlib package must be installed to use this function.

    Parameters
    ----------
    ephem : :class:`desisurvey.ephemerides.Ephemerides`
        Tabulated ephemerides data to use for determining the program.
    start_date : date or None
        First night to include in the plot or use the start of the
        calculated ephemerides.  Must be convertible to a
        date using :func:`desisurvey.utils.get_date`.
    stop_date : date or None
        First night to include in the plot or use the start of the
        calculated ephemerides.  Must be convertible to a
        date using :func:`desisurvey.utils.get_date`.
    style : string
        Plot style to use for the vertical axis: "localtime" shows time
        relative to local midnight, "histogram" shows elapsed time for
        each program during each night, and "cumulative" shows the
        cummulative time for each program since ``start_date``.
    night_start : float
        Start of night in hours relative to local midnight used to set
        y-axis minimum for 'localtime' style and tabulate nightly program.
    night_stop : float
        End of night in hours relative to local midnight used to set
        y-axis maximum for 'localtime' style and tabulate nightly program.
    num_points : int
        Number of subdivisions of the vertical axis to use for tabulating
        the program during each night. The resulting resolution will be
        ``(night_stop - night_start) / num_points`` hours.
    bg_color : matplotlib color
        Axis background color to use.  Must be a valid matplotlib color.
    save : string or None
        Name of file where plot should be saved.  Format is inferred from
        the extension.

    Returns
    -------
    tuple
        Tuple (figure, axes) returned by ``plt.subplots()``.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors
    import matplotlib.dates
    import matplotlib.ticker
    import pytz

    styles = ('localtime', 'histogram', 'cumulative')
    if style not in styles:
        raise ValueError('Valid styles are {0}.'.format(', '.join(styles)))

    if night_start >= night_stop:
        raise ValueError('Expected night_start < night_stop.')

    # Determine plot date range.
    start_date = desisurvey.utils.get_date(start_date or ephem.start)
    stop_date = desisurvey.utils.get_date(stop_date or ephem.stop)
    if start_date >= stop_date:
        raise ValueError('Expected start_date < stop_date.')
    mjd = ephem._table['noon']
    sel = ((mjd >= desisurvey.utils.local_noon_on_date(start_date).mjd) &
           (mjd < desisurvey.utils.local_noon_on_date(stop_date).mjd))
    t = ephem._table[sel]
    num_nights = len(t)

    # Matplotlib date axes uses local time and puts ticks between days
    # at local midnight. We explicitly specify UTC for x-axis labels so
    # that the plot does not depend on the caller's local timezone.
    tz = pytz.utc
    midnight = datetime.time(hour=0)
    xaxis_start = tz.localize(datetime.datetime.combine(start_date, midnight))
    xaxis_stop = tz.localize(datetime.datetime.combine(stop_date, midnight))
    xaxis_lo = matplotlib.dates.date2num(xaxis_start)
    xaxis_hi = matplotlib.dates.date2num(xaxis_stop)

    # Build a grid of elapsed time relative to local midnight during each night.
    midnight = t['noon'] + 0.5
    t_edges = np.linspace(night_start, night_stop, num_points + 1) / 24.
    t_centers = 0.5 * (t_edges[1:] + t_edges[:-1])

    # Calculate the number of hours for each program during each night.
    dt = (night_stop - night_start) / num_points
    hours = np.zeros((3, num_nights))
    observing_night = np.zeros(num_nights, bool)
    for i in np.arange(num_nights):
        if not include_monsoon and desisurvey.utils.is_monsoon(midnight[i]):
            continue
        if not include_full_moon and ephem.is_full_moon(midnight[i]):
            continue
        observing_night[i] = True
        mjd_grid = midnight[i] + t_centers
        dark, gray, bright = ephem.get_program(mjd_grid)
        hours[0, i] = dt * np.count_nonzero(dark)
        hours[1, i] = dt * np.count_nonzero(gray)
        hours[2, i] = dt * np.count_nonzero(bright)

    # Initialize the plot.
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5), squeeze=True)

    if style == 'localtime':

        # Loop over nights to build image data to plot.
        program = np.zeros((num_nights, len(t_centers)))
        for i in np.arange(num_nights):
            if not observing_night[i]:
                continue
            mjd_grid = midnight[i] + t_centers
            dark, gray, bright = ephem.get_program(mjd_grid)
            program[i][dark] = 1.
            program[i][gray] = 2.
            program[i][bright] = 3.

        # Prepare a custom colormap.
        colors = [bg_color, program_color['DARK'],
                  program_color['GRAY'], program_color['BRIGHT']]
        cmap = matplotlib.colors.ListedColormap(colors, 'programs')

        ax.imshow(program.T, origin='lower', interpolation='none',
                  aspect='auto', cmap=cmap, vmin=-0.5, vmax=+3.5,
                  extent=[xaxis_lo, xaxis_hi, night_start, night_stop])

        # Display 24-hour local time on y axis.
        y_lo = int(np.ceil(night_start))
        y_hi = int(np.floor(night_stop))
        y_ticks = np.arange(y_lo, y_hi + 1, dtype=int)
        y_labels = ['{:02d}h'.format(hr) for hr in (24 + y_ticks) % 24]
        config = desisurvey.config.Configuration()
        ax.set_ylabel('Local Time [{0}]'
                      .format(config.location.timezone()), fontsize='x-large')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

    else:

        x = xaxis_lo + np.arange(num_nights) + 0.5
        y = hours if style == 'histogram' else np.cumsum(hours, axis=1)
        size = min(15., (300./num_nights) ** 2)
        opts = dict(linestyle='-', marker='.' if size > 1 else None,
                    markersize=size)
        plt.plot(x, y[0], color=program_color['DARK'], **opts)
        plt.plot(x, y[1], color=program_color['GRAY'], **opts)
        plt.plot(x, y[2], color=program_color['BRIGHT'], **opts)

        ax.set_axis_bgcolor(bg_color)
        plt.ylim(0, 1.07 * y.max())

    # Display dates on the x axis.
    ax.set_xlabel('Survey Date (observing {0} / {1} nights)'
                  .format(np.count_nonzero(observing_night), num_nights),
                  fontsize='x-large')
    ax.set_xlim(xaxis_start, xaxis_stop)
    if num_nights < 50:
        # Major ticks at month boundaries.
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(tz=tz))
        ax.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter('%m/%y', tz=tz))
        # Minor ticks at day boundaries.
        ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator(tz=tz))
    elif num_nights <= 650:
        # Major ticks at month boundaries with no labels.
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(tz=tz))
        ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        # Minor ticks at month midpoints with centered labels.
        ax.xaxis.set_minor_locator(
            matplotlib.dates.MonthLocator(bymonthday=15, tz=tz))
        ax.xaxis.set_minor_formatter(
            matplotlib.dates.DateFormatter('%m/%y', tz=tz))
        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')
    else:
        # Major ticks at year boundaries.
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(tz=tz))
        ax.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter('%Y', tz=tz))

    ax.grid(b=True, which='major', color='w', linestyle=':', lw=1)

    # Draw program labels.
    y = 0.975
    opts = dict(fontsize='xx-large', fontweight='bold', xy=(0, 0),
                horizontalalignment='center', verticalalignment='top',
                xycoords='axes fraction', textcoords='axes fraction')
    ax.annotate('DARK {0:.1f}h'.format(hours[0].sum()), xytext=(0.2, y),
                color=program_color['DARK'], **opts)
    ax.annotate('GRAY {0:.1f}h'.format(hours[1].sum()), xytext=(0.5, y),
                color=program_color['GRAY'], **opts)
    ax.annotate(
        'BRIGHT {0:.1f}h'.format(hours[2].sum()), xytext=(0.8, y),
        color=program_color['BRIGHT'], **opts)

    plt.tight_layout()
    if save:
        plt.savefig(save)

    return fig, ax


def plot_next_field(date_string, obs_num, ephem, window_size=7.,
                    max_airmass=2., min_moon_sep=50., max_bin_area=1.,
                    save=None):
    """Plot diagnostics for the next field selector.

    The matplotlib package must be installed to use this function.

    Parameters
    ----------
    date_string : string
        Observation date of the form 'YYYYMMDD'.
    obs_num : int
        Observation number on the specified night, counting from zero.
    ephem : :class:`desisurvey.ephemerides.Ephemerides`
        Ephemerides covering this night.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec
    import matplotlib.colors

    # Location of Mayall at KPNO.
    where = astropy.coordinates.EarthLocation.from_geodetic(
        lat='31d57m50.30s', lon='-111d35m57.61s', height=2120.*u.m)

    # Lookup the afternoon plan and observed tiles for this night.
    plan = astropy.table.Table.read('obsplan' + date_string + '.fits')
    obs = astropy.table.Table.read('obslist' + date_string + '.fits')

    # Lookup the requested exposure number.
    if obs_num < 0 or obs_num >= len(obs):
        raise ValueError('Invalid obs_num {0}'.format(obs_num))
    tile = obs[obs_num]
    t_start = astropy.time.Time(tile['MJD'], format='mjd')
    exptime = tile['EXPTIME']

    # Lookup the corresponding tile in the plan.
    tile_plan = plan[plan['TILEID'] == tile['TILEID']]
    if len(tile_plan) != 1:
        raise RuntimeError('Observed tile {0} not in plan for {1} obs #{2}.'
                           .format(tile['TILEID'], date_string, obs_num))
    tile_plan = tile_plan[0]

    # Use the exposure midpoint for calculations below.
    when = t_start + 0.5 * exptime * u.s
    when.location = where
    night = ephem.get_night(when)

    # Calculate the program during this night.
    midnight = night['noon'] + 0.5
    t_edges = midnight + np.linspace(
        -window_size, +window_size, 2 * window_size * 60 + 1) / 24.
    t_centers = 0.5 * (t_edges[1:] + t_edges[:-1])
    dark, gray, bright = ephem.get_program(t_centers)

    # Determine the program during the exposure midpoint.
    t_index = np.digitize(when.mjd, t_edges) - 1
    if dark[t_index]:
        program = 'DARK'
    elif gray[t_index]:
        program = 'GRAY'
    else:
        program = 'BRIGHT'

    # Restrict the plan to tiles in the current program.
    plan = plan[plan['PROGRAM'] == program]

    # Calculate the apparent HA of each tile in the plan in degrees [0,360].
    lst = when.sidereal_time(kind='apparent').to(u.deg).value
    ha = np.fmod(lst - plan['RA'] + 360, 360)

    # Calculate the difference between the actual and optimum HA for
    # each tile in degrees [-180, +180].
    dha = np.fmod(ha - 15 * plan['HA'] + 540, 360) - 180

    # Build (ra,dec) grid where quantities will be tabulated.
    max_bin_area = max_bin_area * (np.pi / 180.) ** 2

    # Pick the number of bins in cos(DEC) and RA to use.
    n_cos_dec = int(np.ceil(2 / np.sqrt(max_bin_area)))
    n_ra = int(np.ceil(4 * np.pi / max_bin_area / n_cos_dec))
    # Calculate the actual pixel area in sq. degrees.
    bin_area = 360 ** 2 / np.pi / (n_cos_dec * n_ra)

    # Calculate the bin edges in degrees.
    ra_edges = np.linspace(-180., +180., n_ra + 1)
    dec_edges = np.degrees(np.arcsin(np.linspace(-1., +1., n_cos_dec + 1)))

    # Place a SkyCoord at the center of each bin.
    ra = 0.5 * (ra_edges[1:] + ra_edges[:-1])
    dec = 0.5 * (dec_edges[1:] + dec_edges[:-1])[:, np.newaxis]
    radec_grid = astropy.coordinates.SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    # Ignore atmospheric refraction and restrict to small airmass, for speed.
    altaz_frame = astropy.coordinates.AltAz(
        location=where, obstime=when, pressure=0)

    # Transform (ra,dec) grid to (alt,az)
    altaz_grid = radec_grid.transform_to(altaz_frame)
    zenith = 90 * u.deg - altaz_grid.alt

    # Calculate airmass at each grid point.
    airmass = 1. / np.cos(zenith.to(u.rad).value)

    # Calculate position of moon.
    moon_pos = desisurvey.ephemerides.get_object_interpolator(
        night, 'moon', altaz=True)
    moon_alt, moon_az = moon_pos(when.mjd)
    moon_altaz = astropy.coordinates.AltAz(
        alt=moon_alt * u.deg, az=moon_az * u.deg,
        location=where, obstime=when, pressure=0)
    moon_radec = moon_altaz.transform_to(astropy.coordinates.ICRS)
    moon_ra = moon_radec.ra.to(u.deg).value
    moon_dec = moon_radec.dec.to(u.deg).value
    moon_sep = moon_altaz.separation(altaz_grid).to(u.deg).value
    # Mask regions of the sky too close to the moon if it is above the horizon.
    if moon_alt > 0:
        data_mask = moon_sep < min_moon_sep
    else:
        data_mask = None

    data = desiutil.plots.prepare_data(
        airmass, mask=data_mask, clip_lo='!1.',
        clip_hi='!{0}'.format(max_airmass))

    # Prepare tile coords to display.
    ra_plan = plan['RA'] * u.deg
    dec_plan = plan['DEC'] * u.deg
    ra_tile = [tile['RA']] * u.deg
    dec_tile = [tile['DEC']] * u.deg

    time = when.datetime.time()
    title1 = '{} {:02d}:{:02d} Exp #{}/{} {:.1f}[{:.0f}]m {:04d}-{:6s}'.format(
        when.datetime.date(), time.hour, time.minute, obs_num + 1, len(obs),
        exptime / 60., tile_plan['EXPLEN'] / 60.,
        tile['TILEID'], tile['PROGRAM'])
    title2 = 'LST={:.1f}[{:.0f},{:.0f}]d X={:.2f} M={:.1f}%'.format(
        lst, 15. * tile_plan['LSTMIN'], 15. * tile_plan['LSTMAX'],
        tile['AIRMASS'], 100. * tile['MOONFRAC'])

    # Initialize the plots.
    fig = plt.figure(figsize=(11, 7.5))
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[8, 1],
                                      top=0.95, hspace=0.01)
    top = plt.subplot(gs[0])
    btm = plt.subplot(gs[1])
    plt.suptitle(title1 + ' ' + title2, fontsize='x-large')

    # Draw the night program.
    program_code = np.zeros((len(t_centers), 1))
    program_code[dark] = 1.
    program_code[gray] = 2.
    program_code[bright] = 3.
    colors = ['lightblue', program_color['DARK'],
              program_color['GRAY'], program_color['BRIGHT']]
    cmap = matplotlib.colors.ListedColormap(colors, 'programs')
    btm.imshow(program_code.T, origin='lower', interpolation='none',
               aspect='auto', cmap=cmap, vmin=-0.5, vmax=+3.5,
               extent=[-window_size, +window_size, 0, 1])
    btm.set_yticks([])
    window_hours = int(np.floor(window_size))
    x_ticks = np.arange(2 * window_hours + 1) - window_hours
    x_labels = ['{0:02d}h'.format((hr + 24) % 24) for hr in x_ticks]
    btm.set_xticks(x_ticks)
    btm.set_xticklabels(x_labels)
    btm.axvline(24 * (when.mjd - midnight), color='r', lw=2)
    btm.set_xlabel('Local Time [UTC-7]')

    # Draw an all-sky overview of the afternoon plan and selected tile.
    basemap = desiutil.plots.init_sky(
        galactic_plane_color=program_color[program], ax=top)
    desiutil.plots.plot_grid_map(data, ra_edges, dec_edges, label='Airmass',
                                 cmap='viridis', basemap=basemap)

    dha = desiutil.plots.prepare_data(
        dha, clip_lo=-180, clip_hi=180, save_limits=True)
    desiutil.plots.plot_sky_circles(ra_plan, dec_plan, data=dha, cmap='bwr',
                                    colorbar=False, edgecolor='none',
                                    basemap=basemap)
    desiutil.plots.plot_sky_circles(ra_tile, dec_tile, field_of_view=10,
                                    facecolors=(1, 1, 1, 0.5),
                                    edgecolor=program_color[tile['PROGRAM']],
                                    basemap=basemap)

    # Draw the location of the moon.
    basemap.scatter(
        moon_ra, moon_dec, marker='x', color='k', s=100, latlon=True)

    if save:
        plt.savefig(save)


def plot_planner(p, where=None, when=None, night_summary='dark',
                 dust=True, monsoon=True, fullmoon=True, cmap='magma',
                 save=None):
    """Plot a summary of the planner observing efficiency forecast.

    Requires that the matplotlib package is installed.

    Parameters
    ----------
    p : desisurvey.plan.Planner
        The planner object to use.
    where : string or None
    when : string or None
    night_summary : string
    dust : bool
        Should dust extinction be included in the observing efficiency?
    monsoon : bool
        Do not observe during scheduled monsoon shutdowns?
    fullmoon : bool
        Do not observe during scheduled full-moon breaks?
    cmap : matplotlib colormap spec
        Colormap to use to represent observing efficiency. Not used for a
        time series plot.
    save : string or None
        Name of file where plot should be saved.  Format is inferred from
        the extension.

    Returns
    -------
    tuple
        Tuple (figure, axes) returned by ``plt.subplots()``.
    """
    import matplotlib.pyplot as plt

    if where and when:
        raise ValueError('Cannot specify both where and when.')

    # Reshape time axis into (nights, times). This operation creates
    # a new view with no memory copy.
    fexp = p.fexp.reshape(p.num_nights, p.num_times, -1)

    # Project out the spatial axis if requested. If dust included,
    # we cannot avoid making a temporary copy of the large fexp array.
    if dust and where:
        fexp = fexp.copy() * p.fdust
    if where == 'best':
        # Observe the best pixel during each time slot.
        fexp = fexp.max(axis=2)
    elif where == 'random':
        # Observe a random pixel during each time slot.
        fexp = fexp.mean(axis=2)
    elif where is not None:
        raise ValueError('Invalid where: {0}.'.format(where))

    # Summarize each night using the specified summary statistic.
    if night_summary == 'best':
        # Pick the best time to observe each pixel.
        fexp = fexp.max(axis=1)
    elif night_summary == '24hr':
        # Average over each 24hour period.
        fexp = fexp.mean(axis=1)
    elif night_summary == 'dark':
        # Average over night observing times only.
        n_night_bins = (fexp > 0).sum(axis=1)
        mask = n_night_bins > 0
        fexp = fexp.sum(axis=1)
        fexp[mask] /= n_night_bins[mask]
    else:
        raise ValueError('Invalid night_summary: {0}.'.format(night_summary))

    # Zero out monsoon nights if requested.
    if monsoon:
        fexp[p.calendar['monsoon']] = 0.

    # Zero out full-moon nights if requested.
    if fullmoon:
        fexp[p.calendar['fullmoon']] = 0.

    # Project out the night axis if requested.
    if when == 'best':
        # Observe each pixel during its best night.
        fexp = fexp.max(axis=0)
    elif when == 'random':
        # Observe each pixel during a random night scheduled for observations.
        scheduled = np.ones(p.num_nights, bool)
        if monsoon:
            scheduled[p.calendar['monsoon']] = False
        if fullmoon:
            scheduled[p.calendar['fullmoon']] = False
        fexp = fexp[scheduled].mean(axis=0)
    elif when is not None:
        raise ValueError('Invalid when: {0}.'.format(when))

    # Apply dust exposure factors if requested.
    if dust and not where:
        fexp *= p.fdust

    # Prepare plot labels.
    date_label = 'Nights {0} to {1}'.format(p.start_date, p.stop_date)
    sky_label = 'Sky {0:,} sq.deg. (increasing RA)'.format(
        int(round(p.footprint_area)))

    # Make the plot.
    fig, ax = plt.subplots(figsize=(10, 5.75))
    if when:
        # Plot an all-sky map.
        assert fexp.shape == (len(p.footprint_pixels),)
        # Reconstruct a healpix map masked to our footprint.
        m = np.zeros(p.npix)
        m[p.footprint_pixels] = fexp
        data = desiutil.plots.prepare_data(
            m, mask=~p.footprint, clip_lo=0., clip_hi=1., save_limits=True)
        # Draw the map.
        label = 'Observing Efficiency ({0}, {1})'.format(when, night_summary)
        desiutil.plots.plot_healpix_map(data, label=label, cmap=cmap)
    elif where:
        # Project a time series.
        assert fexp.shape == (p.num_nights,)
        x = np.arange(p.num_nights)
        ax.fill_between(x, fexp, color='r', alpha=0.5)
        ax.plot(x, fexp, 'k-', lw=0.5)
        ax.set_xticks([])
        ax.set_ylim(0., 1.)
        ax.set_xlabel(date_label, fontsize='x-large')
        ax.set_ylabel('Observing Efficiency ({0}, {1})'
                      .format(where, night_summary), fontsize='x-large')
    else:
        # Plot a 2D image.
        assert fexp.shape == (p.num_nights, len(p.footprint_pixels))
        ax.imshow(fexp.T, interpolation='none', aspect='auto', origin='lower',
                  cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(date_label, fontsize='x-large')
        ax.set_ylabel(sky_label, fontsize='x-large')

    plt.tight_layout()
    if save:
        plt.savefig(save)
    return fig, ax


def plot_iers(which='auto', num_points=500, save=None):
    """Plot IERS data from 2015-2025.

    Plots the UT1-UTC time offset and polar x,y motions over a 10-year
    period that includes the DESI survey, to demonstrate the time ranges
    when different sources (IERS-A, IERS-B) are used and where values
    are predicted then fixed.

    This function is primarily intended to document and debug the
    :func:`desisurvey.utils.update_iers` and :func:`desisurvey.utils.freeze_iers` functions.

    Requires that the matplotlib package is installed.

    Parameters
    ----------
    which : 'auto', 'A', 'B' or 'frozen'
        Select which IERS table source to use.  The default 'auto' matches
        the internal astropy default.  Use 'A' or 'B' to force the source
        to be either the latest IERS-A table (which will be downloaded),
        or the IERS-B table packaged with the current version of astropy.
        The 'frozen' option calls :func:`desisurvey.utils.freeze_iers`.
    num_points : int
        The number of times covering 2015-25 to calculate and plot.
    save : string or None
        Name of file where plot should be saved.  Format is inferred from
        the extension.

    Returns
    -------
    tuple
        Tuple (figure, axes) returned by ``plt.subplots()``.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates
    import matplotlib.ticker

    config = desisurvey.config.Configuration()

    # Calculate UTC midnight timestamps covering 2015 - 2025
    start = astropy.time.Time('2015-01-01 00:00')
    stop = astropy.time.Time('2025-01-01 00:00')
    mjd = np.linspace(start.mjd, stop.mjd, num_points)
    times = astropy.time.Time(mjd, format='mjd')

    t_lo = matplotlib.dates.date2num(start.datetime)
    t_hi = matplotlib.dates.date2num(stop.datetime)
    t = np.linspace(t_lo, t_hi, num_points)

    t_start = matplotlib.dates.date2num(config.first_day())
    t_stop = matplotlib.dates.date2num(config.last_day())

    # Load the specified IERS table.
    if which == 'auto':
        iers = astropy.utils.iers.IERS_Auto.open()
    elif which == 'B':
        iers = astropy.utils.iers.IERS_B.open()
    elif which == 'A':
        # This requires network access to download the latest file.
        iers = astropy.utils.iers.IERS_A.open(astropy.utils.iers.IERS_A_URL)
    elif which == 'frozen':
        desisurvey.utils.freeze_iers()
        iers = astropy.utils.iers.IERS_Auto.open()
    else:
        raise ValueError('Invalid which option.')

    # Calculate UT1-UTC using the IERS table.
    dt, dt_status = iers.ut1_utc(times, return_status=True)
    dt = dt.to(u.s).value

    # Calculate polar x,y motion using the IERS table.
    pmx, pmy, pm_status = iers.pm_xy(times, return_status=True)
    pmx = pmx.to(u.arcsec).value
    pmy = pmy.to(u.arcsec).value

    assert np.all(dt_status == pm_status)
    codes = np.unique(dt_status)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    labels = {-2: 'Fixed', 0: 'IERS-B', 1: 'IERS-A', 2: 'Predict'}
    styles = {-2: 'r:', 0: 'b-', 1: 'go', 2: 'r--'}
    ms = 3
    for code in codes:
        sel = dt_status == code
        ax[0].plot(t[sel], dt[sel], styles[code], ms=ms, label=labels[code])
        ax[1].plot(t[sel], pmx[sel], styles[code], ms=ms, label='Polar x')
        ax[1].plot(t[sel], pmy[sel], styles[code], ms=ms, label='Polar y')

    ax[0].legend(ncol=4)
    ax[0].set_ylabel('UT1 - UTC [s]')
    ax[1].set_ylabel('Polar x,y motion [arcsec]')

    for i in range(2):
        # Vertical lines showing the survey start / stop dates.
        ax[i].axvline(t_start, ls='--', c='k')
        ax[i].axvline(t_stop, ls='--', c='k')

    # Use year labels for the horizontal axis.
    xaxis = ax[1].xaxis
    xaxis.set_major_locator(matplotlib.dates.YearLocator())
    xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    ax[i].set_xlim(start.datetime, stop.datetime)

    plt.tight_layout()
    if save:
        plt.savefig(save)

    return fig, ax
