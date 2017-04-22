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


def plot_observations(start_date=None, stop_date=None, what='EXPTIME',
                      verbose=False, save=None):
    """Plot a summary of observed tiles.

    Reads the file ``obsall_list.fits`` and uses :func:`plot_sky_passes` to
    display a summary of observed tiles in each pass.

    Parameters
    ----------
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
        column name in the obsall_list FITS file.  Useful values include
        EXPTIME, OBSSN2, AIRMASS, SEEING.
    verbose : bool
        Print a summary of observed tiles.
    save : string or None
        Name of file where plot should be saved.

    Returns
    -------
    tuple
        Tuple (figure, axes) returned by ``plt.subplots()``.
    """
    t = astropy.table.Table.read('obslist_all.fits')
    if what not in t.colnames:
        raise ValueError('Valid names are: {0}'
                         .format(','.join(t.colnames)))

    start_date = desisurvey.utils.get_date(start_date or t['MJD'][0])
    stop_date = desisurvey.utils.get_date(stop_date or (t['MJD'][-1] + 1.0))
    if start_date >= stop_date:
        raise ValueError('Expected start_date < stop_date.')
    date_label = '{0} to {1}'.format(start_date, stop_date)

    # Convert date range to MJDs at local noon.
    start_mjd = desisurvey.utils.local_noon_on_date(start_date).mjd
    stop_mjd = desisurvey.utils.local_noon_on_date(stop_date).mjd

    # Trim table to requested observations.
    sel = (t['STATUS'] > 0) & (t['MJD'] >= start_mjd) & (t['MJD'] < stop_mjd)
    t = t[sel]

    if verbose:
        print('Observing summary for {0}:'.format(date_label))
        for pass_num in range(8):
            sel_pass = t['PASS'] == pass_num
            n_exps = np.count_nonzero(sel_pass)
            n_tiles = len(np.unique(t['TILEID'][sel_pass]))
            print('Observed {0:7d} tiles with {1:5d} repeats from PASS {2}'
                  .format(n_tiles, n_exps - n_tiles, pass_num))
        n_exps = len(t)
        n_tiles = len(np.unique(t['TILEID']))
        print('Observed {0:7d} tiles with {1:5d} repeats total.'
              .format(n_tiles, n_exps - n_tiles))

    label = '{0} ({1})'.format(what, date_label)
    return desisurvey.plots.plot_sky_passes(
        t['RA'], t['DEC'], t['PASS'], t[what],
        label=label, save=save)


def plot_program(ephem, start_date=None, stop_date=None, window_size=7.,
                 num_points=500, save=None):
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
    window_size : float
        Number of hours on both sides of local midnight to display on the
        vertical axis.
    num_points : int
        Number of subdivisions of the vertical axis to use for tabulating
        the program during each night. The resulting resolution will be
        ``2 * window_size / num_points`` hours.
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

    # Determine plot date range.
    start_date = desisurvey.utils.get_date(start_date or ephem.start)
    stop_date = desisurvey.utils.get_date(stop_date or ephem.stop)
    if start_date >= stop_date:
        raise ValueError('Expected start_date < stop_date.')
    mjd = ephem._table['MJDstart']
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

    # Display 24-hour local time on y axis.
    window_int = int(np.floor(window_size))
    y_ticks = np.arange(-window_int, +window_int + 1, dtype=int)
    y_labels = ['{:02d}h'.format(hr) for hr in (24 + y_ticks) % 24]

    # Build a grid of elapsed time relative to local midnight during each night.
    midnight = t['MJDstart'] + 0.5
    t_edges = np.linspace(-window_size, +window_size, num_points + 1) / 24.
    t_centers = 0.5 * (t_edges[1:] + t_edges[:-1])

    # Loop over nights to build image data to plot.
    program = np.zeros((num_nights, len(t_centers)))
    for i in np.arange(num_nights):
        mjd_grid = midnight[i] + t_centers
        dark, gray, bright = ephem.get_program(mjd_grid)
        program[i][dark] = 1.
        program[i][gray] = 2.
        program[i][bright] = 3.

    # Prepare a custom colormap.
    colors = ['lightblue', program_color['DARK'],
              program_color['GRAY'], program_color['BRIGHT']]
    cmap = matplotlib.colors.ListedColormap(colors, 'programs')

    # Make the plot.
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5), squeeze=True)

    ax.imshow(program.T, origin='lower', interpolation='none',
              aspect='auto', cmap=cmap, vmin=-0.5, vmax=+3.5,
              extent=[xaxis_lo, xaxis_hi, -window_size, +window_size])

    # Display 24-hour local time on the y axis.
    config = desisurvey.config.Configuration()
    ax.set_ylabel('Local Time [{0}]'
                  .format(config.location.timezone()), fontsize='x-large')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Display dates on the x axis.
    ax.set_xlabel('Survey Date', fontsize='x-large')
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
    y = 0.025
    opts = dict(fontsize='xx-large', fontweight='bold',
                horizontalalignment='center',
                xy=(0, 0), textcoords='axes fraction')
    ax.annotate('DARK', xytext=(0.2, y), color=program_color['DARK'], **opts)
    ax.annotate('GRAY', xytext=(0.5, y), color=program_color['GRAY'], **opts)
    ax.annotate(
        'BRIGHT', xytext=(0.8, y), color=program_color['BRIGHT'], **opts)

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
    midnight = night['MJDstart'] + 0.5
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
    moon_pos = desisurvey.ephemerides.get_moon_interpolator(night)
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
