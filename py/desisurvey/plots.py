"""Utility functions for plotting DESI survey progress and planning.
"""
from __future__ import print_function, division

import numpy as np

import astropy.table

import desiutil.plots

import desisurvey.nightcal


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
    # Color code for each pass.
    pass_color = ['k', 'k', 'k', 'k', 'gray', 'orange', 'orange', 'orange']
    for p in range(8):
        basemap = desiutil.plots.init_sky(ax=ax[pass_map[p]],
                                          dec_labels=[-120, 0, 120],
                                          ra_labels=None,
                                          galactic_plane_color=pass_color[p])
        # Select the tiles in this pass.
        sel = np.where(passnum == p)[0]
        z_sel = desiutil.plots.prepare_data(
            z[sel], clip_lo=vmin, clip_hi=vmax, save_limits=True)
        # Plot the sky map for this pass.
        desiutil.plots.plot_sky_circles(
            ra_center=ra[sel], dec_center=dec[sel], data=z_sel,
            colorbar=True, basemap=basemap, edgecolor='none', label=label)
        # Plot the histogram of values for this pass.
        ax[0, 2].hist(z[sel], color=pass_color[p], **hopts)

    # Decorate the histogram subplot.
    ax[0, 2].set_ylim(0, None)
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


def plot_program(filename, save=None):
    """Plot an overview of the DARK/GRAY/BRIGHT program.

    The matplotlib package must be installed to use this function.

    Parameters
    ----------
    filename : string
        Name of a FITS file in the format written by :func:`nightcal.getCalAll`,
        normally of the form 'ephem_<MJD1>_<MJD2>.fits'.
    save : string or None
        Name of file where plot should be saved.  Format is inferred from
        the extension.

    Returns
    -------
    tuple
        Tuple (figure, axes) returned by ``plt.subplots()``.
    """
    import matplotlib.pyplot as plt

    t = astropy.table.Table.read(filename)

    mjd = np.floor(t['MJDsunset'])
    dt = mjd - mjd[0]

    # Center vertical axis on median "midnight"
    midnight = 0.5 * (t['MJDsunset'] + t['MJDsunrise'])
    mjd += np.median(midnight - mjd)

    # Scale vertical axis to hours relative to median midnight.
    y = lambda y_mjd: 24. * (y_mjd - mjd)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7), squeeze=True)

    # Fill the time between 13deg twilights with orange (BRIGHT).
    ax.fill_between(dt, y(t['MJDe13twi']), y(t['MJDm13twi']),
                          color='orange')

    # Fill the time between 15deg twilights with black (DARK).
    ax.fill_between(dt, y(t['MJDetwi']), y(t['MJDmtwi']), color='k')

    # Loop over nights.
    for i, row in enumerate(t):
        # Identify bright time when the moon is up.
        t_moon, bright = desisurvey.nightcal.get_bright(row, interval_mins=2.)
        if len(t_moon) == 0:
            continue
        # Gray time requires 15deg twighlight, moon up and not bright.
        gray = ~bright & (t_moon > row['MJDetwi']) & (t_moon < row['MJDmtwi'])
        # Fill in bright and gray times during this night.
        today = np.full(len(t_moon), dt[i])
        ax.scatter(today[bright], 24 * (t_moon[bright] - mjd[i]),
                         marker='s', s=5, lw=0, c='orange')
        ax.scatter(today[gray], 24 * (t_moon[gray] - mjd[i]),
                         marker='s', s=5, lw=0, c='gray')

    ax.set_axis_bgcolor('lightblue')
    ax.set_xlim(0, dt[-1])
    ax.set_ylim(-6.5, +6.5)
    ax.grid()
    ax.set_xlabel('Elapsed Survey Time [days]', fontsize='x-large')
    ax.set_ylabel('Time During Night [hours]', fontsize='x-large')

    plt.tight_layout()
    if save:
        plt.savefig(save)

    return fig, ax
