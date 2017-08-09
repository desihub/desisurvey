"""Schedule future DESI observations.
"""
from __future__ import print_function, division

import datetime

import numpy as np

import astropy.io.fits
import astropy.table
import astropy.time
import astropy.coordinates
import astropy.units as u

import desiutil.log

import desisurvey.config
import desisurvey.utils
import desisurvey.etc


class Scheduler(object):
    """Initialize a survey scheduler from a FITS file.

    Parameters
    ----------
    name : str
        Name of the scheduler file to load.  Relative paths refer to
        our config output path.
    """
    def __init__(self, name='scheduler.fits'):

        self.log = desiutil.log.get_logger()
        config = desisurvey.config.Configuration()
        output_file = config.get_path(name)

        hdus = astropy.io.fits.open(output_file)
        header = hdus[0].header
        self.start_date = desisurvey.utils.get_date(header['START'])
        self.stop_date = desisurvey.utils.get_date(header['STOP'])
        self.num_nights = (self.stop_date - self.start_date).days
        self.nside = header['NSIDE']
        self.step_size = header['STEP'] * u.min
        self.npix = 12 * self.nside ** 2
        self.pix_area = 360. ** 2 / np.pi / self.npix * u.deg ** 2

        self.tiles = astropy.table.Table.read(output_file, hdu='TILES')
        self.tile_coords = astropy.coordinates.ICRS(
            ra=self.tiles['ra'] * u.deg, dec=self.tiles['dec'] * u.deg)

        self.calendar = astropy.table.Table.read(output_file, hdu='CALENDAR')
        self.etable = astropy.table.Table.read(output_file, hdu='EPHEM')

        self.t_edges = hdus['GRID'].data
        self.t_centers = 0.5 * (self.t_edges[1:] + self.t_edges[:-1])
        self.num_times = len(self.t_centers)

        static = astropy.table.Table.read(output_file, hdu='STATIC')
        self.footprint_pixels = static['pixel'].data
        self.footprint = np.zeros(self.npix, bool)
        self.footprint[self.footprint_pixels] = True
        self.footprint_area = len(self.footprint_pixels) * self.pix_area
        self.fdust = static['dust'].data
        self.pixel_ra = static['ra'].data
        self.pixel_dec = static['dec'].data

        self.fexp = hdus['DYNAMIC'].data
        assert self.fexp.shape == (
            self.num_nights * self.num_times, len(self.footprint_pixels))

        # Load fallback weights into a (3,3) matrix with row, column
        # indices 0=DARK, 1=GRAY, 2=BRIGHT. The row index specifies the
        # current nominal program, and the column index specifies the
        # alternate fall back program.  Weights are relative to 1 for
        # staying within the nominal program.
        fb = config.fallback_weights
        self.fallback_weights = np.identity(3)
        self.fallback_weights[0, 1] = fb.gray_in_dark()
        self.fallback_weights[0, 2] = fb.bright_in_dark()
        self.fallback_weights[1, 0] = fb.dark_in_gray()
        self.fallback_weights[1, 2] = fb.bright_in_gray()
        self.fallback_weights[2, 0] = fb.dark_in_bright()
        self.fallback_weights[2, 1] = fb.gray_in_bright()
        assert np.all(self.fallback_weights >= 0)

        # Calculate target exposure time in seconds of each tile at nominal
        # conditions.
        self.tnom = np.empty(len(self.tiles))
        for i, program in enumerate(('DARK', 'GRAY', 'BRIGHT')):
            sel = self.tiles['program'] == i + 1
            self.tnom[sel] = getattr(
                config.nominal_exposure_time, program)().to(u.s).value

    def index_of_time(self, when):
        """Calculate the temporal bin index of the specified time.

        Parameters
        ----------
        when : astropy.time.Time

        Returns
        -------
        int
            Index of the temporal bin that ``when`` falls into.
        """
        # Look up the night number for this time.
        night = desisurvey.utils.get_date(when)
        i = (night - self.start_date).days
        if i < 0 or i >= self.num_nights:
            raise ValueError('Time out of range: {0} - {1}.'
                             .format(self.start_date, self.stop_date))
        # Find the time relative to local midnight in days.
        dt = when.mjd - desisurvey.utils.local_noon_on_date(night).mjd - 0.5
        # Lookup the corresponding time index offset for this night.
        j = np.digitize(dt, self.t_edges) - 1
        if j < 0 or j >= self.num_times:
            raise ValueError('Time is not during the night.')
        # Combine the night and time indices.
        return i * self.num_times + j

    def time_of_index(self, ij):
        """Calculate the time at the center of the specified temporal bin.

        Parameters
        ----------
        ij : int
            Index of a temporal bin.

        Returns
        -------
        astropy.time.Time
            Time at the center of the specified temporal bin.
        """
        if ij < 0 or ij >= self.num_nights * self.num_times:
            raise ValueError('Time index out of range.')
        i = ij // self.num_times
        j = ij % self.num_times
        night = self.start_date + datetime.timedelta(days=i)
        mjd = desisurvey.utils.local_noon_on_date(
            night).mjd + 0.5 + self.t_centers[j]
        return astropy.time.Time(mjd, format='mjd')

    def index_of_tile(self, tile_id):
        """Calculate the spatial bin index of the specified tile.

        Parameters
        ----------
        tile_id : int
            Tile identifier in the DESI footprint.

        Returns
        -------
        int
            Index of the spatial bin that ``tile_id`` falls into.
        """
        sel = np.where(self.tiles['tileid'] == tile_id)[0]
        if len(sel) == 0:
            raise ValueError('Invalid tile_id: {0}.'.format(tile_id))
        assert len(sel) == 1
        return self.tiles['map'][sel[0]]

    def instantaneous_efficiency(self, when, cutoff, seeing, transparency,
                                 progress, snr2frac, mask=None):
        """Calculate the instantaneous efficiency of all tiles.

        Calculated as ``texp / (texp + toh) * eff`` where the exposure time
        ``texp`` accounts for the tile's remaining SNR**2 and the current
        exposure-time factors, and the ovhead time ``toh`` accounts for
        readout, slew, focus and cosmic splits.

        Parameters
        ----------
        when : astropy.time.Time
            Time for which the efficiency should be calculated.
        cutoff : astropy.time.Time
            Time by which observing must stop tonight.  Any exposure expected
            to last beyond this cutoff will be assigned an instantaneous
            efficiency of zero.
        seeing : float or array
            FWHM seeing value in arcseconds.
        transparency : float or array
            Dimensionless transparency value in the range [0-1].
        progress : desisurvey.progress.Progress
            Record of observations made so far.  Will not be modified by
            calling this method.
        snr2frac : array
            Array of total SNR2 fractions observed so far for each tile.
        mask : array or None
            Boolean mask array specifying which tiles to consider. Use all
            tiles if None.

        Returns
        -------
        tuple
            Tuple (ieff, toh_initial, t_midpt, previous) where ieff are the
            instantaneous efficiences for each tile, toh_initial are the
            corresponding initial overhead times without any cosmic-ray splits,
            t_midpt is the estimated exposure midpoint in seconds relative
            to the input time, and previous is the sky coordinates of the
            previous exposure, or None.
        """
        config = desisurvey.config.Configuration()

        # Select the time slice to use.
        fexp = self.fexp[self.index_of_time(when)].copy()

        # Apply dust extinction.
        fexp *= self.fdust

        # Allocate arrays.
        ntiles = len(self.tiles)
        texp = np.zeros(ntiles)
        toh_initial = np.zeros(ntiles)
        nsplit = np.zeros(ntiles, int)
        ieff = np.zeros(ntiles)
        if mask is None:
            mask = np.ones(ntiles, bool)

        # Calculate the exposure efficiency of each tile at the current time
        # and with the current conditions.
        eff = fexp[self.tiles['map']]
        eff /= desisurvey.etc.seeing_exposure_factor(seeing)
        eff /= desisurvey.etc.transparency_exposure_factor(transparency)

        # Ignore tiles with no efficiency, i.e., not visible now.
        mask = mask & (eff > 0)

        # Scale target exposure time for remaining SNR**2.
        # Any targets that have reached SNR2 >= 1 will have zero efficiency.
        tnom = self.tnom * np.maximum(0., 1. - snr2frac)

        # Estimate the required exposure time.
        texp[mask] = tnom[mask] / eff[mask]

        # Determine the previous pointing if we need to include slew time
        # in the overhead calcluations.
        if progress.last_tile is None:
            # No slew needed for next exposure.
            previous = None
            # No readout time needed for previous exposure.
            deadtime = config.readout_time()
        else:
            last = progress.last_tile
            # How much time has elapsed since the last exposure ended?
            last_end = (last['mjd'] + last['exptime'] / 86400.).max()
            deadtime = max(0., when.mjd - last_end) * u.day
            # Is this the first exposure of the night?
            today = desisurvey.utils.get_date(when)
            if desisurvey.utils.get_date(last_end) < today:
                # No slew necessary.
                previous = None
            else:
                # Where are we slewing from?
                previous = astropy.coordinates.ICRS(
                    ra=last['ra'] * u.deg, dec=last['dec'] * u.deg)

        # Calculate the initial overhead times for each possible tile.
        toh_initial[mask] = desisurvey.utils.get_overhead_time(
            previous, self.tile_coords[mask], deadtime).to(u.s).value

        # Add overhead for any cosmic-ray splits.
        nsplit[mask] = np.floor(
            (texp[mask] / config.cosmic_ray_split().to(u.s).value))
        toh = toh_initial + nsplit * config.readout_time().to(u.s).value

        # Zero the efficiency of any tiles whose exposures cannot be completed
        # before the cutoff.  (Could soften to include tiles whose first
        # exposure is expected to complete before the cutoff.)
        beyond_cutoff = toh + texp > (cutoff - when).to(u.s).value
        mask[beyond_cutoff] = False

        # Calculate the instantaneous efficiency.
        ieff[mask] = tnom[mask] / (toh[mask] + texp[mask])

        # Calculate the exposure midpoint relative to the current time.
        t_midpt = 0.5 * (toh_initial + texp)

        return ieff, toh_initial, t_midpt, previous

    def hourangle_score(self, when, tmid, design_HA, sigma=7.5):
        """
        """
        # Get the current apparent local sidereal time.
        when.location = desisurvey.utils.get_location()
        LST = when.sidereal_time('apparent')
        # Calculate each tile's current hour angle in degrees.
        HA = (LST - self.tiles['ra'] * u.deg).to(u.deg).value
        # Adjust HA for the estimated exposure midpoints (in seconds).
        HA += tmid * 15. / 3600.
        # Ensure that HA is in the range [-180, +180].
        HA = np.fmod(HA + 540, 360) - 180
        assert np.min(HA) >= -180 and np.max(HA) <= +180
        # Compare actual HA with design HA.
        dHA = HA ##- design_HA
        return np.exp(-0.5 * (dHA / sigma) ** 2)

    def rank_score(self, when):
        """
        Calculate percentile rank of present time compared with future times.
        """
        # Get the temporal index for this time.
        ij = self.index_of_time(when)
        # Round down to the start of this night.
        ij0 = ij - (ij % self.num_times)
        # Find the maximum efficiency for each pixel during each remaining
        # night.
        num_nights = self.num_nights - ij0 // self.num_times
        num_pix = len(self.footprint_pixels)
        future_exp = np.zeros((num_nights + 1, num_pix), dtype=np.float32)
        future_exp[1:] = self.fexp[ij0:].reshape(
            num_nights, self.num_times, num_pix).max(axis=1)
        # Compare future max efficiencies with the current efficiencies.
        future_exp[0] = self.fexp[ij]
        # Sort the efficiencies for each pixel.
        order = np.argsort(future_exp, axis=1)
        # Calculate the rank [0, num_nights] of the current time.
        #rank = np.where(order == 0)[1]
        rank2 = order.argsort(axis=1)
        #assert np.all(rank == rank2[0])
        # Each pixel's rank score is the sort position of its current efficiency
        # compared with all future nightly max efficiencies. Scale from 0-1.
        return rank2[0] / float(num_nights)

    def ratio_score(self, when):
        """
        Calculate ratio of present time compared with best future time.
        """
        # Get the temporal index for this time.
        ij = self.index_of_time(when)
        # Find the maximum efficiency of all future times.
        fexp_max = self.fexp[ij:].max(axis=0)
        # Take the ratio of the current efficiency with the future max
        # efficiency.
        ratio = self.fexp[ij] / fexp_max
        return ratio

    def plot_scores(self, when, cutoff, score, mask, ephem, prev, offset=-60):
        """
        """
        # Get the current apparent local sidereal time.
        when.location = desisurvey.utils.get_location()
        LST = when.sidereal_time('apparent').to(u.deg).value
        LSTmax = LST + 360 * (cutoff - when).to(u.day).value
        ra0 = np.fmod(LST - offset + 360, 360) + offset
        ra0max = np.fmod(LSTmax - offset + 360, 360) + offset
        if ephem['moon_alt'] > 0:
            moon_ra = np.fmod(ephem['moon_ra'] - offset + 360, 360) + offset
            moon_dec = ephem['moon_dec']
            moon_size = 10 + 90 * ephem['moon_frac']
        else:
            moon_size = 0
        if prev is not None:
            prev_ra = np.fmod(
                prev.ra.to(u.deg).value - offset + 360, 360) + offset
            prev_dec = prev.dec.to(u.deg).value
        best = np.argmax(score)
        best_ra = np.fmod(self.tiles['ra'][best] - offset + 360, 360) + offset
        best_dec = self.tiles['dec'][best]

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(
            3, 1, figsize=(7, 10), sharex=True, sharey=True)
        axes[0].set_xlim(-60, 300)
        axes[0].set_ylim(-20, 80)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99,
                            wspace=0.01, hspace=0.01)
        vmax = max(0.1, np.max(score))
        for i in range(3):
            ax = axes[i]
            psel = mask & (self.tiles['program'] == i+1)
            ra = np.fmod(self.tiles['ra'][psel] - offset + 360, 360) + offset
            dec = self.tiles['dec'][psel]
            s = ax.scatter(ra, dec, c=score[psel], s=15,
                           vmin=0., vmax=vmax, cmap='viridis', lw=0)
            ax.axvline(ra0, color='k', ls='-', lw=4, alpha=0.25)
            ax.axvline(ra0max, color='r', ls='-', lw=1)
            if moon_size > 0:
                ax.scatter(moon_ra, moon_dec, s=moon_size,
                           marker='s', facecolors='none', edgecolors='r')
            if prev is not None:
                ax.scatter(prev_ra, prev_dec,
                           marker='+', color='r', lw=3, s=150, alpha=0.5)
            if psel[best]:
                ax.scatter(best_ra, best_dec, facecolors='none',
                           edgecolors='r', lw=1, s=150, alpha=0.5)
        plt.colorbar(s, ax=axes[2], orientation='horizontal',
                     fraction=0.10, aspect=40, pad=0.01)
        plt.show()

    def next_tile(self, when, cutoff, seeing, transparency, progress,
                  strategy, plan, plot=False):
        """Return the next tile to observe.

        Parameters
        ----------
        when : astropy.time.Time
            Time at which the next tile decision is being made.
        cutoff : astropy.time.Time
            Time by which observing must stop tonight.
        seeing : float or array
            FWHM seeing value in arcseconds.
        transparency : float or array
            Dimensionless transparency value in the range [0-1].
        progress : desisurvey.progress.Progress
            Record of observations made so far.  Will not be modified by
            calling this method.
        strategy : str
            Strategy to use for scheduling tiles during each night.
        plan : astropy.table.Table
            Table that specifies active tiles and design hour angles.
        plot : bool
            Display a diagnostic plot, which requires that matplotlib is
            installed. Not intended for production use.

        Returns
        -------
        dict
            Dictionary describing the next tile to observe or None if no
            suitable target is available.  The dictionary will contain the
            following keys: tileID, RA, DEC, Program, Ebmv, moon_illum_frac,
            MoonDist, MoonAlt and overhead.  Overhead is the delay (with time
            units) before the shutter can be opened due to slewing and reading
            out any previous exposure.
        """
        config = desisurvey.config.Configuration()
        # Look up the night number for this time.
        night = desisurvey.utils.get_date(when)
        i = (night - self.start_date).days
        if i < 0 or i >= self.num_nights:
            raise ValueError('Time out of range: {0} - {1}.'
                             .format(self.start_date, self.stop_date))
        # Lookup the current ephemerides.
        dt = when.mjd - desisurvey.utils.local_noon_on_date(night).mjd - 0.5
        ij0 = (i * self.num_times + np.digitize(dt, self.t_edges) - 1)
        ephem0 = self.etable[ij0]
        # Only schedule active tiles.
        mask = (plan['priority'] > 0) & plan['available']
        # Do not re-schedule tiles that have reached their min SNR**2 fraction.
        mask = mask & (progress._table['status'] < 2)
        if not np.any(mask):
            self.log.warn('No active tiles at {0}.'.format(when.datetime))
            return None
        # Calculate instantaneous efficiencies, initial overhead times,
        # and estimated exposure midpoints.
        snr2frac = progress._table['snr2frac'].data.sum(axis=1)
        ieff, toh, tmid, prev = self.instantaneous_efficiency(
            when, cutoff, seeing, transparency, progress, snr2frac, mask)
        # Lookup the temporal bin index that each tile's estimated exposure
        # midpoint lands in.
        dt = when.mjd - desisurvey.utils.local_noon_on_date(night).mjd - 0.5
        ij = (i * self.num_times +
              np.digitize(dt + tmid[mask] / 86400., self.t_edges) - 1)
        # Lookup the program during each tile's estimated exposure midpoint:
        # 1 = DARK, 2 = GRAY, 3 = BRIGHT.
        obs_program = self.etable['program'][ij]
        # Program might be zero at begin/end of night: set this to BRIGHT.
        obs_program[obs_program == 0] = 3
        # Lookup the program each candidate tile is assigned to.
        tile_program = self.tiles['program'][mask]
        # Calculate each tile's score using the requested strategy.
        # Initialize score = priority, then apply multiplicative factors.
        score = plan['priority'].data.copy()
        strategy = strategy.split('+')
        if 'greedy' in strategy:
            score *= ieff
        else:
            # Zero the score of unobservable tiles.
            score[ieff == 0] = 0.
        if 'fallback' in strategy:
            score[mask] *= (
                self.fallback_weights[obs_program - 1, tile_program - 1])
        else:
            # Zero score for tiles that would be observed outside their program.
            score[mask] *= (obs_program == tile_program)
        if 'HA' in strategy:
            score *= self.hourangle_score(when, tmid, plan['hourangle'])
        if 'ratio' in strategy:
            score *= self.ratio_score(when)[self.tiles['map']]
        if 'rank' in strategy:
            score *= self.rank_score(when)[self.tiles['map']]
        if plot:
            self.plot_scores(when, cutoff, score, mask, ephem0, prev)
        if np.max(score) <= 0:
            self.log.debug('Found max score {0} at {1}.'
                           .format(np.max(score), when.datetime))
            return None
        # Pick the tile with the highest score.
        best = np.argmax(score)
        tile = self.tiles[best]
        # Calculate separation angle in degrees between the selected tile
        # and the moon.
        ephem = self.etable[ij[np.argmax(score[mask])]]
        moon = astropy.coordinates.ICRS(
            ra=ephem['moon_ra'] * u.deg, dec=ephem['moon_dec'] * u.deg)
        moon_sep = self.tile_coords[best].separation(moon).to(u.deg)
        if ephem['moon_alt'] > 0 and moon_sep < config.avoid_bodies.moon():
            self.log.debug('Best tile has moon_sep {0}.'.format(moon_sep))
            return None
        # Should also check planet separation vetos here
        # ...
        # Calculate the altitude angle of the selected tile.
        zenith = desisurvey.utils.get_observer(
            when, alt=90 * u.deg, az=0 * u.deg
            ).transform_to(astropy.coordinates.ICRS)
        alt = 90 * u.deg - self.tile_coords[best].separation(zenith)
        if alt < config.min_altitude():
            self.log.debug('Best tile has altitude {0:.1f}.'.format(alt))
            return None
        # Prepare the dictionary to return.
        target = dict(tileID=tile['tileid'], RA=tile['ra'], DEC=tile['dec'],
                      Program=('DARK', 'GRAY', 'BRIGHT')[tile['program'] - 1],
                      Ebmv=tile['EBV'], moon_illum_frac=ephem['moon_frac'],
                      MoonDist=moon_sep.value, MoonAlt=ephem['moon_alt'],
                      overhead=toh[best] * u.s)
        return target


# Imports only needed by initialize() go here.
import specsim.atmosphere

import desimodel.io

import desisurvey.ephemerides


def initialize(ephem, start_date=None, stop_date=None, step_size=5.*u.min,
               healpix_nside=16, output_name='scheduler.fits'):
    """Calculate exposure-time factors over a grid of times and pointings.

    Takes about 9 minutes to run and writes a 1.3Gb output file with the
    default parameters.

    Requires that healpy is installed.

    Parameters
    ----------
    ephem : `desisurvey.ephem.Ephemerides`
        Tabulated ephemerides data to use for planning.
    start_date : date or None
        Survey planning starts on the evening of this date. Must be convertible
        to a date using :func:`desisurvey.utils.get_date`.  Use the first night
        of the ephemerides when None.
    stop_date : date or None
        Survey planning stops on the morning of this date. Must be convertible
        to a date using :func:`desisurvey.utils.get_date`.  Use the first night
        of the ephemerides when None.
    step_size : astropy.units.Quantity
        Exposure-time factors are tabulated at this interval during each night.
    healpix_nside : int
        Healpix NSIDE parameter to use for binning the sky. Must be a power of
        two.  Values larger than 16 will lead to holes in the footprint with
        the current implementation.
    output_name : str
        Name of the FITS output file where results are saved. A relative path
        refers to the :meth:`configuration output path
        <desisurvey.config.Configuration.get_path>`.
    """
    import healpy

    log = desiutil.log.get_logger()

    # Freeze IERS table for consistent results.
    desisurvey.utils.freeze_iers()

    config = desisurvey.config.Configuration()
    output_name = config.get_path(output_name)

    start_date = desisurvey.utils.get_date(start_date or ephem.start)
    stop_date = desisurvey.utils.get_date(stop_date or ephem.stop)
    if start_date >= stop_date:
        raise ValueError('Expected start_date < stop_date.')
    mjd = ephem._table['noon']
    sel = ((mjd >= desisurvey.utils.local_noon_on_date(start_date).mjd) &
           (mjd < desisurvey.utils.local_noon_on_date(stop_date).mjd))
    t = ephem._table[sel]
    num_nights = len(t)

    # Build a grid of elapsed time relative to local midnight during each night.
    midnight = t['noon'] + 0.5
    t_edges = desisurvey.ephemerides.get_grid(step_size)
    t_centers = 0.5 * (t_edges[1:] + t_edges[:-1])
    num_points = len(t_centers)

    # Create an empty HDU0 with header info.
    header = astropy.io.fits.Header()
    header['START'] = str(start_date)
    header['STOP'] = str(stop_date)
    header['NSIDE'] = healpix_nside
    header['NPOINTS'] = num_points
    header['STEP'] = step_size.to(u.min).value
    hdus = astropy.io.fits.HDUList()
    hdus.append(astropy.io.fits.ImageHDU(header=header))

    # Save time grid.
    hdus.append(astropy.io.fits.ImageHDU(name='GRID', data=t_edges))

    # Load the list of tiles to observe.
    tiles = astropy.table.Table(
        desimodel.io.load_tiles(onlydesi=True, extra=False))

    # Build the footprint as a healpix map of the requested size.
    # The footprint includes any pixel containing at least one tile center.
    npix = healpy.nside2npix(healpix_nside)
    footprint = np.zeros(npix, bool)
    pixels = healpy.ang2pix(
            healpix_nside, np.radians(90 - tiles['DEC'].data),
            np.radians(tiles['RA'].data))
    footprint[np.unique(pixels)] = True
    footprint_pixels = np.where(footprint)[0]
    num_footprint = len(footprint_pixels)
    log.info('Footprint contains {0} pixels.'.format(num_footprint))

    # Sort pixels in order of increasing phi + 60deg so that the north and south
    # galactic caps are contiguous in the arrays we create below.
    pix_theta, pix_phi = healpy.pix2ang(healpix_nside, footprint_pixels)
    pix_dphi = np.fmod(pix_phi + np.pi / 3, 2 * np.pi)
    sort_order = np.argsort(pix_dphi)
    footprint_pixels = footprint_pixels[sort_order]
    # Calculate sorted pixel (ra,dec).
    pix_theta, pix_phi = healpy.pix2ang(healpix_nside, footprint_pixels)
    pix_ra, pix_dec = np.degrees(pix_phi), 90 - np.degrees(pix_theta)

    # Record per-tile info needed for planning.
    table = astropy.table.Table()
    table['tileid'] = tiles['TILEID'].astype(np.int32)
    table['ra'] = tiles['RA'].astype(np.float32)
    table['dec'] = tiles['DEC'].astype(np.float32)
    table['EBV'] = tiles['EBV_MED'].astype(np.float32)
    table['pass'] = tiles['PASS'].astype(np.int16)
    # Map each tile ID to the corresponding index in our spatial arrays.
    mapper = np.zeros(npix, int)
    mapper[footprint_pixels] = np.arange(len(footprint_pixels))
    table['map'] = mapper[pixels].astype(np.int16)
    # Use a small int to identify the program.
    table['program'] = np.zeros(len(tiles), np.int16)
    for i, program in enumerate(('DARK', 'GRAY', 'BRIGHT')):
        table['program'][tiles['PROGRAM'] == program] = i + 1
    assert np.all(table['program'] > 0)
    hdu = astropy.io.fits.table_to_hdu(table)
    hdu.name = 'TILES'
    hdus.append(hdu)

    # Average E(B-V) for all tiles falling into a pixel.
    tiles_per_pixel = np.bincount(pixels, minlength=npix)
    EBV = np.bincount(pixels, weights=tiles['EBV_MED'], minlength=npix)
    EBV[footprint] /= tiles_per_pixel[footprint]

    # Calculate dust extinction exposure-time factor.
    f_EBV = 1. / desisurvey.etc.dust_exposure_factor(EBV)

    # Save HDU with the footprint and static dust exposure map.
    table = astropy.table.Table()
    table['pixel'] = footprint_pixels
    table['dust'] = f_EBV[footprint_pixels]
    table['ra'] = pix_ra
    table['dec'] = pix_dec
    hdu = astropy.io.fits.table_to_hdu(table)
    hdu.name = 'STATIC'
    hdus.append(hdu)

    # Prepare a table of calendar data.
    calendar = astropy.table.Table()
    calendar['midnight'] = midnight
    calendar['monsoon'] = np.zeros(num_nights, bool)
    calendar['fullmoon'] = np.zeros(num_nights, bool)
    calendar['weather'] = np.zeros(num_nights, np.float32)
    weather_weights = 1 - desisurvey.utils.dome_closed_probabilities()

    # Prepare a table of ephemeris data.
    etable = astropy.table.Table()
    etable['program'] = np.zeros(num_nights * num_points, dtype=np.int16)
    etable['moon_frac'] = np.zeros(num_nights * num_points, dtype=np.float32)
    etable['moon_ra'] = np.zeros(num_nights * num_points, dtype=np.float32)
    etable['moon_dec'] = np.zeros(num_nights * num_points, dtype=np.float32)
    etable['moon_alt'] = np.zeros(num_nights * num_points, dtype=np.float32)
    etable['zenith_ra'] = np.zeros(num_nights * num_points, dtype=np.float32)
    etable['zenith_dec'] = np.zeros(num_nights * num_points, dtype=np.float32)

    # Tabulate MJD and apparent LST values for each time step. We don't save
    # MJD values since they are cheap to reconstruct from the index, but
    # do use them below.
    mjd0 = desisurvey.utils.local_noon_on_date(start_date).mjd + 0.5
    mjd = mjd0 + np.arange(num_nights)[:, np.newaxis] + t_centers
    times = astropy.time.Time(
        mjd, format='mjd', location=desisurvey.utils.get_location())
    etable['lst'] = times.sidereal_time('apparent').flatten().to(u.deg).value

    # Build sky coordinates for each pixel in the footprint.
    pix_theta, pix_phi = healpy.pix2ang(healpix_nside, footprint_pixels)
    pix_ra, pix_dec = np.degrees(pix_phi), 90 - np.degrees(pix_theta)
    pix_sky = astropy.coordinates.ICRS(pix_ra * u.deg, pix_dec * u.deg)

    # Initialize exposure factor calculations.
    alt, az = np.full(num_points, 90.) * u.deg, np.zeros(num_points) * u.deg
    fexp = np.zeros((num_nights * num_points, num_footprint), dtype=np.float32)
    vband_extinction = 0.15154
    one = np.ones((num_points, num_footprint))

    # Loop over nights.
    for i in range(num_nights):
        night = ephem.get_night(midnight[i])
        date = desisurvey.utils.get_date(midnight[i])
        if date.day == 1:
            log.info('Starting {0} (completed {1}/{2} nights)'
                     .format(date.strftime('%b %Y'), i, num_nights))
        # Initialize the slice of the fexp[] time index for this night.
        sl = slice(i * num_points, (i + 1) * num_points)
        # Do we expect to observe on this night?
        calendar[i]['monsoon'] = desisurvey.utils.is_monsoon(midnight[i])
        calendar[i]['fullmoon'] = ephem.is_full_moon(midnight[i])
        # Look up expected dome-open fraction due to weather.
        calendar[i]['weather'] = weather_weights[date.month - 1]
        # Calculate the program during this night.
        mjd = midnight[i] + t_centers
        dark, gray, bright = ephem.get_program(mjd)
        etable['program'][sl][dark] = 1
        etable['program'][sl][gray] = 2
        etable['program'][sl][bright] = 3
        # Zero the exposure factor whenever we are not oberving.
        fexp[sl] = (dark | gray | bright)[:, np.newaxis]
        # Transform the local zenith to (ra,dec).
        zenith = desisurvey.utils.get_observer(
            times[i], alt=alt, az=az).transform_to(astropy.coordinates.ICRS)
        etable['zenith_ra'][sl] = zenith.ra.to(u.deg).value
        etable['zenith_dec'][sl] = zenith.dec.to(u.deg).value
        # Calculate zenith angles to each pixel in the footprint.
        pix_sep = pix_sky.separation(zenith[:, np.newaxis])
        # Zero the exposure factor for pixels below the horizon.
        visible = pix_sep < 90 * u.deg
        fexp[sl][~visible] = 0.
        # Calculate the airmass exposure-time penalty.
        X = desisurvey.utils.cos_zenith_to_airmass(np.cos(pix_sep[visible]))
        fexp[sl][visible] /= desisurvey.etc.airmass_exposure_factor(X)
        # Loop over objects we need to avoid.
        for name in config.avoid_bodies.keys:
            f_obj = desisurvey.ephemerides.get_object_interpolator(night, name)
            # Calculate this object's (dec,ra) path during the night.
            obj_dec, obj_ra = f_obj(mjd)
            sky_obj = astropy.coordinates.ICRS(
                ra=obj_ra[:, np.newaxis] * u.deg,
                dec=obj_dec[:, np.newaxis] * u.deg)
            # Calculate the separation angles to each pixel in the footprint.
            obj_sep = pix_sky.separation(sky_obj)
            if name == 'moon':
                etable['moon_ra'][sl] = obj_ra
                etable['moon_dec'][sl] = obj_dec
                # Calculate moon altitude during the night.
                moon_alt, _ = desisurvey.ephemerides.get_object_interpolator(
                    night, 'moon', altaz=True)(mjd)
                etable['moon_alt'][sl] = moon_alt
                moon_zenith = (90 - moon_alt[:,np.newaxis]) * u.deg
                moon_up = moon_alt > 0
                assert np.all(moon_alt[gray] > 0)
                # Calculate the moon illuminated fraction during the night.
                moon_frac = ephem.get_moon_illuminated_fraction(mjd)
                etable['moon_frac'][sl] = moon_frac
                # Convert to temporal moon phase.
                moon_phase = np.arccos(2 * moon_frac[:,np.newaxis] - 1) / np.pi
                # Calculate scattered moon V-band brightness at each pixel.
                V = specsim.atmosphere.krisciunas_schaefer(
                    pix_sep, moon_zenith, obj_sep,
                    moon_phase, vband_extinction).value
                # Estimate the exposure time factor from V.
                X = np.dstack((one, np.exp(-V), 1/V, 1/V**2, 1/V**3))
                T = X.dot(desisurvey.etc._moonCoefficients)
                # No penalty when the moon is below the horizon.
                T[moon_alt < 0, :] = 1.
                fexp[sl] *= 1. / T
                # Veto pointings within avoidance size when the moon is
                # above the horizon. Apply Gaussian smoothing to the veto edge.
                veto = np.ones_like(T)
                dsep = (obj_sep - config.avoid_bodies.moon()).to(u.deg).value
                veto[dsep <= 0] = 0.
                veto[dsep > 0] = 1 - np.exp(-0.5 * (dsep[dsep > 0] / 3) ** 2)
                veto[moon_alt < 0] = 1.
                fexp[sl] *= veto
            else:
                # Lookup the avoidance size for this object.
                size = getattr(config.avoid_bodies, name)()
                # Penalize the exposure-time with a factor
                # 1 - exp(-0.5*(obj_sep/size)**2)
                penalty = 1. - np.exp(-0.5 * (obj_sep / size).to(1).value ** 2)
                fexp[sl] *= penalty

    # Save calendar table.
    hdu = astropy.io.fits.table_to_hdu(calendar)
    hdu.name = 'CALENDAR'
    hdus.append(hdu)

    # Save ephemerides table.
    hdu = astropy.io.fits.table_to_hdu(etable)
    hdu.name = 'EPHEM'
    hdus.append(hdu)

    # Save dynamic exposure-time factors.
    hdus.append(astropy.io.fits.ImageHDU(name='DYNAMIC', data=fexp))

    # Finalize the output file.
    try:
        hdus.writeto(output_name, overwrite=True)
    except TypeError:
        # astropy < 1.3 uses the now deprecated clobber.
        hdus.writeto(output_name, clobber=True)
    log.info('Plan initialization saved to {0}'.format(output_name))


if __name__ == '__main__':
    """This should eventually be made into a first-class script entry point.
    """
    #stop = desisurvey.utils.get_date('2019-10-03')
    #stop = desisurvey.utils.get_date('2020-07-13')
    stop = None
    ephem = desisurvey.ephemerides.Ephemerides(stop_date=stop)
    initialize(ephem, stop_date=stop)
