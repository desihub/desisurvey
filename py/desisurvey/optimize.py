"""Optimize future DESI observations.
"""
from __future__ import print_function, division

import pkg_resources

import numpy as np
import scipy.special

import astropy.table
import astropy.units as u

import desisurvey.config


def wrap(angle, offset):
    """Wrap values in the range [0, 360] to [offset, offset+360].
    """
    return np.fmod(angle - offset + 360, 360) + offset


class Optimizer(object):
    """Initialize the hour angle assignments for specified tiles.

    Parameters
    ----------
    sched : desisurvey.schedule.Scheduler
        The scheduler object to use for the observing calendar and exposure
        time forecasts.
    program : 'DARK', 'GRAY' or 'BRIGHT'
        Which program to optimize.  Determines the nominal exposure time.
    subset : array or None
        An array of tile ID values to optimize within the specified program.
        Optimizes all tiles in the program if None.
    start : date or None
        Only consider available LST starting from this date.  Use the
        nominal survey start date if None.
    stop : date or None
        Only consider available LST before this end date.  Use the nominal
        survey stop date if None.
    nbins : int
        Number of LST histogram bins to use when calculating the optimization
        metrics.
    init : 'zero', 'info' or 'flat'
        Method for initializing tile hour angles: 'zero' sets all hour angles
        to zero, 'init' reads 'tile-info.fits' and 'flat' matches the CDF of
        available LST to planned LST (without accounting for exposure time).
    origin : float
        Rotate DEC values in plots so that the left edge is at this value
        in degrees.
    center : float or None
        Used by the 'flat' initialization method to specify the starting
        DEC for the CDF balancing algorithm. When None, the 'flat' method
        scans over a grid of center values and picks the best one, but this
        is relatively slow.  Ignored unless init is 'flat'.
    seed : int or None
        Random number seed to use for stochastic elements of the optimizer.
        Do not use None if reproducible results are required.
    oversampling : int
        Oversampling of the LST histogram (relative to nbins) to use when
        spreading each planned observation over its estimated exposure time.
    weights : array
        Array of relative weights to use when selecting which LST bin to
        optimize next.  Candidate bins are ordered by an estimated improvement.
        The length of the weights array determines how many candidates to
        consider, in decreasing order, and the weight values determines their
        relative weight. The next bin to optimize is then selected at random.
    """
    def __init__(self, sched, program, subset=None, start=None, stop=None,
                 nbins=90, init='info', origin=-60, center=220, seed=123,
                 oversampling=32, weights=[5, 4, 3, 2, 1]):

        config = desisurvey.config.Configuration()
        self.gen = np.random.RandomState(seed)
        self.cum_weights = np.asarray(weights, float).cumsum()
        self.cum_weights /= self.cum_weights[-1]

        if start is None:
            start = sched.start_date
        else:
            start = desisurvey.utils.get_date(start)
        if stop is None:
            stop = sched.stop_date
        else:
            stop = desisurvey.utils.get_date(stop)
        if start >= stop:
            raise ValueError('Expected start < stop.')

        # Calculate the time available in bins of LST for this program.
        e = sched.etable
        p_index = dict(DARK=1, GRAY=2, BRIGHT=3)[program]
        sel = (e['program'] == p_index).reshape(
            sched.num_nights, sched.num_times)
        # Zero out nights during monsoon and full moon.
        sel[sched.calendar['monsoon']] = False
        sel[sched.calendar['fullmoon']] = False
        # Zero out nights outside [start:stop].
        sel[:(start - sched.start_date).days] = False
        sel[(stop - sched.start_date).days:] = False
        # Accumulate times in hours over the full survey.
        dt = sched.step_size.to(u.hour).value
        wgt = dt * np.ones((sched.num_nights, sched.num_times))
        # Weight nights for weather availability.
        lst = wrap(e['lst'][sel.flat], origin)
        wgt *= sched.calendar['weather'][:, np.newaxis]
        wgt = wgt[sel].flat
        self.lst_hist, self.lst_edges = np.histogram(
            lst, bins=nbins, range=(origin, origin + 360), weights=wgt)
        self.lst_centers = 0.5 * (self.lst_edges[1:] + self.lst_edges[:-1])
        self.nbins = nbins

        # Get nominal exposure time for this program,
        # converted to LST equivalent in degrees.
        texp_nom = getattr(config.nominal_exposure_time, program)()
        self.dlst_nom = 360 * texp_nom.to(u.day).value

        # Load the tiles for this program.
        p_tiles = sched.tiles[sched.tiles['program'] == p_index]
        # Restrict to a subset of tiles in this program, if requested.
        if subset is not None:
            idx = np.searchsorted(p_tiles['tileid'], subset)
            if not np.all(p_tiles['tileid'][idx] == subset):
                raise ValueError(
                    'Invalid subset for {0} program.'.format(program))
            p_tiles = p_tiles[idx]

        self.ra = wrap(p_tiles['ra'].data, origin)
        self.dec = p_tiles['dec'].data
        self.tid = p_tiles['tileid'].data
        self.ntiles = len(self.ra)

        # Calculate the maximum |HA| in degrees allowed for each tile to stay
        # above the survey minimum altitude (plus a 5 deg padding).
        cosZ_min = np.cos(90 * u.deg - (config.min_altitude() + 5 * u.deg))
        latitude = desisurvey.config.Configuration().location.latitude()
        cosHA_min = (
            (cosZ_min - np.sin(self.dec * u.deg) * np.sin(latitude)) /
            (np.cos(self.dec * u.deg) * np.cos(latitude))).value
        self.max_abs_ha = np.degrees(np.arccos(cosHA_min))

        # Calculate static dust exposure factors for each tile.
        self.dust_factor = desisurvey.etc.dust_exposure_factor(p_tiles['EBV'])

        print('{0} program: {1:.1f}h to observe {2} tiles (texp_nom {3:.1f}).'
              .format(program, self.lst_hist.sum(), len(p_tiles), texp_nom))

        # Precompute coefficients for exposure time calculations.
        latitude = np.radians(config.location.latitude())
        self.A = np.sin(np.radians(self.dec)) * np.sin(latitude)
        self.B = np.cos(np.radians(self.dec)) * np.cos(latitude)

        # Initialize oversampled schedule planning calculations.
        lst_edges_os = np.linspace(
            origin, origin + 360, self.nbins * oversampling + 1)
        self.lst_centers_os = 0.5 * (lst_edges_os[1:] + lst_edges_os[:-1])
        self.oversampling = oversampling
        self.scale_history = []
        self.MSE_history = []

        # Initialize improve() counters.
        self.nslow = 0
        self.nstuck = 0
        self.nimprove = 0

        # Initialize HA assignments for each tile.
        if init == 'zero':
            self.ha = np.zeros(self.ntiles)
        elif init == 'info':
            info = astropy.table.Table.read(
                pkg_resources.resource_filename(
                    'desisurvey', 'data/tile-info.fits'), hdu=1)
            # Lookup each tile ID.
            assert np.all(np.diff(info['TILEID']) > 0)
            idx = np.searchsorted(info['TILEID'], self.tid)
            assert np.all(info['TILEID'][idx] == self.tid)
            self.ha = info['HA'][idx]
            ha_clipped = np.clip(self.ha, -self.max_abs_ha, +self.max_abs_ha)
            if not np.all(self.ha == ha_clipped):
                print('Clipping info HA assignments to airmass limits.')
                self.ha = ha_clipped
        elif init == 'flat':
            if center is None:
                centers = np.arange(0, 360, 5)
            else:
                centers = [center]
            MSE_min = np.inf
            for center in centers:
                # Histogram LST values relative to the specified center.
                lst = wrap(e['lst'][sel.flat], center)
                hist, edges = np.histogram(
                    lst, bins=nbins, range=(center, center + 360), weights=wgt)
                lst_cdf = np.zeros_like(edges)
                lst_cdf[1:] = np.cumsum(hist)
                lst_cdf /= lst_cdf[-1]
                idx = np.argsort(np.argsort(wrap(self.ra, center)))
                tile_cdf = (0.5 + idx) / self.ntiles
                ra = wrap(p_tiles['ra'].data, center)
                tile_lst = np.interp(tile_cdf, lst_cdf, edges)
                ha = np.fmod(tile_lst - ra, 360)
                # Clip tiles to their airmass limits.
                ha = np.clip(ha, -self.max_abs_ha, +self.max_abs_ha)
                self.plan_tiles_os = self.get_plan(ha)
                self.use_plan()
                if  self.MSE_history[-1] < MSE_min:
                    self.ha = ha.copy()
                    MSE_min = self.MSE_history[-1]
                    center_best = center
            if len(centers) > 1:
                import matplotlib.pyplot as plt
                plt.plot(centers, self.scale_history, 'r-')
                plt.xlabel('Central RA [deg]')
                plt.ylabel('Efficiency')
                plt.axvline(center_best)
                rhs = plt.twinx()
                rhs.plot(centers, self.MSE_history, 'bx')
                rhs.set_ylabel('MSE')
                plt.xlim(0, 360)
                plt.show()
            self.scale_history = []
            self.MSE_history = []
        else:
            raise ValueError('Invalid init option: {0}.'.format(init))

        # Calculate schedule plan with initial HA asignments.
        self.plan_tiles_os = self.get_plan(self.ha)
        self.use_plan()
        self.ha_initial = self.ha.copy()
        self.num_adjustments = np.zeros(self.ntiles, int)

    def get_exptime(self, ha, subset=None):
        """Estimate exposure times for the specified tiles.

        Estimates account for airmass and dust extinction only.

        Parameters
        ----------
        ha : array
            Array of hour angle assignments in degrees.
        subset : array or None
            Restrict calculation to a subset of tiles specified by the
            indices in this array, or use all tiles pass to the constructor
            if None.

        Returns
        -------
        tuple
            Tuple (exptime, subset) where exptime is an array of estimated
            exposure times in seconds and subset is the input subset or
            else a slice initialized for all tiles.
        """
        # Calculate for all tiles if no subset is specified.
        if subset is None:
            subset = slice(None)
        # Calculate best-case exposure times (i.e., using only airmass & dust)
        # in degrees for the specified HA assignments.
        cosZ = self.A[subset] + self.B[subset] * np.cos(np.radians(ha))
        X = desisurvey.utils.cos_zenith_to_airmass(cosZ)
        exptime = (self.dlst_nom * desisurvey.etc.airmass_exposure_factor(X) *
                   self.dust_factor[subset])
        return exptime, subset

    def get_plan(self, ha, subset=None):
        """Calculate an LST usage plan for specified hour angle assignments.

        The plan is calculated on the oversampled LST grid.

        Parameters
        ----------
        ha : array
            Array of hour angle assignments in degrees.
        subset : array or None
            Restrict calculation to a subset of tiles specified by the
            indices in this array, or use all tiles pass to the constructor
            if None.

        Returns
        -------
        array
            Array of shape (ntiles, nbins * oversampling) giving the exposure time in hours that each tile needs in each oversampled LST bin.
            When a subset is specified, ntiles only indexes tiles in the subset.
        """
        exptime, subset = self.get_exptime(ha, subset)
        # Calculate LST windows for each tile's exposure.
        lst_mid = self.ra[subset] + ha
        lst_min = lst_mid - 0.5 * exptime
        lst_max = lst_mid + 0.5 * exptime
        # Sample each tile's exposure on the oversampled LST grid.
        dlo = np.fmod(
            self.lst_centers_os - lst_min[:, np.newaxis] + 540, 360) - 180
        dhi = np.fmod(
            self.lst_centers_os - lst_max[:, np.newaxis] + 540, 360) - 180
        # Normalize to hours per LST bin. Factor of 2 sharpens the erf() edges.
        return 24. / self.nbins * np.maximum(
            0, 0.5 * (scipy.special.erf(2.0 * dlo) -
                      scipy.special.erf(2.0 * dhi)))

    def eval_MSE(self, plan_hist):
        """Evaluate the mean-squared error metric for the specified plan.

        This is the metric that :meth:`optimize` attempts to improve. It
        measures the similarity of the available and planned LST histogram
        shapes, but not their normalizations.  A separate :meth:`eval_scale`
        metric measures how efficiently the plan uses the available LST.

        The histogram of available LST is rescaled to the same total time (area)
        before calculating residuals relative to the planned LST usage.

        Parameters
        ----------
        plan_hist : array
            Histogram of planned LST usage for all tiles.

        Returns
        -------
        float
            Mean squared error value.
        """
        # Rescale the available LST total time to the plan total time.
        scale = plan_hist.sum() / self.lst_hist.sum()
        return np.sum((plan_hist - scale * self.lst_hist) ** 2)

    def eval_scale(self, plan_hist):
        """Evaluate the efficiency of the specified plan.

        Calculates the minimum scale factor applied to the available
        LST histogram so that the planned LST usage is always <= the scaled
        available LST histogram.  This value can be intepreted as the fraction
        of the available time required to complete all tiles.

        This metric is only loosely correlated with the MSE metric, so provides
        a useful independent check that the optimization is producing the
        desired results.

        This metric is not well defined if any bin of the available LST
        histogram is empty, which indicates that some tiles will not be
        observable during the [start:stop] range being optimized.  In this case,
        only bins with some available LST are included in the scale calculation.

        Parameters
        ----------
        plan_hist : array
            Histogram of planned LST usage for all tiles.

        Returns
        -------
        float
            Scale factor.
        """
        nonzero = self.lst_hist > 0
        return (plan_hist[nonzero] / self.lst_hist[nonzero]).max()

    def use_plan(self):
        """Use the current oversampled plan and update internal arrays.

        Calculates the downsampled `plan_tiles` and `plan_hist` arrays from
        the oversampled per-tile `plan_tiles_os` array, and records the
        current values of the MSE and scale metrics.
        """
        # Downsample the high resolution plan to the LST bins.
        self.plan_tiles = self.plan_tiles_os.reshape(
            self.ntiles, self.nbins, self.oversampling).mean(axis=-1)
        # Sum over tiles at low resolution.
        self.plan_hist = self.plan_tiles.sum(axis=0)
        # Calculate the amount that the nominal LST budget needs to be rescaled
        # to accomodate this plan.
        self.scale_history.append(self.eval_scale(self.plan_hist))
        self.MSE_history.append(self.eval_MSE(self.plan_hist))

    def next_bin(self):
        """Select which LST bin to adjust next.

        The algorithm determines which bin of the planned LST usage histogram
        should be decreased in order to maximize the decrease of the MSE metric,
        assuming that the decrease is moved to one of the neighboring bins.

        Since each tile's contribution to the plan can, in general, span several
        LST bins and can change its area (exposure time) when its HA is
        adjusted, the assumptions of this algorithm are not valid in detail
        but it usually does a good job anyway.

        This algorithm has a stochastic component controlled by the `weights`
        parameter passed to our constructor, in order to avoid getting stuck
        in a local minimum.

        Returns
        -------
        tuple
            Tuple (idx, dha_sign) where idx is the LST bin index that should
            be decreased (by moving one of the tiles contributing to it) and
            dha_sign gives the sign +/-1 of the HA adjustment required.
        """
        # Rescale the available LST total time to the current plan total time.
        A = self.lst_hist * self.plan_hist.sum() / self.lst_hist.sum()
        # Calculate residuals in each LST bin.
        P = self.plan_hist
        res = P - A
        # Calculate the change from moving tiles between adjacent bins.
        nbins = len(self.lst_hist)
        adjacent = np.roll(np.arange(nbins), -1)
        dres = res[adjacent] - res
        # Cannot move tiles from an empty bin.
        empty = (P == 0)
        dres[empty & (dres < 0)] = 0.
        dres[empty[adjacent] & (dres > 0)] = 0.
        # Select the movements that reduce the MSE between A and P by
        # the largest amounts.
        order = np.argsort(np.abs(dres))[::-1][:len(self.cum_weights)]
        # Randomly select one of these moments, according to our weights.
        which = np.searchsorted(self.cum_weights, self.gen.uniform())
        idx = order[which]
        if dres[idx] == 0:
            raise RuntimeError('Cannot improve MSE.')
        elif dres[idx] > 0:
            idx = adjacent[idx]
            dha_sign = -1
        else:
            dha_sign = +1
        return idx, dha_sign

    def improve(self, frac=1., verbose=False):
        """Perform one iteration of improving the hour angle assignments.

        Each call will adjust the HA of a single tile with a magnitude |dHA|
        specified by the `frac` parameter.

        Parameters
        ----------
        frac : float
            Mean fraction of an LST bin to adjust the selected tile's HA by.
            Actual HA adjustments are randomly distributed around this mean
            to smooth out adjustments.
        verbose : bool
            Print verbose information about the algorithm progress.
        """
        # Randomly perturb the size of the HA adjustment.  This adds some
        # noise but also makes it possible to get out of dead ends.
        frac = np.minimum(2., self.gen.rayleigh(
            scale=np.sqrt(2 / np.pi) * frac))
        # Calculate the initial MSE.
        initial_MSE = self.eval_MSE(self.plan_hist)
        # Try a fast method first, then fall back to a slower method.
        for method in 'next_bin', 'any_bin':
            if method == 'next_bin':
                # Select which bin to move a tile from and in which direction.
                ibin, dha_sign = self.next_bin()
                # Find tiles using more time in this LST bin than in the
                # adjacent bin they would be moving away from.
                ibin_from = (ibin - dha_sign + self.nbins) % self.nbins
                sel = (self.plan_tiles[:, ibin] > 0) & (
                    self.plan_tiles[:, ibin] >= self.plan_tiles[:, ibin_from])
            else:
                # Try a 20% random subset of all tiles.
                self.nslow += 1
                sel = self.gen.permutation(self.ntiles) < (self.ntiles // 5)
                # Randomly select a direction to shift the next tile.
                dha_sign = +1 if self.gen.uniform() > 0.5 else -1

            # Do no move any tiles that are already at their |HA| limits.
            dha = 360. / self.nbins * frac * dha_sign
            veto = np.abs(self.ha + dha) >= self.max_abs_ha
            sel[veto] = False
            # Are there any tiles available to adjust?
            nsel = np.count_nonzero(sel)
            if nsel == 0:
                print('No tiles available for {0} method.'.format(method))
                continue
            # How many times have these tiles already been adjusted?
            nadj = self.num_adjustments[sel]
            if np.min(nadj) < np.max(nadj):
                # Do not adjust a tile that already has received the max
                # number of adjustments.  This has the effect of smoothing
                # the spatial distribution of HA adjustments.
                sel = sel & (self.num_adjustments < np.max(nadj))
                nsel = np.count_nonzero(sel)
            subset = np.where(sel)[0]
            # Calculate how the plan changes by moving each selected tile.
            scenario_os = self.get_plan(self.ha[subset] + dha, subset)
            scenario = scenario_os.reshape(
                nsel, self.nbins, self.oversampling).mean(axis=-1)
            new_MSE = np.zeros(nsel)
            for i, itile in enumerate(subset):
                # Calculate the (downsampled) plan when this tile is moved.
                new_plan_hist = (
                    self.plan_hist - self.plan_tiles[itile] + scenario[i])
                new_MSE[i]= self.eval_MSE(new_plan_hist)
            i = np.argmin(new_MSE)
            if new_MSE[i] > initial_MSE:
                # All candidate adjustments give a worse MSE.
                continue
            # Accept the tile that gives the smallest MSE.
            itile = subset[i]
            self.num_adjustments[itile] += 1
            if verbose:
                print('Moving tile {0} in bin {1} by dHA = {2:.3f}h'
                      .format(self.tid[itile], ibin, dha))
            # Update the plan.
            self.ha[itile] = self.ha[itile] + dha
            assert np.abs(self.ha[itile]) < self.max_abs_ha[itile]
            self.plan_tiles_os[itile] = scenario_os[i]
            # No need to try additional methods.
            break
        self.use_plan()
        if self.MSE_history[-1] >= initial_MSE:
            self.nstuck += 1
        self.nimprove += 1

    def plot(self, save=None):
        """Plot the current optimzation status.

        Requires that matplotlib is installed.

        Parameters
        ----------
        save : str or None
            Filename where the generated plot will be saved.
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax = ax.flatten()
        scale = self.scale_history[-1]
        ax[0].hist(self.lst_centers, bins=self.lst_edges,
                 weights=self.lst_hist * scale, histtype='stepfilled',
                 fc=(1,0,0,0.25), ec='r')
        ax[0].hist(self.lst_centers, bins=self.lst_edges,
                 weights=self.plan_hist, histtype='stepfilled',
                 fc=(0.7,0.7,1), ec='b')
        MSE_scale = self.plan_hist.sum() / self.lst_hist.sum()
        ax[0].hist(self.lst_centers, bins=self.lst_edges,
                 weights=self.lst_hist * MSE_scale, histtype='step',
                 fc=(1,0,0,0.25), ec='r', ls='--')
        # Superimpose the high-resolution plan.
        plan_os = self.plan_tiles_os.sum(axis=0)
        ax[0].plot(self.lst_centers_os, plan_os, 'b-', alpha=0.5, lw=1)

        nbins = len(self.lst_centers)
        idx, dha = self.next_bin()
        ax[0].axvline(self.lst_centers[idx], c='b', ls='-')
        ax[0].axvline(
            self.lst_centers[(idx + nbins + dha) % nbins], c='b', ls='--')

        ax[0].set_xlim(self.lst_edges[0], self.lst_edges[-1])
        ax[0].set_ylim(0, None)
        ax[0].set_xlabel('Local Sidereal Time [deg]')
        ax[0].set_ylabel('Hours / LST Bin')

        ax[1].plot(self.MSE_history, 'r.', ms=10, label='MSE')
        ax[1].legend(loc='lower left', numpoints=1)
        rhs = ax[1].twinx()
        rhs.plot(self.scale_history, 'kx', ms=5, label='Scale')
        rhs.legend(loc='upper right', numpoints=1)
        ax[1].set_xlim(0, len(self.MSE_history))
        ax[1].set_xlabel('Iterations')

        ax[2].hist(self.ha, bins=50)
        ax[2].set_xlabel('Tile Design Hour Angle [deg]')

        s = ax[3].scatter(self.ra, self.dec, c=self.ha - self.ha_initial,
                          s=12, lw=0, cmap='jet')
        ax[3].set_xlim(self.lst_edges[0] - 5, self.lst_edges[-1] + 5)
        ax[3].set_xticks([])
        ax[3].set_ylim(-20, 80)
        plt.colorbar(s, ax=ax[3], orientation='horizontal', pad=0.01)
        plt.tight_layout()
        if save:
            plt.savefig(save)
        plt.show()


if __name__ == '__main__':
    """This should eventually be made into a first-class script entry point.
    """
    import desisurvey.schedule
    scheduler = desisurvey.schedule.Scheduler()
    opt = Optimizer(scheduler, 'GRAY')
    for i in range(10):
        opt.improve(frac=0.25, verbose=True)
