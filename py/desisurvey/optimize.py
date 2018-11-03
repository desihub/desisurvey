"""Optimize future DESI observations.
"""
from __future__ import print_function, division

import pkg_resources

import numpy as np
import scipy.special

import astropy.table
import astropy.coordinates
import astropy.units as u

import desiutil.log

import desisurvey.config


def wrap(angle, offset):
    """Wrap values in the range [0, 360] to [offset, offset+360].
    """
    return np.fmod(angle - offset + 360, 360) + offset


class Optimizer(object):
    """Initialize the hour angle assignments for specified tiles.

    Parameters
    ----------
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
    init : 'zero', 'flat' or 'array'
        Method for initializing tile hour angles: 'zero' sets all hour angles
        to zero, 'flat' matches the CDF of available LST to planned LST (without
        accounting for exposure time), 'array' initializes from the initial_ha
        argument.
    initial_ha : array or None
        Only used when init is 'array'. The subset arg must also be provided
        to specify which tile each HA applies to.
    stretch : float
        Factor to stretch all exposure times by.
    smoothing_radius : :class:`astropy.units.Quantity`
        Gaussian sigma for calculating smoothing weights with angular units.
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
    weights : array
        Array of relative weights to use when selecting which LST bin to
        optimize next.  Candidate bins are ordered by an estimated improvement.
        The length of the weights array determines how many candidates to
        consider, in decreasing order, and the weight values determines their
        relative weight. The next bin to optimize is then selected at random.
    """
    def __init__(self, program, subset=None, start=None, stop=None,
                 nbins=192, init='info', initial_ha=None, stretch=1.0,
                 smoothing_radius=10,
                 origin=-60, center=None, seed=123, weights=[5, 4, 3, 2, 1]):

        tiles = desisurvey.tiles.get_tiles()
        if program not in tiles.PROGRAMS:
            raise ValueError('Invalid program name: "{}".'.format(program))
        if not isinstance(smoothing_radius, u.Quantity):
            smoothing_radius = smoothing_radius * u.deg
        self.log = desiutil.log.get_logger()
        config = desisurvey.config.Configuration()
        self.gen = np.random.RandomState(seed)
        self.cum_weights = np.asarray(weights, float).cumsum()
        self.cum_weights /= self.cum_weights[-1]

        if start is None:
            start = config.first_day()
        else:
            start = desisurvey.utils.get_date(start)
        if stop is None:
            stop = config.last_day()
        else:
            stop = desisurvey.utils.get_date(stop)
        if start >= stop:
            raise ValueError('Expected start < stop.')

        # Calculate the time available in bins of LST for this program.
        ##e = sched.etable
        p_index = tiles.PROGRAM_INDEX[program]

        sel = (e['program'] == p_index).reshape(
            sched.num_nights, sched.num_times)
        # Zero out nights during monsoon and full moon.
        sel[sched.calendar['monsoon']] = False
        sel[sched.calendar['fullmoon']] = False
        # Zero out nights outside [start:stop].
        sel[:(start - config.first_day()).days] = False
        sel[(stop - config.first_day()).days:] = False
        # Accumulate times in hours over the full survey.
        dt = sched.step_size.to(u.hour).value
        wgt = dt * np.ones((sched.num_nights, sched.num_times))
        # Weight nights for weather availability.
        sel_flat = sel.flatten()
        lst = wrap(e['lst'][sel_flat], origin)
        wgt *= sched.calendar['weather'][:, np.newaxis]
        wgt = wgt[sel].flatten()
        if np.all(wgt == 0):
            raise RuntimeError('All weather weights are zero.')
        self.lst_hist, self.lst_edges = np.histogram(
            lst, bins=nbins, range=(origin, origin + 360), weights=wgt)
        self.lst_centers = 0.5 * (self.lst_edges[1:] + self.lst_edges[:-1])
        self.nbins = nbins
        self.origin = origin
        self.stretch = stretch
        self.lst_hist_sum = self.lst_hist.sum()
        self.binsize = 360. / self.nbins

        # Get nominal exposure time for this program,
        # converted to LST equivalent in degrees.
        texp_nom = getattr(config.nominal_exposure_time, program)()
        self.dlst_nom = 360 * texp_nom.to(u.day).value

        # Load the tiles for this program.
        p_tiles = sched.tiles[sched.tiles['program'] == p_index]
        # Restrict to a subset of tiles in this program, if requested.
        if subset is not None:
            subset = np.asarray(subset)
            idx = np.searchsorted(p_tiles['tileid'], subset)
            if not np.all(p_tiles['tileid'][idx] == subset):
                bad = set(subset) - set(p_tiles['tileid'])
                raise ValueError(
                    'Subset contains non-{0} tiles: {1}.'
                    .format(program, ','.join([str(n) for n in bad])))
            p_tiles = p_tiles[idx]

        self.ra = wrap(p_tiles['ra'].data, origin)
        self.dec = p_tiles['dec'].data
        self.tid = p_tiles['tileid'].data
        self.ntiles = len(self.ra)

        # Initialize an index array for the selected tiles.
        self.idx = np.searchsorted(sched.tiles['tileid'], self.tid)
        assert(np.all(sched.tiles['tileid'][self.idx] == self.tid))

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

        # Initialize smoothing weights.
        self.init_smoothing(smoothing_radius)

        self.log.info(
            '{0}: {1:.1f}h for {2} tiles (texp_nom {3:.1f}, stretch {4:.3f}).'
            .format(program, self.lst_hist_sum, self.ntiles, texp_nom, stretch))

        # Precompute coefficients for exposure time calculations.
        latitude = np.radians(config.location.latitude())
        self.A = np.sin(np.radians(self.dec)) * np.sin(latitude)
        self.B = np.cos(np.radians(self.dec)) * np.cos(latitude)

        # Initialize metric histories.
        self.scale_history = []
        self.loss_history = []
        self.RMSE_history = []

        # Initialize improve() counters.
        self.nslow = 0
        self.nimprove = 0
        self.nsmooth = 0

        # Calculate schedule plan with HA=0 asignments to establish
        # the smallest possible total exposure time.
        self.plan_tiles = self.get_plan(np.zeros(self.ntiles))
        self.use_plan(save_history=False)
        self.min_total_time = self.plan_hist.sum()

        # Initialize HA assignments for each tile.
        if init == 'zero':
            self.ha = np.zeros(self.ntiles)
        elif init == 'array':
            if subset is None:
                raise ValueError('Must specify subset when init is "array".')
            if len(initial_ha) != self.ntiles:
                raise ValueError('Array initial_ha has wrong length.')
            self.ha = np.asarray(initial_ha)
        elif init == 'flat':
            if center is None:
                centers = np.arange(-180, 180, 5)
            else:
                centers = [center]
            min_score = np.inf
            scores = []
            for center in centers:
                # Histogram available LST relative to the specified center.
                lst = wrap(e['lst'][sel_flat], center)
                hist, edges = np.histogram(
                    lst, bins=nbins, range=(center, center + 360), weights=wgt)
                # Calculate the CDF of available LST.
                lst_cdf = np.zeros_like(edges)
                lst_cdf[1:] = np.cumsum(hist)
                lst_cdf /= lst_cdf[-1]
                # Calculate the CDF of planned LST usage relative to the same
                # central LST, assuming HA=0. Instead of spreading each exposure
                # over multiple LST bins, add its entire HA=0 exposure time at
                # LST=RA.
                exptime, _ = self.get_exptime(ha=np.zeros(self.ntiles))
                tile_ra = wrap(p_tiles['ra'].data, center)
                sort_idx = np.argsort(tile_ra)
                tile_cdf = np.cumsum(exptime[sort_idx])
                tile_cdf /= tile_cdf[-1]
                # Use linear interpolation to find an LST for each tile that
                # matches the plan CDF to the available LST CDF.
                new_lst = np.interp(tile_cdf, lst_cdf, edges)
                # Calculate each tile's HA as the difference between its HA=0
                # LST and its new LST after CDF matching.
                ha = np.empty(self.ntiles)
                ha[sort_idx] = np.fmod(new_lst - tile_ra[sort_idx], 360)
                # Clip tiles to their airmass limits.
                ha = np.clip(ha, -self.max_abs_ha, +self.max_abs_ha)
                # Calculate the score for this HA assignment.
                self.plan_tiles = self.get_plan(ha)
                self.use_plan(save_history=False)
                scores.append(self.eval_score(self.plan_hist))
                # Keep track of the best score found so far.
                if scores[-1] < min_score:
                    self.ha = ha.copy()
                    min_score = scores[-1]
                    center_best = center
            self.log.info(
                'Center flat initial HA assignments at LST {:.0f} deg.'
                .format(center_best))
        else:
            raise ValueError('Invalid init option: {0}.'.format(init))

        # Check initial HA assignments against airmass limits.
        ha_clipped = np.clip(self.ha, -self.max_abs_ha, +self.max_abs_ha)
        if not np.all(self.ha == ha_clipped):
            delta = np.abs(self.ha - ha_clipped)
            idx = np.argmax(delta)
            self.log.warn('Clipped {0} HA assignments to airmass limits.'
                          .format(np.count_nonzero(delta)))
            self.log.warn('Max clip is {0:.1f} deg for tile {1}.'
                          .format(delta[idx], self.tid[idx]))
            self.ha = ha_clipped

        # Calculate schedule plan with initial HA asignments.
        self.plan_tiles = self.get_plan(self.ha)
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
            exposure times in degrees and subset is the input subset or
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
                   self.dust_factor[subset]) * self.stretch
        return exptime, subset

    def get_plan(self, ha, subset=None):
        """Calculate an LST usage plan for specified hour angle assignments.

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
            Array of shape (ntiles, nbins) giving the exposure time in hours
            that each tile needs in each LST bin. When a subset is specified,
            ntiles only indexes tiles in the subset.
        """
        exptime, subset = self.get_exptime(ha, subset)
        # Calculate LST windows for each tile's exposure.
        lst_mid = self.ra[subset] + ha
        lst_min = np.fmod(
            lst_mid - 0.5 * exptime - self.origin + 360, 360) + self.origin
        ##assert np.all(lst_min >= self.origin)
        ##assert np.all(lst_min < self.origin + 360)
        lst_max = np.fmod(
            lst_mid + 0.5 * exptime - self.origin + 360, 360) + self.origin
        ##assert np.all(lst_max >= self.origin)
        ##assert np.all(lst_max < self.origin + 360)
        # Calculate each exposure's overlap with each LST bin.
        lo = np.clip(
            self.lst_edges[1:] - lst_min[:, np.newaxis], 0, self.binsize)
        hi = np.clip(
            lst_max[:, np.newaxis] - self.lst_edges[:-1], 0, self.binsize)
        plan = lo + hi
        plan[lst_max > lst_min] -= self.binsize
        ##assert np.allclose(plan.sum(axis=1), exptime)
        # Convert from degrees to hours.
        return plan * 24. / 360.

    def eval_score(self, plan_hist):
        """Evaluate the score that improve() tries to minimize.

        Score is calculated as 100 * RMSE + 100 * loss.

        Parameters
        ----------
        plan_hist : array
            Histogram of planned LST usage for all tiles.

        Returns
        -------
        float
            Score value.
        """
        return (
            100 * self.eval_RMSE(plan_hist) +
            100 * self.eval_loss(plan_hist))

    def eval_RMSE(self, plan_hist):
        """Evaluate the mean-squared error metric for the specified plan.

        This is the metric that :meth:`optimize` attempts to improve. It
        measures the similarity of the available and planned LST histogram
        shapes, but not their normalizations.  A separate :meth:`eval_scale`
        metric measures how efficiently the plan uses the available LST.

        The histogram of available LST is rescaled to the same total time (area)
        before calculating residuals relative to the planned LST usage.

        RMSE values are scaled by (10K / ntiles) so the absolute metric value
        is more consistent when ntiles is varied.

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
        plan_sum = plan_hist.sum()
        scale = plan_sum / self.lst_hist_sum
        residuals = plan_hist - scale * self.lst_hist
        return np.sqrt(residuals.dot(residuals) * self.nbins) / plan_sum

    def eval_loss(self, plan_hist):
        """Evaluate relative loss of current plan relative to HA=0 plan.

        Calculated as (T-T0)/T0 where T is the total exposure time of the
        current plan and T0 is the total exposure time of an HA=0 plan.

        Parameters
        ----------
        plan_hist : array
            Histogram of planned LST usage for all tiles.

        Returns
        -------
        float
            Loss factor.
        """
        return plan_hist.sum() / self.min_total_time - 1.0

    def eval_scale(self, plan_hist):
        """Evaluate the efficiency of the specified plan.

        Calculates the minimum scale factor applied to the available
        LST histogram so that the planned LST usage is always <= the scaled
        available LST histogram.  This value can be intepreted as the fraction
        of the available time required to complete all tiles.

        This metric is only loosely correlated with the RMSE metric, so provides
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

    def use_plan(self, save_history=True):
        """Use the current plan and update internal arrays.

        Calculates the `plan_hist` arrays from the per-tile `plan_tiles` array,
        and records the current values of the RMSE and scale metrics.
        """
        self.plan_hist = self.plan_tiles.sum(axis=0)
        if not np.all(np.isfinite(self.plan_hist)):
            raise RuntimeError('Found invalid plan_tiles in use_plan().')
        if save_history:
            self.scale_history.append(self.eval_scale(self.plan_hist))
            self.loss_history.append(self.eval_loss(self.plan_hist))
            self.RMSE_history.append(self.eval_RMSE(self.plan_hist))

    def next_bin(self):
        """Select which LST bin to adjust next.

        The algorithm determines which bin of the planned LST usage histogram
        should be decreased in order to maximize the decrease of the score,
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
        A = self.lst_hist * self.plan_hist.sum() / self.lst_hist_sum
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
        # Select the movements that reduce the RMSE between A and P by
        # the largest amounts.
        order = np.argsort(np.abs(dres))[::-1][:len(self.cum_weights)]
        # Randomly select one of these moments, according to our weights.
        which = np.searchsorted(self.cum_weights, self.gen.uniform())
        idx = order[which]
        if dres[idx] == 0:
            raise RuntimeError('Cannot improve RMSE.')
        elif dres[idx] > 0:
            idx = adjacent[idx]
            dha_sign = -1
        else:
            dha_sign = +1
        return idx, dha_sign

    def improve(self, frac=1.):
        """Perform one iteration of improving the hour angle assignments.

        Each call will adjust the HA of a single tile with a magnitude \|dHA\|
        specified by the `frac` parameter.

        Parameters
        ----------
        frac : float
            Mean fraction of an LST bin to adjust the selected tile's HA by.
            Actual HA adjustments are randomly distributed around this mean
            to smooth out adjustments.
        """
        # Randomly perturb the size of the HA adjustment.  This adds some
        # noise but also makes it possible to get out of dead ends.
        frac = np.minimum(2., self.gen.rayleigh(
            scale=np.sqrt(2 / np.pi) * frac))
        # Calculate the initial score.
        initial_score = self.eval_score(self.plan_hist)
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
                self.log.debug('No tiles available for {0} method.'
                               .format(method))
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
            scenario = self.get_plan(self.ha[subset] + dha, subset)
            new_score = np.zeros(nsel)
            for i, itile in enumerate(subset):
                # Calculate the (downsampled) plan when this tile is moved.
                new_plan_hist = (
                    self.plan_hist - self.plan_tiles[itile] + scenario[i])
                new_score[i]= self.eval_score(new_plan_hist)
            i = np.argmin(new_score)
            if new_score[i] > initial_score:
                # All candidate adjustments give a worse score.
                continue
            # Accept the tile that gives the smallest score.
            itile = subset[i]
            self.num_adjustments[itile] += 1
            # Update the plan.
            self.ha[itile] = self.ha[itile] + dha
            assert np.abs(self.ha[itile]) < self.max_abs_ha[itile]
            self.plan_tiles[itile] = scenario[i]
            # No need to try additional methods.
            break
        self.use_plan()
        self.nimprove += 1

    def init_smoothing(self, radius):
        """Calculate and save smoothing weights.

        Weights for each pair of tiles [i,j] are calculated as::

            wgt[i,j] = exp(-0.5 * (sep[i,j]/radius) ** 2)

        where sep[i,j] is the separation angle between the tile centers.

        Parameters
        ----------
        radius : astropy.units.Quantity
            Gaussian sigma for calculating weights with angular units.
        """
        separations = desisurvey.utils.separation_matrix(
            self.ra, self.dec, self.ra, self.dec)
        ratio = separations / radius.to(u.deg).value
        self.smoothing_weights = np.exp(-0.5 * ratio ** 2)
        # Set self weight to zero.
        self_weights = np.diag(self.smoothing_weights)
        assert np.allclose(self_weights, 1.)
        self.smoothing_weights -= np.diag(self_weights)
        self.smoothing_sums = self.smoothing_weights.sum(axis=1)

    def smooth(self, alpha=0.1):
        """Smooth the current HA assignments.

        Each HA is replaced with a smoothed value::

            (1-alpha) * HA + alpha * HA[avg]

        where HA[avg] is the weighted average of all other tile HA assignments.
        """
        avg_ha = self.smoothing_weights.dot(self.ha) / self.smoothing_sums
        self.ha = (1 - alpha) * self.ha + alpha * avg_ha
        self.plan_tiles = self.get_plan(self.ha)
        self.use_plan()

    def plot(self, save=None, relative=True):
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
        RMSE_scale = self.plan_hist.sum() / self.lst_hist_sum
        ax[0].hist(self.lst_centers, bins=self.lst_edges,
                 weights=self.lst_hist * RMSE_scale, histtype='step',
                 fc=(1,0,0,0.25), ec='r', ls='--')

        nbins = len(self.lst_centers)
        idx, dha = self.next_bin()
        ax[0].axvline(self.lst_centers[idx], c='b', ls='-')
        ax[0].axvline(
            self.lst_centers[(idx + nbins + dha) % nbins], c='b', ls='--')

        ax[0].set_xlim(self.lst_edges[0], self.lst_edges[-1])
        ax[0].set_ylim(0, None)
        ax[0].set_xlabel('Local Sidereal Time [deg]')
        ax[0].set_ylabel('Hours / LST Bin')

        ax[1].plot(self.RMSE_history, 'r.', ms=10, label='RMSE')
        ax[1].legend(loc='lower left', numpoints=1)
        rhs = ax[1].twinx()
        rhs.plot(self.scale_history, 'kx', ms=5, label='Scale')
        #rhs.plot(self.loss_history, 'g+', ms=5, label='Loss')
        rhs.legend(loc='upper right', numpoints=1)
        ax[1].set_xlim(-0.1, len(self.RMSE_history) - 0.9)
        ax[1].set_xlabel('Iterations')

        ax[2].hist(self.ha, bins=50, histtype='stepfilled')
        ax[2].set_xlabel('Tile Design Hour Angle [deg]')

        if relative:
            c = self.ha - self.ha_initial
            clabel = 'Tile Hour Angle Adjustments [deg]'
        else:
            c = self.ha
            clabel = 'Tile Hour Angle [deg]'
        s = ax[3].scatter(self.ra, self.dec, c=c, s=12, lw=0, cmap='jet')
        ax[3].set_xlim(self.lst_edges[0] - 5, self.lst_edges[-1] + 5)
        ax[3].set_xticks([])
        ax[3].set_xlabel(clabel)
        ax[3].set_ylim(-20, 80)
        ax[3].set_yticks([])
        cbar = plt.colorbar(s, ax=ax[3], orientation='vertical',
                            fraction=0.05, pad=0.01, format='%.1f')
        cbar.ax.tick_params(labelsize=9)
        plt.tight_layout()
        if save:
            plt.savefig(save)
        else:
            plt.show()


if __name__ == '__main__':
    """This should eventually be made into a first-class script entry point.
    """
    opt = Optimizer(scheduler, 'GRAY')
    for i in range(10):
        opt.improve(frac=0.25, verbose=True)
