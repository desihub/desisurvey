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
    """
    """
    def __init__(self, p, program='GRAY', subset=None, nbins=90, init='info',
                 origin=-60, center=220, seed=123, oversampling=32,
                 weights=[5, 4, 3, 2, 1]):
        """
        """
        config = desisurvey.config.Configuration()
        self.gen = np.random.RandomState(seed)
        self.cum_weights = np.asarray(weights, float).cumsum()
        self.cum_weights /= self.cum_weights[-1]

        # Calculate the time available in bins of LST for this program.
        e = p.etable
        p_index = dict(DARK=1, GRAY=2, BRIGHT=3)[program]
        sel = (e['program'] == p_index).reshape(p.num_nights, p.num_times)
        # Zero out nights during monsoon and full moon.
        sel[p.calendar['monsoon']] = False
        sel[p.calendar['fullmoon']] = False
        # Accumulate times in hours over the full survey.
        dt = p.step_size.to(u.hour).value
        wgt = dt * np.ones((p.num_nights, p.num_times))
        # Weight nights for weather availability.
        lst = wrap(e['lst'][sel.flat], origin)
        wgt *= p.calendar['weather'][:, np.newaxis]
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
        p_tiles = p.tiles[p.tiles['program'] == p_index]
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
        """
        """
        exptime, subset = self.get_exptime(ha, subset)
        ##########################################################
        # Increase exposure times for debugging only.
        # exptime *= 5
        ##########################################################
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

    def MSE(self, plan_hist):
        # Rescale the available LST total time to the plan total time.
        scale = plan_hist.sum() / self.lst_hist.sum()
        return np.sum((plan_hist - scale * self.lst_hist) ** 2)

    def use_plan(self):
        """Use current self.plan_tiles_os.
        """
        # Downsample the high resolution plan to the LST bins.
        self.plan_tiles = self.plan_tiles_os.reshape(
            self.ntiles, self.nbins, self.oversampling).mean(axis=-1)
        # Sum over tiles at low resolution.
        self.plan_hist = self.plan_tiles.sum(axis=0)
        # Calculate the amount that the nominal LST budget needs to be rescaled
        # to accomodate this plan.
        scale = (self.plan_hist / self.lst_hist).max()
        self.scale_history.append(scale)
        self.MSE_history.append(self.MSE(self.plan_hist))

    def next_bin(self):
        """Select which LST bin to adjust next"""
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
        #idx = np.argmax(np.abs(dres))
        if dres[idx] == 0:
            raise RuntimeError('Cannot improve MSE.')
        elif dres[idx] > 0:
            idx = adjacent[idx]
            dha_sign = -1
        else:
            dha_sign = +1
        return idx, dha_sign

    def improve(self, frac=1., clip=50., verbose=False):
        """
        """
        # Select which bin to move a tile from and in which direction.
        ibin, dha_sign = self.next_bin()
        # Find tiles using more time in this LST bin than in the adjcent bin they
        # would be moving away from.
        ibin_from = (ibin - dha_sign + self.nbins) % self.nbins
        ##sel = (self.plan_tiles[:, ibin] > 0) & (
        ##    self.plan_tiles[:, ibin] >= self.plan_tiles[:, ibin_from])
        sel = (self.plan_tiles[:, ibin] > 0)
        nsel = np.count_nonzero(sel)
        if nsel == 0:
            print('No tiles to adjust in LST bin {0}.'.format(ibin))
            self.use_plan()
            return
        # How many times have these tiles already been adjusted?
        nadj = self.num_adjustments[sel]
        if np.min(nadj) < np.max(nadj):
            # Do not adjust a tile that already has received the max
            # number of adjustments.  This has the effect of smoothing
            # the spatial distribution of HA adjustments.
            sel = sel & (self.num_adjustments < np.max(nadj))
            nsel = np.count_nonzero(sel)
        subset = np.where(sel)[0]
        # Randomly perturb the size of the HA adjustment.  This adds some noise
        # but also makes it possible to get out of dead ends.
        frac = np.minimum(2., self.gen.rayleigh(scale=np.sqrt(2 / np.pi) * frac))
        dha = 360. / self.nbins * frac * dha_sign
        # Calculate how the plan would change by moving each selected tile.
        scenario_os = self.get_plan(self.ha[subset] + dha, subset)
        scenario = scenario_os.reshape(
            nsel, self.nbins, self.oversampling).mean(axis=-1)
        MSE = self.MSE(self.plan_hist)
        new_MSE = np.zeros(nsel)
        for i, itile in enumerate(subset):
            # Calculate the (downsampled) plan when this tile is moved.
            new_plan_hist = self.plan_hist - self.plan_tiles[itile] + scenario[i]
            new_MSE[i]= self.MSE(new_plan_hist)
        i = np.argmin(new_MSE)
        if new_MSE[i] > MSE:
            # Randomly accept with prob proportional to exp(-dMSE/MSE).
            accept_prob = np.exp((MSE - new_MSE[i]) / MSE)
            assert accept_prob > 0 and accept_prob < 1
            if self.gen.uniform() > accept_prob:
                # Reject this adjustment which makes MSE worse.
                self.use_plan()
                return
        itile = subset[i]
        self.num_adjustments[itile] += 1
        if verbose:
            print('Moving tile {0} in bin {1} by dHA = {2:.3f}h'
                  .format(self.tid[itile], ibin, dha))
        # Update the plan.
        self.ha[itile] = np.clip(self.ha[itile] + dha, -clip, +clip)
        self.plan_tiles_os[itile] = scenario_os[i]
        self.use_plan()

    def plot(self, save=None):
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
    import desisurvey.plan
    planner = desisurvey.plan.Planner()
    opt = Optimizer(planner, 'GRAY')
    for i in range(10):
        opt.improve(frac=0.25, verbose=True)
