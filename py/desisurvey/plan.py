"""Plan future DESI observations.
"""
from __future__ import print_function, division

import numpy as np

import astropy.table
import astropy.units as u

import desiutil.log

import desimodel.io

import desisurvey.utils


def create(hourangles, priorities):
    """Create a new plan for the start of the survey.
    """
    tiles = astropy.table.Table(
        desimodel.io.load_tiles(onlydesi=True, extra=False))

    plan = astropy.table.Table()
    plan['tileid'] = tiles['TILEID']
    plan['ra'] = tiles['RA']
    plan['dec'] = tiles['DEC']
    plan['pass'] = tiles['PASS']

    plan['priority'] = priorities
    plan['hourangle'] = hourangles
    # Assume that all first-layer tiles have targets assigned to fibers.
    plan['available'] = (
        (plan['pass'] == 0) | (plan['pass'] == 4) | (plan['pass'] == 5))
    return plan


def update_available(plan, progress):
    """Update list of available tiles.

    A tile becomes available when all overlapping tiles in the previous pass
    of the same program are complete. A newly available tile is ready for fiber
    assignment. Overlap is defined as center_separation < 2 * tile_radius.

    Parameters
    ----------
    plan : astropy.table.Table
        A table created and updated using functions in this package.
    progress : desisurvey.progress.Progress
        A record of observing progress so far.

    Returns
    -------
    plan
        The input plan with the 'available' column updated.
    """
    log = desiutil.log.get_logger()
    # Look up the nominal tile radius for determining overlaps.
    config = desisurvey.config.Configuration()
    tile_radius = config.tile_radius().to(u.deg).value
    # Find complete tiles.
    complete = (progress._table['status'] == 2)
    # Loop over passes.
    ra = plan['ra']
    dec = plan['dec']
    for passnum in range(8):
        under = (plan['pass'] == passnum)
        over = np.zeros_like(under)
        overattr = 'P'+str(passnum)
        if not hasattr(config.fiber_assignment_order, overattr):
            # These tiles should be available from the start of the survey.
            if not np.all(plan['available'][under]):
                raise RuntimeError('Expected all tiles available in pass {0}.'
                                   .format(passnum))
        else:
            overpasses = getattr(config.fiber_assignment_order, overattr)()
            for overpass in overpasses.split('+'):
                if not len(overpass) == 2 and overpass[0] == 'P':
                    raise RuntimeError(
                        'Invalid pass in fiber_assignment_order: {0}.'
                        .format(overpass))
                over |= (plan['pass'] == int(overpass[1]))
            overlapping = desisurvey.utils.separation_matrix(
                ra[under], dec[under], ra[over], dec[over], 2 * tile_radius)
            avail = np.all(~overlapping | complete[over], axis=1)
            new_avail = avail & ~plan['available'][under]
            if np.any(new_avail):
                new_tiles = plan['tileid'][under][new_avail]
                log.info(
                    'New tiles available in pass {0}: {1}.'
                    .format(passnum, ','.join([str(tid) for tid in new_tiles])))
                plan['available'][under] = avail
    return plan
