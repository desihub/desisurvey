"""Plan future DESI observations.
"""
from __future__ import print_function, division

import numpy as np

import astropy.table

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


def update_available(plan, progress, tile_radius=1.62):
    """Update list of available tiles.

    A tile becomes available when all overlapping tiles in the previous pass
    of the same program are complete. A newly available tile is ready for fiber
    assignment.

    Overlap is defined as center_separation < 2 * tile_radius, using a default
    tile radius based on the discussion at
    https://github.com/desihub/desimodel/pull/37#issuecomment-270788581

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
    # Find complete tiles.
    complete = (progress._table['status'] == 2)
    # Loop over passes.
    for passnum in range(8):
        sel = (plan['pass'] == passnum)
        ra = plan['ra'][sel]
        dec = plan['dec'][sel]
        if passnum in (0, 4, 5):
            # These tiles should be available from the start of the survey.
            if not np.all(plan['available'][sel]):
                raise RuntimeError('Expected all tiles available in pass {0}.'
                                   .format(passnum))
        else:
            # Check for tiles fully covered by the previous pass.
            overlapping = (desisurvey.utils.separation_matrix(
                ra, dec, ra_prev, dec_prev) < 2 * tile_radius)
            avail = np.all(~overlapping | complete[sel_prev], axis=1)
            new_avail = avail & ~plan['available'][sel]
            if np.any(new_avail):
                new_tiles = plan['tileid'][sel][new_avail]
                log.info(
                    'New tiles available in pass {0}: {1}.'
                    .format(passnum, ','.join([str(tid) for tid in new_tiles])))
                plan['available'][sel] = avail
        ra_prev = ra
        dec_prev = dec
        sel_prev = sel
    return plan
