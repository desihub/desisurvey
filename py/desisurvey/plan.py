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

    Parameters
    ----------
    hourangles : array
        1D array of floats with design hour angles in degrees to use for
        each tile.
    priorities : array
        1D array of non-negative floats with initial priorities to use for each
        tile. Priority normalization is arbitrary, but higher values correspond
        to higher priority observing.
    """
    tiles = astropy.table.Table(
        desimodel.io.load_tiles(onlydesi=True, extra=False))
    ntiles = len(tiles)

    hourangles = np.asarray(hourangles)
    if len(hourangles.shape) != 1 or len(hourangles) != ntiles:
        raise ValueError('Invalid hourangles parameter.')

    priorities = np.asarray(priorities)
    if len(priorities.shape) != 1 or len(priorities) != ntiles:
        raise ValueError('Invalid priorities parameter.')

    plan = astropy.table.Table()
    plan['tileid'] = tiles['TILEID']
    plan['ra'] = tiles['RA']
    plan['dec'] = tiles['DEC']
    plan['pass'] = tiles['PASS']

    plan['priority'] = priorities
    plan['hourangle'] = hourangles

    # Record day-number (relative to config.first_day) when a tile is first
    # covered by any tiles in passes that must be completed before
    # fiber assignment can be run.
    config = desisurvey.config.Configuration()
    num_nights = (config.last_day() - config.first_day()).days
    plan['covered'] = np.full(ntiles, num_nights, int)

    # Any passes with no fiber-assignment dependency are initially available.
    dependent = config.fiber_assignment_order.keys
    plan['available'] = np.zeros(ntiles, bool)
    for passnum in range(8):
        if 'P' + str(passnum) not in dependent:
            sel = plan['pass'] == passnum
            plan['covered'][sel] = -1
            plan['available'][sel] = True

    return plan


def update_available(plan, progress, night, fiber_assignment_delay):
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
    night : datetime.date
        Date when planning is being performed, used to interpret the
        next parameter.
    fiber_assignment_delay : int
        Number of nights delay between when a tile is covered and then
        subsequently made available for observing.

    Returns
    -------
    plan
        The input plan with the 'covered' and 'available' columns updated.
    """
    log = desiutil.log.get_logger()
    # Look up the nominal tile radius for determining overlaps.
    config = desisurvey.config.Configuration()
    tile_radius = config.tile_radius().to(u.deg).value
    # Look up the current night number.
    night_number = (night - config.first_day()).days
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
            covered = np.all(~overlapping | complete[over], axis=1)
            new_covered = covered & (plan['covered'][under] > night_number)
            if np.any(new_covered):
                new_tiles = plan['tileid'][under][new_covered]
                log.info(
                    'New tiles covered in pass {0}: {1}.'
                    .format(passnum, ','.join([str(tid) for tid in new_tiles])))
                # Record the night number when these tiles were first covered.
                new = under.copy()
                new[under] = new_covered
                plan['covered'][new] = night_number
            # Check if any tiles are newly available now.
            avail = plan['available'][under] | (
                plan['covered'][under] + fiber_assignment_delay <= night_number)
            new_avail = avail & ~(plan['available'][under])
            if np.any(new_avail):
                new_tiles = plan['tileid'][under][new_avail]
                log.info(
                    'New tiles available in pass {0}: {1}.'
                    .format(passnum, ','.join([str(tid) for tid in new_tiles])))
                # Record the night number when these tiles were first covered.
                plan['available'][under] |= new_avail
    return plan
