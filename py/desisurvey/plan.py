"""Plan future DESI observations.
"""
from __future__ import print_function, division

import numpy as np

import astropy.table
import astropy.units as u

import desiutil.log

import desimodel.io

import desisurvey.config
import desisurvey.optimize
import desisurvey.progress
import desisurvey.schedule


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


def update_active(plan, progress):
    """Identify the active tiles given the survey progress so far.
    """
    log = desiutil.log.get_logger()
    progress = desisurvey.progress.Progress(restore=progress)
    # Match plan tiles to the progress table.
    idx = np.searchsorted(plan['tileid'], progress._table['tileid'])
    assert np.all(progress._table['tileid'][idx] == plan['tileid'])
    incomplete = progress._table['status'][idx] < 2
    # Loop over fiber-assignment groups.
    active = np.zeros_like(incomplete)
    for group in np.unique(plan['group']):
        sel = plan['group'] == group
        # Loop over priorities in descending order for this group.
        for priority in np.unique(plan['priority'][sel])[::-1]:
            # Identify tiles that still need observing in this (group, priority).
            psel = sel & (plan['priority'] == priority) & incomplete
            if np.count_nonzero(psel) > 0:
                log.info('Adding {0} active tiles from group {1} priority {2}'
                         .format(np.count_nonzero(psel), group, priority))
                active[psel] = True
                break
    plan['active'] = active
    return plan


def get_optimizer(plan, scheduler, program, start, stop, init):
    """Return an optimizer for all tiles in the specified program.
    """
    program_passes = dict(DARK=(0, 3), GRAY=(4, 4), BRIGHT=(5, 7))
    passes = program_passes[program]
    passnum = plan['pass']
    sel = plan['active'] & (passnum >= passes[0]) & (passnum <= passes[1])
    print('Optimizing {0} active {1} tiles.'
          .format(np.count_nonzero(sel), program))
    popt = desisurvey.optimize.Optimizer(
        scheduler, program, plan['tileid'][sel], start, stop, init=init)
    assert np.all(popt.tid == plan['tileid'][sel])
    return popt


def update(plan, progress, scheduler, start, stop, init='info',
           nopts=(5000,), fracs=(0.5,), plot_basename=None):
    """Update the hour angle assignments in a plan based on survey progress.

    Returns None if all tiles have been observed.
    """
    log = desiutil.log.get_logger()
    log.info('Updating plan for {0} to {1}'.format(start, stop))
    if len(nopts) != len(fracs):
        raise ValueError('Must have same lengths for nopts, fracs.')
    # Update the active-tile assignments.
    plan = update_active(plan, progress)
    if np.count_nonzero(plan['active']) == 0:
        return None
    # Specify HA assignments for the active tiles in each program.
    for program in 'DARK', 'GRAY', 'BRIGHT':
        popt = get_optimizer(plan, scheduler, program, start, stop, init)
        for nopt, frac in zip(nopts, fracs):
            for j in range(nopt):
                popt.improve(frac)
        if plot_basename is not None:
            popt.plot(save='{0}_{1}.png'.format(plot_basename, program))
        plan['hourangle'][popt.idx] = popt.ha
    return plan


def update_required(plan, progress):
    """Test if all active tiles in any group are complete.
    """
    answer = False
    log = desiutil.log.get_logger()
    # Match plan tiles to the progress table.
    idx = np.searchsorted(plan['tileid'], progress._table['tileid'])
    assert np.all(progress._table['tileid'][idx] == plan['tileid'])
    incomplete = progress._table['status'][idx] < 2
    # Loop over fiber-assignment groups.
    for group in np.unique(plan['group']):
        # Find active tiles in this group.
        sel = (plan['group'] == group) & plan['active']
        if np.count_nonzero(sel) == 0:
            log.info('Group {0} is complete.'.format(group))
            continue
        priority = np.unique(plan['priority'][sel])
        if len(priority) != 1:
            raise RuntimeError('Found mixed priorities {0} for group {1}'
                               .format(priority, group))
        nremaining = np.count_nonzero(sel & incomplete)
        log.info('Group {0} Priority {1} has {2:4d} tile(s) remaining.'
                 .format(group, priority[0], nremaining))
        if nremaining == 0:
            answer = True
    return answer


if __name__ == '__main__':
    """Regression test for loading a plan from a YAML file"""
    p = Planner()
