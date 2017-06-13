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


def baseline(tiles):
    """Tabulate the group and priority assignments of the baseline plan.
    """
    config = desisurvey.config.Configuration().full_depth_field

    passnum = tiles['PASS']
    dark = (passnum < 4)
    gray = (passnum == 4)
    bright = (passnum > 4)

    # Specify the fiber-assignment sequencing of each pass.
    fa1 = (passnum == 0) | (passnum == 4) | (passnum == 5)
    fa2 = (passnum == 1) | (passnum == 6)
    fa3 = (passnum == 2) | (passnum == 3) | (passnum == 7)
    fa_priority = fa1 * 3 + fa2 * 2 + fa3 * 1

    # Specify the sky regions with independent sequencing.
    NGC = (tiles['RA'] > 75.0) & (tiles['RA'] < 300.0)
    SGC = ~NGC
    dec = tiles['DEC']
    dec_min = np.full(len(dec), config.min_declination().to(u.deg).value)
    dec_max = np.full(len(dec), config.max_declination().to(u.deg).value)
    pad = config.first_pass_padding().to(u.deg).value
    dec_min[fa1] -= pad
    dec_max[fa1] += pad
    DN = NGC & (dec >= dec_min) & (dec <= dec_max)
    N1 = NGC & (dec < dec_min)
    N2 = NGC & (dec > dec_max)
    S1 = SGC & (dec < 5)
    S2 = SGC & (dec >= 5)

    # Combine pass and region priorities.
    group = ((dark & NGC) * 1 + (dark & SGC) * 2 +
             (gray & NGC) * 3 + (gray & SGC) * 4 +
             (bright & NGC) * 5 + (bright & SGC) * 6)
    priority = ((DN | S1) * (6 + fa_priority) +
                (N1 | S2) * (3 + fa_priority) +
                N2 * fa_priority)

    return group, priority


def create(planner=baseline):
    """Create a new plan for the start of the survey.
    """
    tiles = astropy.table.Table(
        desimodel.io.load_tiles(onlydesi=True, extra=False))

    group, priority = planner(tiles)

    table = astropy.table.Table()
    table['tileid'] = tiles['TILEID']
    table['ra'] = tiles['RA']
    table['dec'] = tiles['DEC']
    table['pass'] = tiles['PASS']
    table['group'] = group
    table['priority'] = priority
    table['active'] = np.zeros(len(tiles), bool)
    table['hourangle'] = np.zeros(len(tiles))
    return table


def update(plan, progress=None, plot_basename=None):
    """Identify the active tiles and update their hour angle assignments.
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
    # Calculate active-tile HA assignments separately each program.
    passnum = plan['pass']
    program_passes = ((0, 3), (4, 4), (5, 7))
    scheduler = desisurvey.schedule.Scheduler()
    for i, program in enumerate(('DARK', 'GRAY', 'BRIGHT')):
        passes = program_passes[i]
        sel = active & (passnum >= passes[0]) & (passnum <= passes[1])
        log.info('Optimizing {0} active {1} tiles.'
                 .format(np.count_nonzero(sel), program))
        popt = desisurvey.optimize.Optimizer(
            scheduler, program, plan['tileid'][sel], init='info')
        assert np.all(popt.tid == plan['tileid'][sel])
        ##return popt
        for frac in (0.5,):
            for j in range(5000):
                popt.improve(frac)
        print('{0}: {1}, {2}, {3}'.format(
                program, popt.nimprove, popt.nslow, popt.nstuck))
        if plot_basename is not None:
            popt.plot(save='{0}_{1}.png'.format(plot_basename, program))
        plan['hourangle'][sel] = popt.ha

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
    """This should eventually be made into a first-class script entry point.

    This takes a few minutes to run and writes four files to $DESISURVEY:
    - initial_plan.fits
    - initial_plan_DARK.png
    - initial_plan_GRAY.png
    - initial_plan_BRIGHT.png
    """
    config = desisurvey.config.Configuration()
    plan = update(create(), plot_basename=config.get_path('initial_plan'))
    plan.write(config.get_path('initial_plan.fits'), overwrite=True)
