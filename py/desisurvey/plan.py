"""Plan future DESI observations.
"""
from __future__ import print_function, division

import os
import re

import yaml

import numpy as np

import astropy.table
import astropy.utils.data
import astropy.units as u

import desiutil.log

import desimodel.io

import desisurvey.config
import desisurvey.optimize
import desisurvey.progress
import desisurvey.schedule


class Planner(object):
    """Load an observing plan from the specified file.

    Read tile group definitions and observing rules from the specified
    YAML file.
    """
    def __init__(self, file_name='plan.yaml', restore=None):
        # Load the table of tiles in the DESI footprint.
        tiles = astropy.table.Table(
            desimodel.io.load_tiles(onlydesi=True, extra=False))
        num_tiles = len(tiles)
        passnum = tiles['PASS']
        dec = tiles['DEC']
        NGC = (tiles['RA'] > 75.0) & (tiles['RA'] < 300.0)
        SGC = ~NGC

        # Initialize regexp for parsing "GROUP_NAME(PASS)"
        parser = re.compile('([^\(]+)\(([0-7])\)$')

        # Get the full path of the YAML file to read.
        if os.path.isabs(file_name):
            full_path = file_name
        else:
            # Locate the config file in our package data/ directory.
            full_path = astropy.utils.data._find_pkg_data_path(
                os.path.join('data', file_name))

        # Read the YAML file into memory.
        with open(full_path) as f:
            config = yaml.safe_load(f)

        group_names = []
        group_ids = np.zeros(num_tiles, int)
        group_rules = {}

        for group_name in config:
            group_sel = np.ones(num_tiles, bool)
            node = config[group_name]

            # Parse optional geographical attribute.
            cap = node.get('cap')
            if cap == 'N':
                group_sel[~SGC] = False
            elif cap == 'S':
                group_sel[~NGC] = False
            dec_min = node.get('dec_min')
            if dec_min is not None:
                group_sel[dec < float(dec_min)] = False
            dec_max = node.get('dec_max')
            if dec_max is not None:
                group_sel[dec >= float(dec_max)] = False

            # Parse required "passes" attribute.
            passes = node.get('passes')
            if passes is None:
                raise RuntimeError(
                    'Missing required passes for {0}.'.format(group_name))
            passes = [int(p) for p in str(passes).split(',')]

            # Create GROUP(PASS) combinations.
            for p in passes:
                pass_name = '{0}({1:d})'.format(group_name, p)
                group_names.append(pass_name)
                group_id = len(group_names)
                pass_sel = group_sel & (passnum == p)
                if np.any(group_ids[pass_sel] != 0):
                    other_id = np.unique(group_ids[pass_sel])[-1]
                    raise RuntimeError(
                        'Some tiles assigned to multiple groups: {0}, {1}.'
                        .format(group_names[other_id - 1], pass_name))
                group_ids[pass_sel] = group_id
                group_rules[pass_name] = {}

            # Parse rules for this group.
            rules = node.get('rules')
            if rules is None:
                raise RuntimeError(
                    'Missing required rules for {0}.'.format(group_name))
            for target in rules:
                target_parsed = parser.match(target)
                if not target_parsed or target_parsed.groups(1) == group_name:
                    raise RuntimeError('Invalid rule target: {0}'.format(target))
                for trigger in rules[target]:
                    if trigger == 'START':
                        continue
                    trigger_parsed = parser.match(trigger)
                    if not trigger_parsed:
                        raise RuntimeError(
                            'Invalid rule trigger: {0}.'.format(trigger))
                    try:
                        new_weight = float(rules[target][trigger])
                    except ValueError:
                        raise RuntimeError(
                            'Invalid new weight for trigger {0}: {1}.'
                            .format(trigger, rules[target][trigger]))
                group_rules[target][trigger] = new_weight

        # Check that all groups have at least one rule.
        for pass_name in group_rules:
            if group_rules[pass_name] == {}:
                raise RuntimeError('Missing rules for {0}.'.format(pass_name))

        # Check that all tiles are assigned to exactly one group.
        if np.any(group_ids == 0):
            orphans = (group_ids == 0)
            passes = ','.join([str(s) for s in np.unique(passnum[orphans])])
            raise RuntimeError(
                '{0} tiles in passes {1} not assigned to any group.'
                .format(np.count_nonzero(orphans), passes))

        self.group_names = group_names
        self.group_ids = group_ids
        self.group_rules = group_rules

        if restore is not None:
            self.plan = astropy.table.Table.read(restore)
        else:
            # Create a new plan.
            self.plan = astropy.table.Table()
            self.plan['tileid'] = tiles['TILEID']
            self.plan['ra'] = tiles['RA']
            self.plan['dec'] = tiles['DEC']
            self.plan['pass'] = tiles['PASS']
            self.plan['group'] = self.group_ids
            self.plan['weight'] = np.zeros(len(tiles))
            ##self.plan['priority'] = priority
            ##self.plan['active'] = np.zeros(len(tiles), bool)
            self.plan['hourangle'] = np.zeros(len(tiles))


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

    plan = astropy.table.Table()
    plan['tileid'] = tiles['TILEID']
    plan['ra'] = tiles['RA']
    plan['dec'] = tiles['DEC']
    plan['pass'] = tiles['PASS']
    plan['group'] = group
    plan['priority'] = priority
    plan['active'] = np.zeros(len(tiles), bool)
    plan['hourangle'] = np.zeros(len(tiles))
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
