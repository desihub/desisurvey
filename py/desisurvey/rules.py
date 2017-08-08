"""Manage and apply observing priorities using rules.
"""
from __future__ import print_function, division

import os
import re

import yaml

import numpy as np

import astropy.table
import astropy.utils.data

import desimodel.io


class Rules(object):
    """Load rules from the specified file.

    Read tile group definitions and observing rules from the specified
    YAML file.
    """
    def __init__(self, file_name='rules.yaml', restore=None):
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
                    if trigger != 'START':
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
