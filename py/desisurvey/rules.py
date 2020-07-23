"""Manage and apply tile observing priorities using rules.
"""
from __future__ import print_function, division

import os
import re
import collections

import yaml

import numpy as np

import astropy.table
import astropy.utils.data
import astropy.units as u

import desimodel.io
import desiutil.log

import desisurvey.config
import desisurvey.utils
import desisurvey.tiles


# Loads a YAML file with dictionary key ordering preserved.
# https://stackoverflow.com/questions/5121931/
# in-python-how-can-you-load-yaml-mappings-as-ordereddicts/21048064#21048064
def _ordered_load(stream, Loader=yaml.Loader,
                  object_pairs_hook=collections.OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


class Rules(object):
    """Load rules from the specified file.

    Read tile group definitions and observing rules from the specified
    YAML file.

    Parameters
    ----------
    file_name : str
        Name of YAML file containing the rules to use. A relative path refers
        to our configured output path.
    """
    def __init__(self, file_name='rules.yaml'):
        self.log = desiutil.log.get_logger()
        config = desisurvey.config.Configuration()
        tile_radius = config.tile_radius().to(u.deg).value

        tiles = desisurvey.tiles.get_tiles()
        NGC = (tiles.tileRA > 75.0) & (tiles.tileRA < 300.0)
        SGC = ~NGC

        # Initialize regexp for parsing "GROUP_NAME(PASS)"
        parser = re.compile(r'([^\(]+)\(([0-99]+)\)$')

        # Get the full path of the YAML file to read.
        if os.path.isabs(file_name):
            full_path = file_name
        else:
            # Locate the config file in our package data/ directory.
            full_path = astropy.utils.data._find_pkg_data_path(
                os.path.join('data', file_name))

        # Read the YAML file into memory.
        with open(full_path) as f:
            rules_dict = _ordered_load(f, yaml.SafeLoader)

        group_names = []
        group_ids = np.zeros(tiles.ntiles, int)
        dec_priority = np.ones(tiles.ntiles, float)
        group_rules = {}
        group_max_orphans = {}

        for group_name in rules_dict:
            group_sel = np.ones(tiles.ntiles, bool)
            node = rules_dict[group_name]

            # Parse optional geographical attribute.
            cap = node.get('cap')
            if cap == 'N':
                group_sel[SGC] = False
            elif cap == 'S':
                group_sel[NGC] = False
            dec_min = node.get('dec_min')
            if dec_min is not None:
                group_sel[tiles.tileDEC < float(dec_min)] = False
            dec_max = node.get('dec_max')
            if dec_max is not None:
                group_sel[tiles.tileDEC >= float(dec_max)] = False
            max_orphans = node.get('max_orphans') or 0
            covers = node.get('covers')
            if covers is not None:
                # Build the set of all tiles that must be covered.
                under = np.zeros(tiles.ntiles, bool)
                for name in covers.split('+'):
                    try:
                        group_id = group_names.index(name) + 1
                    except ValueError:
                        raise RuntimeError('Invalid covers target: {0}.'
                                           .format(name))
                    under |= (group_ids == group_id)

            # Parse required "passes" attribute.
            passes = node.get('passes')
            if passes is None:
                raise RuntimeError(
                    'Missing required passes for {0}.'.format(group_name))
            passes = [int(p) for p in str(passes).split(',')]
            for p in tiles.passes:
                if p not in passes:
                    group_sel[tiles.passnum == p] = False

            # Create GROUP(PASS) subgroup combinations.
            final_group_sel = np.zeros(tiles.ntiles, bool)
            for p in passes:
                pass_name = '{0}({1:d})'.format(group_name, p)
                group_names.append(pass_name)
                group_id = len(group_names)
                pass_sel = group_sel & (tiles.passnum == p)
                if covers is not None:
                    # Limit to tiles covering at least one tile in "under".
                    matrix = desisurvey.utils.separation_matrix(
                        tiles.tileRA[pass_sel], tiles.tileDEC[pass_sel], tiles.tileRA[under], tiles.tileDEC[under],
                        2 * tile_radius)
                    overlapping = np.any(matrix, axis=1)
                    pass_sel[pass_sel] &= overlapping
                # Remove any tiles in this pass that have already been assigned
                # to a previously defined subgroup.
                pass_sel[pass_sel] &= ~(group_ids[pass_sel] != 0)
                final_group_sel |= pass_sel
                group_ids[pass_sel] = group_id
                group_rules[pass_name] = {'START': 0.0}
                group_max_orphans[pass_name] = max_orphans

            # Some tiles may be dropped by covering requirements in this
            # or previous groups.
            assert not np.any(~group_sel & final_group_sel)
            group_sel = final_group_sel

            # Calculate priority multipliers to implement optional DEC ordering.
            dec_order = node.get('dec_order')
            if dec_order is not None and np.any(group_sel):
                dec_group = tiles.tileDEC[group_sel]
                lo, hi = np.min(dec_group), np.max(dec_group)

                slope = float(dec_order)
                if lo == hi:
                    dec_priority[group_sel] = 1.0
                elif slope > 0:
                    dec_priority[group_sel] = (
                        1 + slope * (hi-dec_group) / (hi-lo))
                else:
                    dec_priority[group_sel] = (
                        1 - slope * (dec_group-lo) / (hi-lo))
            else:
                assert np.all(dec_priority[group_sel] == 1)

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
                    assert target in group_rules
                    group_rules[target][trigger] = new_weight

        # Check that all tiles are assigned to exactly one group.
        if np.any(group_ids == 0):
            orphans = (group_ids == 0)
            passes = ','.join([str(s) for s in np.unique(tiles.passnum[orphans])])
            raise RuntimeError(
                '{0} tiles in passes {1} not assigned to any group.'
                .format(np.count_nonzero(orphans), passes))

        # Check that all rule triggers are valid subgroup names.
        for name in group_names:
            for target in group_rules[name]:
                if target == 'START':
                    continue
                if target not in group_names:
                    raise RuntimeError(
                        'Invalid target {0} in {1} rule.'.format(target, name))

        self.group_names = group_names
        self.group_ids = group_ids
        self.group_rules = group_rules
        self.dec_priority = dec_priority
        self.group_max_orphans = group_max_orphans

    def apply(self, completed):
        """Apply rules to determine tile priorites based on those completed so far.

        Parameters
        ----------
        completed : array
            Boolean array of per-tile completion status.

        Returns
        -------
        array
            Array of per-tile observing priorities.
        """
        # First pass through groups to check trigger conditions.
        triggered = {'START': True}
        config = desisurvey.config.Configuration()
        for i, name in enumerate(self.group_names):
            gid = i+1
            group_sel = self.group_ids == gid
            if not np.any(group_sel) and not config.commissioning:
                self.log.error('No tiles covered by rule {}'.format(name))
            ngroup = np.count_nonzero(group_sel)
            ndone = np.count_nonzero(completed[group_sel])
            max_orphans = self.group_max_orphans[name]
            triggered[name] = (ndone + max_orphans >= ngroup)
        # Second pass through groups to apply rules.
        priorities = np.zeros_like(self.dec_priority)
        for gid, name in zip(np.unique(self.group_ids), self.group_names):
            priority = 0
            for condition, value in self.group_rules[name].items():
                if triggered[condition]:
                    priority = max(priority, value)
            sel = self.group_ids == gid
            priorities[sel] = priority * self.dec_priority[sel]
        return priorities
