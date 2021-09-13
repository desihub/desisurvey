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

try:
    from astropy.utils.data import get_pkg_data_path
except ImportError:
    # Astropy < 4.3
    from astropy.utils.data import _find_pkg_data_path as get_pkg_data_path

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
        self.min_snr2_fraction = config.min_snr2_fraction()
        self.finish_started_priority = config.finish_started_priority()
        self.ignore_completed_priority = config.ignore_completed_priority()
        self.boost_priority_by_passnum = config.boost_priority_by_passnum()
        self.adjacency_priority = config.adjacency_priority()

        tiles = desisurvey.tiles.get_tiles()
        NGC = (tiles.tileRA > 75.0) & (tiles.tileRA < 300.0)
        SGC = ~NGC

        # Initialize regexp for parsing "GROUP_NAME(PROGRAM)"
        parser = re.compile('([^\(]+)\(([^\(]+)\)$')

        # Get the full path of the YAML file to read.
        if os.path.isabs(file_name):
            full_path = file_name
        elif os.path.exists(config.get_path(file_name)):
            full_path = config.get_path(file_name)
        else:
            # Locate the config file in our package data/ directory.
            full_path = get_pkg_data_path(os.path.join('data', file_name))

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

            # Parse required "passes" attribute.
            programs = node.get('programs')
            if programs is None:
                raise RuntimeError(
                    'Missing required programs for {0}.'.format(group_name))
            programs = [p.strip() for p in str(programs).split(',')]
            for p in tiles.programs:
                if p not in programs:
                    group_sel[tiles.tileprogram == p] = False

            # Create GROUP(PROGRAM) subgroup combinations.
            final_group_sel = np.zeros(tiles.ntiles, bool)
            for p in programs:
                program_name = '{0}({1})'.format(group_name, p)
                group_names.append(program_name)
                group_id = len(group_names)
                program_sel = group_sel & (tiles.tileprogram == p)
                # Remove any tiles in this pass that have already been assigned
                # to a previously defined subgroup.
                program_sel[program_sel] &= ~(group_ids[program_sel] != 0)
                final_group_sel |= program_sel
                group_ids[program_sel] = group_id
                group_rules[program_name] = {'START': 0.0}
                group_max_orphans[program_name] = max_orphans

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
            orphans = (group_ids == 0) & (tiles.in_desi)
            programs = ','.join([str(s) for s in np.unique(tiles.tileprogram[orphans])])
            self.log.warning(
                '{0} tiles in passes {1} not assigned to any group.  These '
                'tiles will be given zero priority. '
                .format(np.count_nonzero(orphans), programs))

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

    def apply(self, donefrac):
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

        tiles = desisurvey.tiles.get_tiles()
        nogray = tiles.nogray

        # First pass through groups to check trigger conditions.
        triggered = {'START': True}
        notilescoveredrules = []
        for i, name in enumerate(self.group_names):
            gid = i+1
            group_sel = self.group_ids == gid
            if not np.any(group_sel):
                notilescoveredrules.append(name)
            ngroup = np.count_nonzero(group_sel)
            completed = donefrac >= self.min_snr2_fraction
            ndone = np.count_nonzero(completed[group_sel])
            max_orphans = self.group_max_orphans[name]
            triggered[name] = (ndone + max_orphans >= ngroup)
        notilescoveredrules = [x for x in notilescoveredrules
                               if not nogray or '(GRAY)' not in name]
        if len(notilescoveredrules) > 0:
            self.log.debug('No tiles covered by rules {}'.format(
                ' '.join(notilescoveredrules)))

        # Second pass through groups to apply rules.
        priorities = np.zeros_like(self.dec_priority)
        for gid, name in zip(np.unique(self.group_ids), self.group_names):
            priority = 0
            for condition, value in self.group_rules[name].items():
                if triggered[condition]:
                    priority = max(priority, value)
            sel = self.group_ids == gid
            priorities[sel] = priority * self.dec_priority[sel]
        priorities *= (1 + self.boost_priority_by_passnum)**tiles.tilepass
        priorities *= (1 + self.finish_started_priority*(donefrac > 0))
        neighborfrac = completed_neighbor_fraction(
            tiles, donefrac >= self.min_snr2_fraction)
        priorities *= (1 + self.adjacency_priority*neighborfrac)
        if self.ignore_completed_priority > 0:
            priorities *= np.where(donefrac >= self.min_snr2_fraction,
                                   self.ignore_completed_priority, 1)
        priorities *= tiles.priority_boostfac
        return priorities


def completed_neighbor_fraction(tiles, completed):
    cache = getattr(completed_neighbor_fraction, 'neighborcache', None)
    if cache is None:
        n1 = np.repeat(np.arange(len(tiles.neighbors)),
                       [len(x) for x in tiles.neighbors])
        n2 = np.concatenate([x for x in tiles.neighbors if len(x) > 0])
        completed_neighbor_fraction.neighborcache = (n1, n2)
    else:
        n1, n2 = cache
    if len(completed) != tiles.ntiles:
        raise ValueError('shape mismatch between completed and tiles!')
    res = np.bincount(n1, weights=completed[n2],
                      minlength=len(tiles.neighbors))
    nneighbor = np.bincount(n1, minlength=len(tiles.neighbors))
    res = res / (nneighbor + (nneighbor == 0))
    return res
