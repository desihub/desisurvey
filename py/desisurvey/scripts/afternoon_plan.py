import desisurvey
import desisurvey.tiles
import desisurvey.rules
import desisurvey.plan
import desisurvey.scheduler
import desiutil.log
import re
import os
import shutil
import subprocess
from desisurvey.scripts import collect_etc
import desimodel
import desimodel.io


def afternoon_plan(night=None, restore_etc_stats='most_recent',
                   configfn='config.yaml',
                   fiber_assign_dir=None, spectra_dir=None, simulate=False,
                   desisurvey_output=None):
    """Perform daily afternoon planning.

    Afternoon planning identifies tiles available for observation and assigns
    priorities.  It must be performed before the NTS can identify new tiles to
    observe.

    Params
    ------
    night : str
        Night to plan (YYYMMDD).  Default tonight.

    restore_etc_stats : str
        Previous planned night (YYYMMDD) or etc_stats filename.
        Special strings 'start_fresh' and 'most_recent' trigger starting fresh
        and searching for the most recent file.
        Used for restoring the previous completion status of all tiles.
        Defaults to 'most_recent'.

    configfn : str
        File name of desisurvey config to use for plan.


    fiber_assign_dir : str
        Directory where fiber assign files are found.

    spectra_dir : str
        Directory where spectra are found.

    simulate : bool
        Use simulated afternoon planning rather than real afternoon planning.

    desisurvey_output : str
        Afternoon planning config is stored to desisurvey_output/{night}/.
        Default to DESISURVEY_OUTPUT if None.
    """
    log = desiutil.log.get_logger()
    if night is None:
        night = desisurvey.utils.get_current_date()
    else:
        night = desisurvey.utils.get_date(night)
    nightstr = desisurvey.utils.night_to_str(night)

    if desisurvey_output is None:
        if os.environ.get('DESISURVEY_OUTPUT') is None:
            log.error('Must set environment variable '
                      'DESISURVEY_OUTPUT!')
            return
        desisurvey_output = os.environ['DESISURVEY_OUTPUT']
    directory = os.path.join(desisurvey_output, nightstr)
    if not os.path.exists(directory):
        os.mkdir(directory)

    if configfn is None:
        configfn = desisurvey.config.Configuration._get_full_path(
            'config.yaml')
    if not os.path.exists(configfn):
        configfn = desisurvey.config.Configuration._get_full_path(configfn)

    # figuring out the current date requires having already loaded a
    # configuration file; we need to get rid of that.
    desisurvey.config.Configuration.reset()
    config = desisurvey.config.Configuration(configfn)
    log.info('Loading configuration from {}...'.format(configfn))
    tilefn = find_tile_file(config.tiles_file())
    rulesfn = find_rules_file(config.rules_file())
    if not os.path.exists(tilefn):
        log.error('{} does not exist, failing!'.format(tilefn))
        return
    if not os.path.exists(rulesfn):
        log.error('{} does not exist, failing!'.format(rulesfn))
        return
    newtilefn = os.path.join(directory, os.path.basename(tilefn))
    newrulesfn = os.path.join(directory, os.path.basename(rulesfn))
    newconfigfn = os.path.join(directory, os.path.basename(configfn))
    if os.path.exists(newtilefn):
        log.error('{} already exists, failing!'.format(newtilefn))
        return
    if os.path.exists(newrulesfn):
        log.error('{} already exists, failing!'.format(newrulesfn))
        return

    editedtiles = False
    editedrules = False
    editedoutputpath = False

    with open(configfn) as fp:
        lines = fp.readlines()
        for i in range(len(lines)):
            if re.match('^output_path:.*', lines[i]):
                lines[i] = (
                    'output_path: {}'.format(desisurvey_output) +
                    '  # edited by afternoon planning\n')
                editedoutputpath = True
            elif re.match('^tiles_file:.*', lines[i]):
                lines[i] = ('tiles_file: {}'.format(newtilefn) +
                            '  # edited by afternoon planning\n')
                editedtiles = True
            elif re.match('^rules_file:.*', lines[i]):
                lines[i] = ('rules_file: {}'.format(newrulesfn) +
                            '  # edited by afternoon planning\n')
                editedrules = True
    if not (editedtiles and editedrules and editedoutputpath):
        log.error('Could not find either tiles, rules, or output_path '
                  'in config file; failing!')
        return
    with open(newconfigfn, 'w') as fp:
        fp.writelines(lines)

    shutil.copy(tilefn, newtilefn)
    shutil.copy(rulesfn, newrulesfn)

    desisurvey.config.Configuration.reset()
    config = desisurvey.config.Configuration(newconfigfn)
    _ = desisurvey.tiles.get_tiles(use_cache=False, write_cache=True)
    rules = desisurvey.rules.Rules(config.rules_file())
    planner = desisurvey.plan.Planner(rules, simulate=simulate)
    scheduler = desisurvey.scheduler.Scheduler()

    if spectra_dir is None:
        spectra_dir = os.environ.get('DESI_SPECTRA_DIR', None)
    if spectra_dir is None:
        raise ValueError('Must pass spectra_dir to afternoon_plan or set '
                         'DESI_SPECTRA_DIR.')

    tiles, exps = collect_etc.scan_directory(spectra_dir,
                                             start_from=restore_etc_stats)
    collect_etc.write_tile_exp(tiles, exps, os.path.join(
        directory, 'etc-stats-{}.fits'.format(nightstr)))
    planner.set_donefrac(tiles['TILEID'], tiles['DONEFRAC_ETC'],
                         tiles['LASTEXPID_ETC'])

    planner.afternoon_plan(night, fiber_assign_dir=fiber_assign_dir)
    planner.save('{}/desi-status-{}.fits'.format(nightstr, nightstr))
    for fn in [newtilefn, newrulesfn, newconfigfn,
               os.path.join(directory, 'desi-status-{}.fits'.format(nightstr)),
               os.path.join(directory, 'etc-stats-{}.fits'.format(nightstr))]:
        subprocess.run(['chmod', 'a-w', fn])


def find_rules_file(file_name):
    from pkg_resources import resource_filename
    if os.path.isabs(file_name):
        full_path = file_name
    elif os.path.exists(file_name):
        return os.path.abspath(file_name)
    else:
        full_path = resource_filename('desisurvey',
                                      os.path.join('data', file_name))
    return full_path


def find_tile_file(file_name):
    if os.path.isabs(file_name):
        full_path = file_name
    elif os.path.exists(file_name):
        return os.path.abspath(file_name)
    else:
        # Locate the config file in our package data/ directory.
        full_path = desimodel.io.findfile(os.path.join('footprint', file_name))
    return full_path


def parse(options=None):
    """Parse command-line options for running afternoon planning.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Perform afternoon planning.',
        epilog='EXAMPLE: %(prog)s --night 2020-01-01')
    parser.add_argument('--night', type=str,
                        help='night to plan, default: tonight',
                        default=None)
    parser.add_argument('--restore_etc_stats', type=str,
                        help=('etc_stats file to restore. Default: '
                              '"most_recent", search for most recent.  '
                              '"fresh" to start fresh.'),
                        default='most_recent')
    parser.add_argument('--config', type=str, default=None,
                        help='config file to use for night')
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):
    outputdir = os.environ.get('DESISURVEY_OUTPUT', None)
    log = desiutil.log.get_logger()
    if outputdir is None:
        log.error('Environment variable DESISURVEY_OUTPUT must be set.')
        raise ValueError('Environment variable DESISURVEY_OUTPUT must be set.')

    afternoon_plan(night=args.night, restore_etc_stats=args.restore_etc_stats,
                   configfn=args.config)
