import desisurvey
import desisurvey.tiles
import desisurvey.rules
import desisurvey.plan
import desisurvey.scheduler
import desiutil.log
import glob
import re
import os
import shutil
from desisurvey.scripts import collect_etc


"""
This needs a bit more thought.

"Currently" config is a real path, and it uses DESISURVEY_OUTPUT.
DESISURVEY_OUTPUT gets prepended to the status file.
The tiles file is specified in config.yaml as a real path.
The rules file is specified in config.yaml as a real path (especially
if I define that to be the case; could instead have it look in config.yaml
by default; otherwise go to existing rules file).

This all sounds fine; these files could get copied to the AP directory
for the night and then used.  The config.yaml file would need to be
updated on copy to point to these copied versions, but that sounds fine, if
a little annoying.

I want to propose that DESISURVEY_OUTPUT is something like:
/path/to/desisurvey_output/YYYMMDD/
NTS gets run pointed to /path/to/desisurvey_output/YYYYMMDD/config.yaml
where it then sees all the other files it needs (rules, tiles, status file)
That all sounds okay.

Also need to find a past status file.

Okay, so say DESISURVEY_OUTPUT is a /path/to/desisurvey_output.
Make NTS and afternoon planning always careful to call config.get_path()
with a night argument.

Then we can easily find past nights.


"""

def afternoon_plan(night=None, restore_etc_stats=None, configfn='config.yaml',
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
        Used for restoring the previous completion status of all tiles.
        Defaults to not restoring status, i.e., all previous tile completion
        information is recomputed from the spectra_dir.

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

    # figuring out the current date requires having already loaded a
    # configuration file; we need to get rid of that.
    desisurvey.config.Configuration.reset()
    config = desisurvey.config.Configuration(configfn)
    log.info('Loading configuration from {}...'.format(configfn))
    if not os.path.exists(configfn):
        configfn = desisurvey.config.Configuration._get_full_path(configfn)
    tilefn = config.get_path(config.tiles_file())
    rulesfn = config.get_path(config.rules_file())
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
    shutil.copy(tilefn, newtilefn)
    shutil.copy(rulesfn, newrulesfn)

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

    desisurvey.config.Configuration.reset()
    config = desisurvey.config.Configuration(newconfigfn)
    tilesob = desisurvey.tiles.get_tiles(use_cache=False, write_cache=True)
    rules = desisurvey.rules.Rules(config.rules_file())
    planner = desisurvey.plan.Planner(rules, simulate=simulate)
    scheduler = desisurvey.scheduler.Scheduler()

    if spectra_dir is None:
        spectra_dir = config.spectra_dir()
    tiles, exps = collect_etc.scan_directory(spectra_dir,
                                             start_from=restore_etc_stats)
    collect_etc.write_tile_exp(tiles, exps, os.path.join(
        directory, 'etc_stats-{}.fits'.format(nightstr)))
    planner.set_donefrac(tiles['TILEID'], tiles['DONEFRAC_ETC'],
                         tiles['LASTEXPID_ETC'])

    planner.afternoon_plan(night, fiber_assign_dir=fiber_assign_dir)
    planner.save('{}/desi-status-{}.fits'.format(nightstr, nightstr))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Perform afternoon planning.',
        epilog='EXAMPLE: %(prog)s --night 2020-01-01')
    parser.add_argument('--night', type=str,
                        help='night to plan, default: tonight',
                        default=None)
    parser.add_argument('--restore_etc_stats', type=str,
                        help='etc_stats file to restore. Default: start fresh.',
                        default=None)
    parser.add_argument('--config', type=str, default=None,
                        help='config file to use for night')
    args = parser.parse_args()

    outputdir = os.environ.get('DESISURVEY_OUTPUT', None)
    log = desiutil.log.get_logger()
    if outputdir is None:
        log.error('Environment variable DESISURVEY_OUTPUT must be set.')
        raise ValueError('Environment variable DESISURVEY_OUTPUT must be set.')

    afternoon_plan(night=args.night, restore_etc_stats=args.restore_etc_stats,
                   configfn=args.config)
