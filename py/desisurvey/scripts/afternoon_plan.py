import desisurvey
import desisurvey.tiles
import desisurvey.config
import glob
import re
import os
from desisurvey.scripts import collect_etc

def afternoon_plan(night=None, lastnight=None, fiber_assign_dir=None,
                   spectra_dir=None, simulate=False):
    """
    Perform daily afternoon planning.

    Afternoon planning identifies tiles available for observation and assigns
    priorities.  It must be performed before the NTS can identify new tiles to
    observe.

    Params
    ------
    night : str, ISO 8601.  The night to plan.  Default tonight.

    lastnight : str, ISO 8601.  The previous planned night.  Used for restoring
        the previous completion status of all tiles.  Defaults to not
        restoring status, i.e., all previous tile completion information is
        ignored!

    fiber_assign_dir : str.  Directory where fiber assign files are found.

    spectra_dir : str.  Directory where spectra are found.
    """
    if night is None:
        night = desisurvey.utils.get_current_date()

    night = desisurvey.utils.get_date(night)
    desisurvey.config.Configuration.reset()
    config = desisurvey.config.Configuration()
    tilesob = desisurvey.tiles.get_tiles(use_cache=False, write_cache=True)
    rules = desisurvey.rules.Rules(config.rules())
    if lastnight is not None:
        planner = desisurvey.plan.Planner(
            rules, restore='desi-status-{}.fits'.format(lastnight), simulate=simulate)
        scheduler = desisurvey.scheduler.Scheduler(
            restore='desi-status-{}.fits'.format(lastnight))
    else:
        planner = desisurvey.plan.Planner(rules, simulate=simulate)
        scheduler = desisurvey.scheduler.Scheduler()

    nightstr = desisurvey.utils.night_to_str(night)

    completed = scheduler.completed
    if spectra_dir is not None:
        config = desisurvey.config.Configuration()
        etcfn = config.get_path('etc-status-{}.fits'.format(nightstr))
        if os.path.exists(etcfn):
            from astropy.io import fits
            print('Using pre-existing ETC status file %s.' % etcfn)
            tiles = fits.getdata(etcfn, 'TILES')
        else:
            tiles, exps = desisurvey.scripts.collect_etc.scan_directory(spectra_dir)
            collect_etc.write_tile_exp(tiles, exps, etcfn)
        planner.set_donefrac(tiles['TILEID'], tiles['DONEFRAC'], tiles['LASTEXPID'])

    planner.afternoon_plan(night, fiber_assign_dir=fiber_assign_dir)
    planner.save('desi-status-{}.fits'.format(nightstr))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Perform afternoon planning.',
        epilog='EXAMPLE: %(prog)s --night 2020-01-01')
    parser.add_argument('--night', type=str,
                        help='night to plan, default: tonight',
                        default=None)
    parser.add_argument('--lastnight', type=str,
                        help='night to restore, default: start fresh.',
                        default=None)

    args = parser.parse_args()
    afternoon_plan(args.night, args.lastnight)
