import desisurvey
import desisurvey.tiles
import desisurvey.rules
import desisurvey.plan
import desisurvey.scheduler
import desisurvey.holdingpen
import desiutil.log
import re
import os
import shutil
import subprocess
from desisurvey.scripts import collect_etc
import desimodel.io
import numpy as np
from astropy.table import Table
from desisurvey.utils import yesno


def afternoon_plan(night=None, exposures=None,
                   configfn='config.yaml',
                   spectra_dir=None,
                   desisurvey_output=None, nts_dir=None, sv=False,
                   surveyops=None):
    """Perform daily afternoon planning.

    Afternoon planning identifies tiles available for observation and assigns
    priorities.  It must be performed before the NTS can identify new tiles to
    observe.

    Params
    ------
    night : str
        Night to plan (YYYMMDD).  Default tonight.

    exposures : str
        File name of exposures file to restore.  Default of None looks in
        $SURVEYOPS/ops/exposures.csv.

    configfn : str
        File name of desisurvey config to use for plan.

    spectra_dir : str
        Directory where spectra are found.

    desisurvey_output : str
        Afternoon planning config is stored to desisurvey_output/{night}/.
        Default to DESISURVEY_OUTPUT if None.

    nts_dir : str
        Store afternoon planning to desisurvey_output/{nts_dir} rather than
        to desisurvey_output/{night}.
        Default to None.

    sv : bool
        if True, trigger special tweaking of OBSCONDITIONS in tile file,
        donefrac in status file.

    surveyops : str
        surveyops SVN directory.  Default of None triggers looking at the
        SURVEYOPS environment variable.
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

    if os.path.exists(configfn):
        dirname, fname = os.path.split(configfn)
        if dirname == '':
            configfn = './'+configfn
    else:
        configfn = os.path.join(desisurvey_output, configfn)

    config = desisurvey.config.Configuration(configfn)
    nts_survey = config.survey()

    if nts_dir is None:
        subdir = nightstr + '-' + nts_survey.lower()
    else:
        subdir = nts_dir
    directory = os.path.join(desisurvey_output, subdir)
    if not os.path.exists(directory):
        os.mkdir(directory)
        os.chmod(directory, 0o2777)

    surveyopsdir = (surveyops if surveyops is not None
                    else os.environ.get('SURVEYOPS', None))
    if surveyopsdir is not None:
        ret = subprocess.run(['svn', 'up',
                              os.path.join(surveyopsdir, 'ops')])
        ret = subprocess.run(
            ['svn', 'up',
             os.path.join(surveyopsdir, 'mtl', 'mtl-done-tiles.ecsv')])
        if ret.returncode != 0:
            log.info('Failed to update surveyops.')
    else:
        log.info('SURVEYOPS directory not found; not performing '
                 'surveyops updates.')

    # compulsively make sure that the main SURVEYOPS repository is up to date
    # this ensures that tiles designed on the fly will use recent MTLs.
    # note, this occurs after the "private" surveyops/mtl/mtl-done-tiles.ecsv
    # update.  In the event that a check in landed between the two svn updates,
    # AP would not release updated tiles for observation that night.  That's
    # fine.  In the opposite order, AP would release tiles, but they would use
    # out of date MTLs, a disaster.
    mainsurveyopsdir = os.path.join(
        os.environ['DESI_ROOT'], 'survey', 'ops', 'surveyops', 'trunk')
    subprocess.run(['svn', 'up', mainsurveyopsdir])

    fbadir = os.environ['FIBER_ASSIGN_DIR']
    # update FIBER_ASSIGN_DIR
    subprocess.run(['svn', 'up', fbadir])

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

    if surveyopsdir is not None:
        import filecmp
        surveyopstilefn = os.path.join(surveyopsdir, 'ops',
                                       os.path.basename(tilefn))
        if not os.path.exists(surveyopstilefn):
            log.info('No SURVEYOPS tile file; not updating from SURVEYOPS.')
            doupdate = False
        else:
            doupdate = not filecmp.cmp(tilefn, surveyopstilefn, shallow=False)
        if doupdate:
            qstr = ('tile file in SURVEYOPS is updated relative '
                    'to {}.  Overwrite with SURVEYOPS tile file?'.format(
                        tilefn))
            update = yesno(qstr)
            if update:
                shutil.copy(surveyopstilefn, tilefn)

    # config file will always be called config.yaml so ICS knows where to look
    newconfigfn = os.path.join(directory, 'config.yaml')
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
                    "output_path: '{DESISURVEY_OUTPUT}'"
                    '  # edited by afternoon planning\n')
                editedoutputpath = True
            elif re.match('^tiles_file:.*', lines[i]):
                lines[i] = (
                    "tiles_file: {}/{}  # edited by afternoon planning")
                lines[i] = lines[i].format(subdir, os.path.basename(newtilefn))
                editedtiles = True
            elif re.match('^rules_file:.*', lines[i]):
                lines[i] = (
                    "rules_file: {}/{}  # edited by afternoon planning")
                lines[i] = lines[i].format(subdir,
                                           os.path.basename(newrulesfn))
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
    planner = desisurvey.plan.Planner(rules)

    if spectra_dir is None:
        spectra_dir = os.environ.get('DESI_SPECTRA_DIR', None)
    if spectra_dir is None:
        raise ValueError('Must pass spectra_dir to afternoon_plan or set '
                         'DESI_SPECTRA_DIR.')

    offlinepipelinefiles = ['tsnr-exposures.fits', 'tiles.csv']
    for i, fn in enumerate(offlinepipelinefiles):
        os.system('wget -q https://data.desi.lbl.gov/desi/spectro/redux/daily/'
                  '{0} -O {0}.tmp'.format(fn))
        filelen = os.stat('{}.tmp'.format(fn)).st_size
        if filelen > 0:
            os.rename('{}.tmp'.format(fn), fn)
        else:
            log.warning('Updating {} failed!'.format(fn))
        if not os.path.exists(fn):
            offlinepipelinefiles[i] = None

    if surveyopsdir is None:
        raise ValueError('SURVEYOPS is not set; cannot do MTL updates; '
                         'failing!')
    else:
        mtldonefn = os.path.join(surveyopsdir, 'mtl', 'mtl-done-tiles.ecsv')
        badexpfn = os.path.join(surveyopsdir, 'ops', 'bad_exp_list.csv')

    if exposures is None:
        # expdir = os.path.join(os.environ['SURVEYOPS'], 'ops')
        if surveyopsdir is not None:
            expdir = os.path.join(surveyopsdir, 'ops')
        else:
            expdir = os.environ['DESISURVEY_OUTPUT']
        exposures = os.path.join(expdir, 'exposures.ecsv')
    tiles, exps = collect_etc.scan_directory(
        spectra_dir, start_from=exposures,
        offlinedepth=offlinepipelinefiles[0],
        offlinetiles=offlinepipelinefiles[1],
        mtldone=mtldonefn,
        badexp=badexpfn)
    collect_etc.write_exp(exps, os.path.join(directory, 'exposures.ecsv'))

    planner.set_donefrac(tiles['TILEID'], tiles['DONEFRAC'],
                         ignore_pending=True, nobs=tiles['NOBS'])
    m = tiles['OFFLINE_DONE'] != 0
    planner.set_donefrac(tiles['TILEID'][m], status=['obsend']*np.sum(m),
                         ignore_pending=True)
    m = tiles['MTL_DONE'] != 0
    planner.set_donefrac(tiles['TILEID'][m], status=['done']*np.sum(m),
                         ignore_pending=True)
    svmode = getattr(config, 'svmode', None)
    svmode = svmode() if svmode is not None else False
    if svmode:
        # overwrite donefracs
        from desisurvey import svstats
        numnight = collect_etc.number_per_night(exps)
        donefracnight = svstats.donefrac_nnight(numnight)
        _, md, mt = np.intersect1d(donefracnight['TILEID'],
                                   tiles['TILEID'], return_indices=True)
        nneeded = donefracnight['NNIGHT_NEEDED']
        nneeded = nneeded + (nneeded == 0)
        planner.set_donefrac(donefracnight['TILEID'],
                             donefracnight['NNIGHT'] / nneeded,
                             ignore_pending=True)

    planner.afternoon_plan(night)
    planner.save(os.path.join(subdir, os.path.basename(newtilefn)))
    # force reload of tile file from cache.
    _ = desisurvey.tiles.get_tiles(use_cache=False, write_cache=True)
    faholddir = os.environ.get('FA_HOLDING_PEN', None)
    if faholddir is not None:
        desisurvey.holdingpen.maintain_holding_pen_and_svn(
            fbadir, faholddir, mtldonefn)
    else:
        log.error('FA_HOLDING_PEN is None, skipping holding pen '
                  'maintenance!')
    # pick up any changes to available tiles.
    planner.afternoon_plan(night)
    planner.save(os.path.join(subdir, os.path.basename(newtilefn)))

    for fn in [newrulesfn, newconfigfn]:
        subprocess.run(['chmod', 'a-w', fn])
    if surveyopsdir is not None:
        surveyopsopsdir = os.path.join(surveyopsdir, 'ops')
        subprocess.run(['cp', os.path.join(directory, 'exposures.ecsv'),
                        surveyopsopsdir])
        subprocess.run(['cp', newtilefn, surveyopsopsdir])
        subprocess.run(['svn', 'status', surveyopsopsdir])
        if yesno('Okay to commit updates to SURVEYOPS svn?'):
            subprocess.run(
                ['svn', 'ci', surveyopsopsdir,
                 '-m "Update exposures and tiles for %s"' % nightstr])
    shutil.copy(newtilefn, tilefn)  # update working directory tilefn


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
    parser.add_argument('--exposures', type=str, default=None,
                        help=('exposures file to use. If not set, use '
                              '$SURVEYOPS/ops/exposures.ecsv'))
    parser.add_argument('--config', type=str, default=None,
                        help='config file to use for night')
    parser.add_argument('--nts-dir', type=str, default=None,
                        help=('subdirectory of DESISURVEY_OUTPUT in which to '
                              'store plan.'))
    parser.add_argument('--surveyops', type=str, default=None,
                        help=('SURVEYOPS SVN directory, default SURVEYOPS '
                              'environment variable.'))
    parser.add_argument('--sv',
                        action='store_true',
                        help='turn on special SV planning mode.')
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

    configfn = args.config
    if configfn is None:
        config = desisurvey.config.Configuration()
    elif os.path.exists(configfn):
        config = desisurvey.config.Configuration(configfn)
    else:
        configfn = os.path.join(os.environ['DESISURVEY_OUTPUT'], configfn)
        config = desisurvey.config.Configuration(configfn)
    log.info('Loading configuration from {}...'.format(config.file_name))

    afternoon_plan(night=args.night, exposures=args.exposures,
                   configfn=args.config, nts_dir=args.nts_dir, sv=args.sv,
                   surveyops=args.surveyops)
