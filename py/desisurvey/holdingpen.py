import os
import subprocess
import re
import glob
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
import desiutil.log
import desisurvey.config
import desisurvey.plan
import desisurvey.tiles
from desisurvey.utils import yesno

logger = desiutil.log.get_logger()


def make_tileid_list(fadir):
    fafiles = glob.glob(os.path.join(fadir, '**/*.fits*'), recursive=True)
    rgx = re.compile(r'.*fiberassign-(\d+)\.fits(\.gz)?')
    existing_tileids = []
    existing_fafiles = []
    for fn in fafiles:
        match = rgx.match(fn)
        if match:
            existing_tileids.append(int(match.group(1)))
            existing_fafiles.append(fn)
    return np.array(existing_tileids), np.array(existing_fafiles)


def tileid_to_clean(faholddir, fadir, mtldone):
    """Identify invalidated fiberassign files for deletion.

    Scans faholddir for fiberassign files.  Compares the MTLTIMES with the times in
    the mtldone file.  If a fiberassign file was designed before an overlapping
    tile which later had MTL updates, that fiberassign file is "invalid" and
    should be deleted.

    Parameters
    ----------
    faholddir : str
        directory name of fiberassign holding pen
    fadir : str
        directory name of svn-controlled fiber assign directory.
    mtldone : array
        numpy array of finished tile MTL updates.  Must contain at least
        TIMESTAMP and TILEID fields.
    """
    import dateutil.parser
    cfg = desisurvey.config.Configuration()
    tiles = desisurvey.tiles.get_tiles()
    plan = desisurvey.plan.Planner(restore=cfg.tiles_file())
    existing = tiles.tileID[plan.tile_status != 'unobs']
    m = (plan.tile_status == 'unobs') & (plan.tile_priority <= 0)
    existing_tileids, existing_fafiles = make_tileid_list(faholddir)
    intiles = np.isin(existing_tileids, tiles.tileID)
    existing_tileids = existing_tileids[intiles]
    existing_fafiles = existing_fafiles[intiles]
    existing = existing[np.isin(existing, existing_tileids)]
    logger.info('Reading in MTLTIME header from %d fiberassign files...' %
                len(existing_fafiles))
    mtltime = [fits.getheader(fn).get('MTLTIME', 'None')
               for fn in existing_fafiles]
    m = np.array([mtltime0 is not None for mtltime0 in mtltime])
    if np.any(~m):
        logger.warning('MTLTIME not found for tiles {}!'.format(
            ' '.join([x for x in existing_fafiles[~m]])))
    existing_tileids = existing_tileids[m]
    existing_fafiles = existing_fafiles[m]
    mtltime = np.array(mtltime)[m]
    mtltime = Time([dateutil.parser.parse(mtltime0)
                    for mtltime0 in mtltime]).mjd
    # we have the mtl times for all existing fa files.
    # we want the largest MTL time of any overlapping tile which has
    # status != 'unobs'
    tilemtltime = np.zeros(tiles.ntiles, dtype='f8') - 1
    index, mask = tiles.index(existing_tileids, return_mask=True)
    if np.sum(~mask) > 0:
        logger.info('Ignoring {} TILEID not in the tile file'.format(
            np.sum(~mask)))
    index = index[mask]
    existing_tileids = existing_tileids[mask]
    mtltime = mtltime[mask]
    tilemtltime[index] = mtltime
    # this has the MTL design time of all of the tiles.
    # we also need the MTL done time of all the tiles.
    index, mask = tiles.index(mtldone['TILEID'], return_mask=True)
    mtldonetime = [dateutil.parser.parse(mtltime0)
                   for mtltime0 in mtldone['TIMESTAMP']]
    mtldonetime = Time(mtldonetime).mjd
    tilemtldonetime = np.zeros(tiles.ntiles, dtype='f8')
    tilemtldonetime[index[mask]] = mtldonetime[mask]
    maxoverlappingtilemtldonetime = np.zeros(tiles.ntiles, dtype='f8')
    for i, neighbors in enumerate(tiles.overlapping):
        if len(neighbors) == 0:
            continue
        maxoverlappingtilemtldonetime[i] = np.max(tilemtldonetime[neighbors])
    expired = ((maxoverlappingtilemtldonetime > tilemtltime)
               & (plan.tile_status == 'unobs') & (tilemtltime > -1))
    for tileid in existing:
        tileidpadstr = '%06d' % tileid
        fafn = os.path.join(fadir, tileidpadstr[:3],
                            'fiberassign-%s.fits.gz' % tileidpadstr)
        if not os.path.exists(fafn):
            logger.error('Tile {} is not unobs, '.format(fafn) +
                         'but does not exist in SVN?!')
    return tiles.tileID[expired]


def remove_tiles_from_dir(dirname, tileid):
    for tileid0 in tileid:
        for ext in ['fits.gz', 'png', 'log']:
            expidstr= '{:06d}'.format(tileid0)
            os.remove(os.path.join(
                dirname, expidstr[:3],
                'fiberassign-{}.{}'.format(expidstr, ext)))


def missing_tileid(fadir, faholddir):
    """Return missing TILEID and superfluous TILEID.

    The fiberassign holding pen should include all TILEID
    for available, unobserved tiles.  It should include no TILEID
    for unavailable or observed tiles.  This function computes the list
    of TILEID that should exist, but do not, as well as the list of TILEID
    that should not exist, but do.

    Parameters
    ----------
    fadir : str
        directory name of fiberassign directory
    faholddir : str
        directory name of fiberassign holding pen

    Returns
    -------
    missingtiles, extratiles
    missingtiles : array
        array of TILEID for tiles that do not exist, but should.
        These need to be designed and added to the holding pen.
    extratiles : array
        array of TILEID for tiles that exist, but should not.
        These need to be deleted from the holding pen.
    """
    cfg = desisurvey.config.Configuration()
    tiles = desisurvey.tiles.get_tiles()
    plan = desisurvey.plan.Planner(restore=cfg.tiles_file())
    tileid, fafn = make_tileid_list(faholddir)
    shouldexist = tiles.tileID[(plan.tile_status == 'unobs') &
                               (plan.tile_priority > 0)]
    missingtiles = set(shouldexist) - set(tileid)
    shouldnotexist = tiles.tileID[(plan.tile_status != 'unobs') |
                                  (plan.tile_priority <= 0)]
    doesexist = np.isin(tileid, shouldnotexist)
    count = 0
    for tileid0 in tileid[doesexist]:
        expidstr = '{:06d}'.format(tileid0)
        if not os.path.exists(os.path.join(
                fadir, expidstr[:3],
                'fiberassign-{}.fits.gz'.format(expidstr))):
            logger.error('TILEID %d should be checked into and is not!' %
                         tileid0)
        else:
            count += 1
    if count > 0:
        logger.info('Confirmed %d files in SVN also in holding pen.' %
                    count)
        logger.info('TILEID: ' + ' '.join(
            [str(x) for x in np.sort(tileid[doesexist])]))
    return (np.sort(np.array([x for x in missingtiles])),
            np.sort(tileid[doesexist]))


def maintain_svn(svn, untrackedonly=True, verbose=False):
    cfg = desisurvey.config.Configuration()
    tiles = desisurvey.tiles.get_tiles()
    plan = desisurvey.plan.Planner(restore=cfg.tiles_file())
    fnames = []
    if untrackedonly:
        res = subprocess.run(['svn', 'status', svn], capture_output=True)
        output = res.stdout.decode('utf8')
        for line in output.split('\n'):
            if len(line) == 0:
                continue
            modtype = line[0]
            if modtype != '?':
                print('unrecognized line: "{}", ignoring.'.format(line))
                continue
            # new file.  We need to check it in or delete it.
            fname = line[8:]
            fnames.append(fname)
    else:
        import glob
        fnames = glob.glob(os.path.join(svn, '**/*'), recursive=True)
    rgx = re.compile(svn + '/' +
                     r'\d\d\d/fiberassign-(\d+)\.(fits|fits\.gz|png|log)')
    todelete = []
    tocommit = []
    mintileid = np.min(tiles.tileID)
    maxtileid = np.max(tiles.tileID)
    for fname in fnames:
        match = rgx.match(fname)
        if not match:
            if verbose:
                logger.warn('unrecognized filename: "{}", ignoring.'.format(fname))
            continue
        tileid = int(match.group(1))
        idx, mask = tiles.index(tileid, return_mask=True)
        if not mask:
            if verbose and (tileid >= mintileid) and (tileid <= maxtileid):
                logger.warn('unrecognized TILEID {}, ignoring.'.format(tileid))
            continue
        if plan.tile_status[idx] == 'unobs':
            todelete.append(fname)
        else:
            tocommit.append(fname)
    if not untrackedonly:
        tocommit = []
    return todelete, tocommit


def execute_svn_maintenance(todelete, tocommit, echo=False, svnrm=False):
    if echo:
        cmd = ['echo', 'svn']
    else:
        cmd = ['svn']
    for fname in todelete:
        if svnrm:
            subprocess.run(cmd + ['rm', fname])
        else:
            if not echo:
                os.remove(fname)
            else:
                print('removing ', fname)
    for fname in tocommit:
        subprocess.run(cmd + ['add', fname])


def maintain_holding_pen_and_svn(fbadir, faholddir, mtldonefn):
    todelete, tocommit = maintain_svn(fbadir)
    if len(todelete) + len(tocommit) > 0:
        logger.info(('To delete from %s:\n' % fbadir) +
                    '\n'.join([os.path.basename(x) for x in todelete]))
        logger.info(('To commit to %s:\n' % fbadir) +
                    '\n'.join([os.path.basename(x) for x in tocommit]))
        qstr = ('Preparing to perform svn fiberassign maintenance, '
                'deleting {} and committing {} files.  Continue?'.format(
                    len(todelete), len(tocommit)))
        okay = yesno(qstr)
        if okay:
            execute_svn_maintenance(todelete, tocommit, echo=True)
            okay = yesno('The following commands will be executed.  '
                         'Still okay?')
            if okay:
                execute_svn_maintenance(todelete, tocommit)
            okay = yesno('Commit to svn?')
            if okay:
                subprocess.run(['svn', 'ci', fbadir,
                                '-m "Adding newly observed tiles."'])
    if mtldonefn is not None:
        invalid = tileid_to_clean(faholddir, fbadir, Table.read(mtldonefn))
    if len(invalid) > 0:
        okay = yesno(('Deleting %d out-of-date fiberassign files from ' +
                      'holding pen .  Continue?').format(len(invalid)))
        if okay:
            remove_tiles_from_dir(faholddir, invalid)
    missing, extra = missing_tileid(fbadir, faholddir)
    if len(extra) > 0:
        okay = yesno(('Deleting %d fiberassign files in SVN from the '
                      'holding pen.  Continue?') % len(extra))
        if okay:
            remove_tiles_from_dir(faholddir, extra)
    if len(missing) < 100:
        logger.info('Need to design the following tiles here! ' +
                    ' '.join([str(x) for x in missing]))
    else:
        logger.info('Need to design many (%d) tiles here!' % len(missing))
