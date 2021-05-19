import os
import subprocess
import re
import glob
import numpy as np
from astropy.io import fits
from astropy.time import Time
import desiutil.log
import desisurvey.config
import desisurvey.plan
import desisurvey.tiles

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


def tileid_to_clean(fadir, mtldone):
    """Identify invalidated fiberassign files for deletion.

    Scans fadir for fiberassign files.  Compares the MTLTIMES with the times in
    the mtldone file.  If a fiberassign file was designed before an overlapping
    tile which later had MTL updates, that fiberassign file is "invalid" and
    should be deleted.

    Parameters
    ----------
    fadir : str
        directory name of fiberassign holding pen
    mtldone : array
        numpy array of finished tile MTL updates.  Must contain at least
        TIMESTAMP and TILEID fields.
    """
    cfg = desisurvey.config.Configuration()
    tiles = desisurvey.tiles.get_tiles()
    plan = desisurvey.plan.Planner(restore=cfg.tiles_file())
    existing = tiles.tileID[plan.tile_status != 'unobs']
    m = (plan.tile_status == 'unobs') & (plan.tile_priority <= 0)
    unavailable = tiles.tileID[m]
    existing_tileids, existing_fafiles = make_tileid_list(fadir)
    intiles = np.isin(existing_tileids, tiles.tileID)
    existing_tileids = existing_tileids[intiles]
    existing_fafiles = existing_fafiles[intiles]
    mtltime = np.array([fits.getheader(fn).get('MTLTIME', 'None')
                        for fn in existing_fafiles])
    mtltime = Time(mtltime).mjd
    # we have the mtl times for all existing fa files.
    # we want the largest MTL time of any overlapping tile which has
    # status != 'unobs'
    tilemtltime = np.zeros(tiles.ntiles, dtype='f8')
    index, mask = tiles.index(existing_tileids, return_mask=True)
    if np.sum(mask) > 0:
        logger.info('Ignoring {} TILEID not in the tile file'.format(
            np.sum(~mask)))
    index = index[mask]
    existing_tileids = existing_tileids[mask]
    mtltime = mtltime[mask]
    tilemtltime[index] = mtltime
    # this has the MTL design time of all of the tiles.
    # we also need the MTL done time of all the tiles.
    index, mask = tiles.index(mtldone['TILEID'], return_mask=True)
    mtldonetime = np.zeros(tiles.ntiles, dtype='f8')
    mtldonetime[index[mask]] = Time(mtldone['TIMESTAMP'][mask]).mjd
    maxoverlappingmtldonetime = np.zeros(tiles.ntiles, dtype='f8')
    for i, neighbors in enumerate(tiles.overlapping):
        maxoverlappingmtldonetime[i] = np.max(mtldonetime[neighbors])
    expired = maxoverlappingmtldonetime > mtltime
    tilestoclean = np.unique(
        np.concatenate([existing, unavailable, tiles.tileID[expired]]))
    return tilestoclean


def missing_tileid(fadir):
    """Return missing TILEID and superfluous TILEID.

    The fiberassign holding pen should include all TILEID
    for available, unobserved tiles.  It should include no TILEID
    for unavailable or observed tiles.  This function computes the list
    of TILEID that should exist, but do not, as well as the list of TILEID
    that should not exist, but do.

    Parameters
    ----------
    fadir : str
        directory name of fiberassign holding pen

    Returns
    -------
    missingtiles, extratiles
    missingtiles : array
        array of TILEID for tiles that do not exist, but should.
    extratiles : array
        array of TILEID for tiles that exist, but should not.
    """
    cfg = desisurvey.config.Configuration()
    tiles = desisurvey.tiles.get_tiles()
    plan = desisurvey.plan.Planner(restore=cfg.tiles_file())
    tileid, fafn = make_tileid_list(fadir)
    shouldexist = tiles.tileID[(plan.tile_status == 'unobs') &
                               (plan.tile_priority > 0)]
    missingtiles = set(shouldexist) - set(tileid)
    shouldnotexist = tiles.tileID[(plan.tile_status != 'unobs') |
                                  (plan.tile_priority <= 0)]
    doesexist = np.isin(tileid, shouldnotexist)
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

