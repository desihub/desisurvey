import os
import glob
import re
import json
import numpy as np
from astropy.io import fits
import desiutil.log
import desisurvey.etc
import desisurvey.tiles
import desisurvey.ephem
import astropy.table

log = desiutil.log.get_logger()


class subslices:
    "Iterator for looping over subsets of an array"
    def __init__(self, data, uind=None):
        if uind is None:
            self.uind = np.unique(data, return_index=True)[1]
        else:
            self.uind = uind.copy()
        self.ind = 0
        self.length = len(data)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.uind)

    def __next__(self):
        if self.ind == len(self.uind):
            raise StopIteration
        if self.ind == len(self.uind)-1:
            last = self.length
        else:
            last = self.uind[self.ind+1]
        first = self.uind[self.ind]
        self.ind += 1
        return first, last

    def next(self):
        return self.__next__()


def cull_old_files(files, start_from):
    """Return only subset of files with EXPID larger than any EXPID in
    start_from.
    """
    expid = np.array([int(os.path.basename(f)[5:13]) for f in files])
    # extract just the expid; better to do this with a regex, but...
    maxexpid = np.max(start_from['EXPID'])
    return [f for (f, e) in zip(files, expid) if e > maxexpid]


def scan_directory(dirname, start_from=None,
                   offlinedepth=None, mtldone=None):
    """Scan directory for spectra with ETC statistics to collect.

    Parameters
    ----------
    dirname : str
        directory path to scan.  All fits files under dirname are searched
        for ETC statistics.
        This needs to be updated to ~DESI files only, with more care given
        to where these keywords are actually found.
    start_from : str
        etc_stats file to start from.  Nights already in the etc_stats file
        will not be collected.  If a YYYMMDD string, look for etc_stats file
        in DESISURVEY_OUTPUT/YYYYMMDD/etc_stats-{YYYYMMDD}.fits
        None indicates starting fresh.
    offlinedepth : str
        offline depth file to use.  Fills out donefracs according to
        R_DEPTH in the file, plus config.nominal_exposure_time
    mtldone : str
        mtl done file to use.  Fills out done status according to presence
        in mtl done file.
    """
    log.info('Scanning {} for desi exposures...'.format(dirname))
    if start_from is None:
        files = glob.glob(os.path.join(dirname, '**/desi*.fits.fz'),
                          recursive=True)
        start_exps = None
    else:
        files = []
        subdirs = os.listdir(dirname)
        subdirs = np.sort(subdirs)[::-1]
        if os.path.exists(start_from):
            fn = start_from
        else:
            fn = os.path.join(os.environ['DESISURVEY_OUTPUT'], start_from)
        if not os.path.exists(fn):
            log.error('Could not find file {} to start from!'.format(
                start_from))
            return
        start_exps = astropy.table.Table.read(fn)
        # painful, fragile invocations to make sure that columns
        # aren't truncated at fewer characters than we might want.
        obstype = np.zeros(len(start_exps), dtype='U20')
        obstype[:] = start_exps['OBSTYPE']
        start_exps['OBSTYPE'] = obstype
        program = np.zeros(len(start_exps), dtype='U20')
        program[:] = start_exps['PROGRAM']
        start_exps['PROGRAM'] = program
        quality = np.zeros(len(start_exps), dtype='U20')
        quality[:] = start_exps['QUALITY']
        start_exps['QUALITY'] = quality
        comments = np.zeros(len(start_exps), dtype='U80')
        comments[:] = start_exps['COMMENTS']
        start_exps['COMMENTS'] = comments
        maxexpid = np.max(start_exps['EXPID'])
        for subdir in subdirs:
            if not os.path.isdir(os.path.join(dirname, subdir)):
                continue
            files0 = glob.glob(os.path.join(dirname, subdir,
                                            '**/desi*.fits.fz'))
            expids = [re.findall(r'-(\d+)\.', os.path.basename(f)) for f
                      in files0]
            if len(expids) == 0:
                log.info('No desi exposures on night {}'.format(subdir))
                continue
            expids = [int(expid[0]) for expid in expids if len(expid) > 0]
            files += files0
            if min(expids) <= maxexpid:
                break
        files = cull_old_files(files, start_exps)

    log.info(('Found {} new raw spectra, extracting header ' +
              'information...').format(len(files)))
    exps = np.zeros(len(files), dtype=[
        ('NIGHT', 'U8'), ('TILEID', 'i4'), ('EXPID', 'i4'),
        ('OBSTYPE', 'U20'), ('PROGRAM', 'U20'), ('EXPTIME', 'f4'),
        ('EFFTIME_ETC', 'f4'), ('EFFTIME_SPEC', 'f4'), ('EFFTIME', 'f4'),
        ('GOALTIME', 'f4'),
        ('QUALITY', 'U20'), ('COMMENTS', 'U80')])
    for i, fn in enumerate(files):
        if (i % 1000) == 0:
            log.info('Extracting headers from file {} of {}...'.format(
                i+1, len(files)))
        hdr = {}
        for ename in ['SPEC', 'SPS']:
            try:
                hdr = fits.getheader(fn, ename)
                break
            except Exception:
                continue
        hasrequest = os.path.exists(
            fn.replace('desi-', 'request-').replace('.fits.fz', '.json'))
        etcfn = fn.replace('desi-', 'etc-').replace('.fits.fz', '.json')
        hasetc = os.path.exists(etcfn)
        etctime = -1
        if hasetc:
            try:
                etc = json.load(open(etcfn))
                etctime = etc['accum']['efftime'][-1]
            except Exception as e:
                print('Exception reading file ', etcfn, e)
        exps['NIGHT'][i] = hdr.get('NIGHT', -1)
        exps['TILEID'][i] = hdr.get('TILEID', -1)
        exps['EXPID'][i] = hdr.get('EXPID', -1)
        exps['OBSTYPE'][i] = str(hdr.get('OBSTYPE', '')).upper().strip()
        exps['PROGRAM'][i] = str(hdr.get('PROGRAM', '')).strip()
        exps['EXPTIME'][i] = hdr.get('EXPTIME', -1)
        exps['EFFTIME_ETC'][i] = etctime
        exps['EFFTIME_SPEC'][i] = -1
        exps['GOALTIME'][i] = hdr.get('GOALTIME', -1)
        exps['QUALITY'][i] = 'good' if hasrequest else 'bad'
        exps['COMMENTS'][i] = ''
    exps['EFFTIME'] = -1
    exptime_clipped = np.where(exps['EXPTIME'] < 59.0, 0, exps['EXPTIME'])
    exps['EFFTIME'] = np.where(exps['EFFTIME_ETC'] >= 0, exps['EFFTIME_ETC'],
                               exptime_clipped)
    obstypes = np.array([f.upper().strip() for f in exps['OBSTYPE']],
                        dtype=exps['OBSTYPE'].dtype)
    m = (exps['TILEID'] == -1) | (obstypes != 'SCIENCE')
    if np.any(m):
        log.info('Ignoring {} files due to weird OBSTYPE or TILEID'.format(
            np.sum(m)))
        exps = exps[~m]
    exps = exps[np.argsort(exps['EXPID'])]
    if start_exps is not None:
        exps = astropy.table.vstack([start_exps, astropy.table.Table(exps)])
    if offlinedepth is not None:
        exps = update_donefrac_from_offline(exps, offlinedepth)
    # replace EFFTIME with EFFTIME_SPEC where available
    exps['EFFTIME'] = np.where(exps['EFFTIME_SPEC'] < 0,
                               exps['EFFTIME'], exps['EFFTIME_SPEC'])
    ntiles = len(np.unique(exps['TILEID']))
    tiles = np.zeros(ntiles, dtype=[
        ('TILEID', 'i4'), ('PROGRAM', 'U20'), ('EFFTIME', 'f4'),
        ('DONEFRAC', 'f4'),
        ('NOBS', 'i4'), ('MTL_DONE', 'bool')])
    s = np.argsort(exps['TILEID'])
    nomtimefa = np.zeros(len(tiles), dtype='f4')
    for i, (f, l) in enumerate(subslices(exps['TILEID'][s])):
        ind = s[f:l]
        tiles['TILEID'][i] = exps['TILEID'][ind[0]]
        tiles['EFFTIME'][i] = np.sum(np.clip(
            exps['EFFTIME'][ind], 0, np.inf))
        # potentially only want to consider "good" exposures here?
        tiles['NOBS'][i] = len(ind)
        nomtimefa[i] = np.median(exps['GOALTIME'][ind])
        if np.any(np.abs(exps['GOALTIME'][ind] - nomtimefa[i]) > 1):
            log.warning('Inconsistent GOALTIME on tile ', tiles['TILEID'][i])
    if mtldone is not None:
        tiles = update_mtldone(tiles, mtldone)
    tob = desisurvey.tiles.get_tiles()
    idx, mask = tob.index(tiles['TILEID'], return_mask=True)
    tiles['PROGRAM'] = 'UNKNOWN'
    tiles['PROGRAM'][mask] = [tob.tileprogram[i] for i in idx[mask]]
    nomtimetf = desisurvey.tiles.get_nominal_program_times(
        tiles['PROGRAM'])
    nomtime = np.where(nomtimefa > 0, nomtimefa, nomtimetf)
    tiles['DONEFRAC'] = tiles['EFFTIME'] / nomtime
    s = np.argsort(exps['EXPID'])
    exps = exps[s]
    return tiles, exps


def write_exp(exps, fn):
    """Write tile & exposure ETC statistics from numpy objects.

    Parameters
    ----------
    tiles : array
        tile ETC statistics

    exps : array
        exposure ETC statistics
    """
    tab = astropy.table.Table(exps)
    tab['NIGHT'].description = 'YYYYMMDD'
    tab['TILEID'].description = 'Tile ID'
    tab['EXPID'].description = 'Exposure ID'
    tab['OBSTYPE'].description = 'Exposure OBSTYPE'
    tab['EXPTIME'].description = 'Exposure time'
    tab['EXPTIME'].format = '%9.3f'
    tab['EFFTIME_ETC'].description = 'ETC effective exposure time'
    tab['EFFTIME_ETC'].format = '%9.3f'
    tab['EFFTIME_SPEC'].description = 'Effective time from offline pipeline.'
    tab['EFFTIME_SPEC'].format = '%7.3f'
    tab['EFFTIME'].description = 'Adopted effective time.'
    tab['EFFTIME'].format = '%7.3f'
    tab['QUALITY'].description = 'Exposure quality'
    tab['COMMENTS'].description = 'Additional comments'
    tab['GOALTIME'].description = 'Goal effective exposure time'
    tab['GOALTIME'].format = '%6.1f'
    tab['PROGRAM'].description = 'PROGRAM of exposure'
    tab.write(fn, overwrite=True)


def convert_fits(fits):
    outdtype = fits.dtype.descr
    newdtype = []
    for field in outdtype:
        newfield = tuple(field)
        if 'S' in newfield[-1]:
            newfield = newfield[:-1] + (newfield[-1].replace('S', 'U'),)
        newdtype += [newfield]
    out = np.zeros(len(fits), dtype=newdtype)
    for field in fits.dtype.names:
        out[field] = fits[field]
    return out


def get_conditions(mjd):
    """Determine DARK/GRAY/BRIGHT for a set of exposures.

    Parameters
    ----------
    mjd: array of mjds to query

    Returns
    -------
    conditions: array of strings DARK, GRAY, BRIGHT, UNKNOWN
    """
    tiles = desisurvey.tiles.get_tiles()
    ephem = desisurvey.ephem.get_ephem()
    # taken in 2019 or later, with known MJD---removes some test exposures
    okmjd = np.isfinite(mjd) & (mjd > 58484)
    nights = ephem.table
    indices = np.repeat(np.arange(len(nights)), 2)
    startstop = np.concatenate([[night['brightdusk'], night['brightdawn']]
                                for night in nights])
    nightind = np.zeros(len(mjd), dtype='f8')
    nightind[~okmjd] = -1
    nightind[okmjd] = np.interp(mjd[okmjd], startstop, indices)
    # a lot of exposures apparently taken during the day?
    # We should mark these as belonging to the "day" program?
    mday = nightind != nightind.astype('i4')
    if np.sum(mday & okmjd) > 0:
        log.info('%d/%d exposures were taken during the day.' %
                 (np.sum(mday & okmjd), np.sum(okmjd)))
    nightind = nightind.astype('i4')
    conditions = []
    for mjd0, night in zip(mjd[~mday & okmjd],
                           nights[nightind[~mday & okmjd]]):
        nighttimes = np.concatenate([[night['brightdusk'], night['dusk']],
                                     night['changes'][night['changes'] != 0],
                                     [night['dawn'], night['brightdawn']]])
        nightprograms = np.concatenate([
            [tiles.CONDITION_INDEX['BRIGHT']],
            night['programs'][night['programs'] != -1],
            [tiles.CONDITION_INDEX['BRIGHT']]])
        if len(nightprograms) != len(nighttimes)-1:
            raise ValueError('number of program changes does not match '
                             'number of programs!')
        condind = np.interp(mjd0, nighttimes,
                            np.arange(len(nighttimes)))
        condition = nightprograms[condind.astype('i4')]
        conditions.append(condition)
    out = np.zeros(len(mjd), dtype='i4')
    out[:] = -1
    out[~mday & okmjd] = conditions
    # this program index is ~backwards from OBSCONDITIONS.
    newout = np.full(len(out), 'UNKNOWN', dtype='U8')
    for condition in tiles.CONDITIONS:
        m = out == tiles.CONDITION_INDEX[condition]
        newout[m] = condition
    return newout


def number_per_night(exps, nightly_donefrac_requirement=0.5):
    tiles = desisurvey.tiles.get_tiles()
    m = exps['EXPTIME'] > 30
    exps = exps[m]
    s = np.argsort(exps['TILEID'])
    out = np.zeros(len(np.unique(exps['TILEID'])),
                   dtype=[('TILEID', 'i4'), ('NEXP', 'i4'),
                          ('NNIGHT', 'f4')])
    programs = np.zeros(len(exps), dtype='U20')
    programs[:] = 'UNKNOWN'
    ind, mask = tiles.index(exps['TILEID'], return_mask=True)
    programs[mask] = tiles.tileprogram[ind[mask]]
    nomtimes = desisurvey.tiles.get_nominal_program_times(programs)
    efftimes = np.clip(exps['EFFTIME'], 0, np.inf)
    donefrac = efftimes / nomtimes
    for i, (f, l) in enumerate(subslices(exps['TILEID'][s])):
        ind = s[f:l]
        out['TILEID'][i] = exps['TILEID'][ind[0]]
        out['NEXP'][i] = np.sum(m)
        nights = exps['NIGHT'][ind]
        for night in np.unique(nights):
            m = nights == night
            totaldonefrac = np.sum(
                np.clip(donefrac[ind[m]], 0, np.inf))
            if totaldonefrac > nightly_donefrac_requirement:
                out['NNIGHT'][i] += 1
            else:
                out['NNIGHT'][i] += 0.1
                # still gets credit for having been started
    return out


def update_donefrac_from_offline(exps, offlinefn):
    offline = fits.getdata(offlinefn)
    tiles = desisurvey.tiles.get_tiles()
    tileprogram = tiles.tileprogram
    tileprograms = np.zeros(len(exps), dtype='U80')
    mt, me = desisurvey.utils.match(tiles.tileID, exps['TILEID'])
    tileprograms[:] = 'UNKNOWN'
    tileprograms[me] = [p.strip() for p in tileprogram[mt]]
    me, mo = desisurvey.utils.match(exps['EXPID'], offline['EXPID'])
    nomtimes, timetypes = desisurvey.tiles.get_nominal_program_times(
        tileprograms[me], return_timetypes=True)
    uofflineprograms = np.unique(tileprograms[me])
    unknown = []
    config = desisurvey.config.Configuration()
    for p in uofflineprograms:
        if p not in config.programs.keys:
            unknown.append(p)
    log.warning('Unknown programs '+', '.join(unknown)+', using DARK '
                'effective time.')
    offlinetimetypes = np.zeros(len(offline), dtype='U80')
    offlinetimetypes[:] = 'DARK'
    offlinetimetypes[mo] = timetypes
    offline_eff_time = np.where(offlinetimetypes == 'DARK',
                                offline['ELG_EFFTIME_DARK'],
                                offline['BGS_EFFTIME_BRIGHT'])
    if ((len(np.unique(exps['EXPID'])) != len(exps)) or
            (len(np.unique(offline['EXPID'])) != len(offline))):
        raise ValueError('weird duplicate EXPID in exps or offline')
    # offline has R_DEPTH in units of time
    # now we need the goal exposure times
    # these are just the nominal times
    exps = exps.copy()
    exps['EFFTIME_SPEC'][me] = offline_eff_time[mo]
    return exps


def update_mtldone(tiles, mtldonefn):
    mtldone = astropy.table.Table.read(mtldonefn)
    mt, mm = desisurvey.utils.match(tiles['TILEID'], mtldone['TILEID'])
    tiles = tiles.copy()
    tiles['MTL_DONE'] = False
    tiles['MTL_DONE'][mt] = True
    usedmtl = np.zeros(len(mtldone), dtype='i4')
    usedmtl[mm] = 1
    nunused = np.sum(usedmtl == 0)
    if nunused > 0:
        log.debug('MTLs completed for {} unknown tiles?'.format(nunused))
    return tiles


def read_exp(fn):
    """Read tile & exposure ETC statistics file from filename.

    This function works around some fits string issues in old versions
    of astropy.

    Parameters
    ----------
    fn : str

    Returns
    -------
    tiles, exps
    numpy arrays for the tile and exposure ETC files
    """
    return astropy.table.Table.read(fn)


def parse(options=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='Collect ETC statistics from spectra.',
        epilog='EXAMPLE: %(prog)s directory outfile')
    parser.add_argument('directory', type=str,
                        help='directory to scan for spectra')
    parser.add_argument('outfile', type=str,
                        help='file to write out')
    parser.add_argument('--start_from', type=str, default=None,
                        help='etc_stats file to start from')
    parser.add_argument('--offlinedepth', type=str,
                        help='offline depth file to use')
    parser.add_argument('--mtldone', type=str,
                        help='mtl done file to use')
    parser.add_argument('--config', type=str, default=None,
                        help='configuration file to use')
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def main(args):
    import desisurvey.config
    _ = desisurvey.config.Configuration(args.config)
    res = scan_directory(args.directory, start_from=args.start_from,
                         offlinedepth=args.offlinedepth,
                         mtldone=args.mtldone)
    if res is not None:
        tiles, exps = res
        write_exp(exps, args.outfile)
    else:
        raise ValueError('Could not collect ETC files.')
