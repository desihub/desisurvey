import os
import glob
import re
import numpy as np
from astropy.io import fits
import desiutil.log
import desisurvey.etc
import desisurvey.tiles
import desisurvey.ephem
from astropy import units as u


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


def scan_directory(dirname, simulate_donefrac=False, start_from=None,
                   offlinedepth=None):
    """Scan directory for spectra with ETC statistics to collect.

    Parameters
    ----------
    dirname : str
        directory path to scan.  All fits files under dirname are searched
        for ETC statistics.
        This needs to be updated to ~DESI files only, with more care given
        to where these keywords are actually found.
    simulate_donefrac : bool
        instead of tabulating DONEFRAC, tabulate EXPTIME / fac instead.
        this is useful when DONEFRAC is not being computed.
    start_from : str
        etc_stats file to start from.  Nights already in the etc_stats file
        will not be collected.  If a YYYMMDD string, look for etc_stats file
        in DESISURVEY_OUTPUT/YYYYMMDD/etc_stats-{YYYYMMDD}.fits
        "fresh" or None indicates starting fresh.
        "most_recent" indicates searching DESISURVEY_OUTPUT for the most recent
        file to restore.
    offlinedepth : str
        offline depth file to use.  Fills out donefracs according to
        R_DEPTH in the file, plus config.nominal_exposure_time
    """
    log.info('Scanning {} for desi exposures...'.format(dirname))
    if start_from is None or (start_from == "fresh"):
        files = glob.glob(os.path.join(dirname, '**/desi*.fits.fz'),
                          recursive=True)
        start_exps = None
    else:
        files = []
        subdirs = os.listdir(dirname)
        subdirs = np.sort(subdirs)[::-1]
        if start_from == 'most_recent':
            etcfns = glob.glob(os.path.join(os.environ['DESISURVEY_OUTPUT'],
                                            '**/etc-stats-*.fits'))
            yyyymmdd = []
            for fn in etcfns:
                rgx = re.compile('etc-stats-([0-9]{8}).fits')
                match = rgx.match(os.path.basename(fn))
                if match is not None:
                    yyyymmdd.append(int(match.groups(1)[0]))
                else:
                    yyyymmdd.append(-1)
            maxind = np.argmax(yyyymmdd)
            fn = etcfns[maxind]
            log.info('Restoring etc-stats from {}'.format(fn))
        elif os.path.exists(start_from):
            fn = start_from
        else:
            fn = os.path.join(os.environ['DESISURVEY_OUTPUT'], start_from,
                              'etc_stats-{}.fits'.format(start_from))
        if not os.path.exists(fn):
            log.error('Could not find file {} to start from!'.format(
                start_from))
            return
        start_tiles, start_exps = read_tile_exp(fn)
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
        ('EXPID', 'i4'), ('FILENAME', 'U200'),
        ('TILEID', 'i4'), ('DONEFRAC_EXP_ETC', 'f4'),
        ('EXPTIME', 'f4'), ('MJD_OBS', 'f8'), ('FLAVOR', 'U80')])
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
        exps['EXPID'][i] = hdr.get('EXPID', -1)
        exps['FILENAME'][i] = hdr.get('filename', fn)
        exps['TILEID'][i] = hdr.get('TILEID', -1)
        exps['DONEFRAC_EXP_ETC'][i] = hdr.get('DONEFRAC', -1)
        exps['EXPTIME'][i] = hdr.get('EXPTIME', -1)
        exps['MJD_OBS'][i] = hdr.get('MJD-OBS', -1)
        exps['FLAVOR'][i] = hdr.get('FLAVOR', 'none')
    flavors = np.array([f.upper().strip() for f in exps['FLAVOR']],
                       dtype=exps['FLAVOR'].dtype)
    m = (exps['TILEID'] == -1) | (flavors != 'SCIENCE')
    if np.any(m):
        log.info('Ignoring {} files due to weird FLAVOR or TILEID'.format(
            np.sum(m)))
        exps = exps[~m]
    if simulate_donefrac:
        tiles = desisurvey.tiles.get_tiles()
        program = np.full(len(exps), 'UNKNOWN', dtype='U80')
        ind, mask = tiles.index(exps['TILEID'], return_mask=True)
        program[mask] = tiles.tileprogram[ind[mask]]
        nomtime = get_nominal_program_times(program)
        airmass = np.ones(len(exps), dtype='f4')
        airmass[mask] = tiles.airmass_at_mjd(exps['MJD_OBS'][mask],
                                             mask=ind[mask])
        expfac = np.ones(len(exps), dtype='f4')
        expfac *= desisurvey.etc.airmass_exposure_factor(airmass)
        expfac[mask] *= tiles.dust_factor[ind[mask]]
        exps['DONEFRAC_EXP_ETC'] = exps['EXPTIME']/expfac/nomtime
    exps = exps[np.argsort(exps['EXPID'])]
    if start_exps is not None:
        exps = np.concatenate([start_exps, exps])
    if offlinedepth is not None:
        exps = update_donefrac_from_offline(exps, offlinedepth)
    ntiles = len(np.unique(exps['TILEID']))
    tiles = np.zeros(ntiles, dtype=[
        ('TILEID', 'i4'), ('DONEFRAC_ETC', 'f4'), ('EXPTIME', 'f4'),
        ('LASTEXPID_ETC', 'i4'), ('NOBS_ETC', 'i4'), ('LASTMJD_ETC', 'f8')])
    s = np.argsort(exps['TILEID'])
    for i, (f, l) in enumerate(subslices(exps['TILEID'][s])):
        ind = s[f:l]
        tiles['TILEID'][i] = exps['TILEID'][ind[0]]
        tiles['DONEFRAC_ETC'][i] = np.sum(np.clip(
            exps['DONEFRAC_EXP_ETC'][ind], 0, np.inf))
        tiles['EXPTIME'][i] = np.sum(np.clip(
            exps['EXPTIME'][ind], 0, np.inf))
        tiles['LASTEXPID_ETC'][i] = np.max(exps['EXPID'][ind])
        # potentially only want to consider "good" exposures here?
        tiles['NOBS_ETC'][i] = len(ind)
        tiles['LASTMJD_ETC'][i] = np.max(exps['MJD_OBS'][ind])
    return tiles, exps


def write_tile_exp(tiles, exps, fn):
    """Write tile & exposure ETC statistics from numpy objects.

    Parameters
    ----------
    tiles : array
        tile ETC statistics

    exps : array
        exposure ETC statistics
    """
    fits.writeto(fn, tiles, header=fits.Header(dict(EXTNAME='TILES')))
    fits.append(fn, exps, header=fits.Header(dict(EXTNAME='EXPS')))


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
    conditions: conditions mask, 1 for DARK, 2 for GRAY, 4 for BRIGHT
    -1 for problematic MJD (too early, NaN, or during the day)
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
            [tiles.PROGRAM_INDEX['BRIGHT']],
            night['programs'][night['programs'] != -1],
            [tiles.PROGRAM_INDEX['BRIGHT']]])
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
    newout = out.copy()
    for condition in tiles.OBSCONDITIONS:
        m = out == tiles.PROGRAM_INDEX[condition]
        newout[m] = tiles.OBSCONDITIONS[condition]
    return newout


def number_in_conditions(exps, nightly_donefrac_requirement=0.5):
    tiles = desisurvey.tiles.get_tiles()
    m = exps['EXPTIME'] > 30
    exps = exps[m]
    conditions = get_conditions(exps['MJD_OBS']+exps['EXPTIME']/2/60/60/24)
    s = np.argsort(exps['TILEID'])
    out = np.zeros(len(np.unique(exps['TILEID'])),
                   dtype=[('TILEID', 'i4'), ('NEXP_BRIGHT', 'i4'),
                          ('NEXP_GRAY', 'i4'), ('NEXP_DARK', 'i4'),
                          ('NNIGHT_BRIGHT', 'f4'), ('NNIGHT_GRAY', 'f4'),
                          ('NNIGHT_DARK', 'f4')])
    conddict = {ind: cond for cond, ind in tiles.OBSCONDITIONS.items()}
    for i, (f, l) in enumerate(subslices(exps['TILEID'][s])):
        ind = s[f:l]
        out['TILEID'][i] = exps['TILEID'][ind[0]]
        for cond in ['DARK', 'BRIGHT', 'GRAY']:
            m = conditions[ind] == tiles.OBSCONDITIONS[cond]
            out['NEXP_'+cond][i] = np.sum(m)
        # at Kitt Peak, a night never crosses an MJD boundary.
        # So we count nights by counting the number of unique
        # mjd integers.
        nights = exps['MJD_OBS'][ind].astype('i4')
        for night in np.unique(nights):
            m = nights == night
            totaldonefrac = np.sum(
                np.clip(exps['DONEFRAC_EXP_ETC'][ind[m]], 0, np.inf))
            cond = conditions[ind[m]]
            cond = cond[cond >= 0]
            if len(cond) == 0:
                # all images taken ~during the day, don't know what to do.
                continue
            cond = conddict[cond[0]]
            if totaldonefrac > nightly_donefrac_requirement:
                out['NNIGHT_'+cond][i] += 1
            else:
                out['NNIGHT_'+cond][i] += 0.1
                # still gets credit for having been started
    return out


def get_nominal_program_times(tileprogram):
    config = desisurvey.config.Configuration()
    cfgnomtimes = config.nominal_exposure_time
    nomtimes = []
    unknownprograms = []
    nunknown = 0
    for program in tileprogram:
        nomprogramtime = getattr(cfgnomtimes, program, 300)
        if not isinstance(nomprogramtime, int):
            nomprogramtime = nomprogramtime().to(u.s).value
        else:
            unknownprograms.append(program)
            nunknown += 1
        nomtimes.append(nomprogramtime)
    if nunknown > 0:
        log.info(('%d observations of unknown programs\n' % nunknown) +
                 'unknown programs: '+' '.join(np.unique(unknownprograms)))
    nomtimes = np.array(nomtimes)
    return nomtimes


def update_donefrac_from_offline(exps, offlinefn):
    offline = fits.getdata(offlinefn)
    tiles = desisurvey.tiles.Tiles()
    tileprogram = tiles.tileprogram
    tileprograms = np.zeros(len(exps), dtype='U80')
    mt, me = desisurvey.utils.match(tiles.tileID, exps['TILEID'])
    tileprograms[:] = 'UNKNOWN'
    tileprograms[me] = tileprogram[mt]
    me, mo = desisurvey.utils.match(exps['EXPID'], offline['EXPID'])
    nomtimes = get_nominal_program_times(tileprograms[me])
    try:
        offline_eff_time = offline['R_DEPTH_EBVAIR']
    except Exception:
        offline_eff_time = offline['R_DEPTH']
    if ((len(np.unique(exps['EXPID'])) != len(exps)) or
            (len(np.unique(offline['EXPID'])) != len(offline))):
        raise ValueError('weird duplicate EXPID in exps or offline')
    # offline has R_DEPTH in units of time
    # now we need the goal exposure times
    # these are just the nominal times
    exps = exps.copy()
    exps['DONEFRAC_EXP_ETC'][me] = offline_eff_time[mo]/nomtimes
    return exps


def read_tile_exp(fn):
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
    tiles = fits.getdata(fn, 'TILES')
    exps = fits.getdata(fn, 'EXPS')
    return convert_fits(tiles), convert_fits(exps)


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
    parser.add_argument('--simulate_donefrac', action='store_true',
                        help='use exptime/fac instead of DONEFRAC')
    parser.add_argument('--offlinedepth', type=str,
                        help='offline depth file to use')
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
                         simulate_donefrac=args.simulate_donefrac,
                         offlinedepth=args.offlinedepth)
    if res is not None:
        tiles, exps = res
        write_tile_exp(tiles, exps, args.outfile)
    else:
        raise ValueError('Could not collect ETC files.')
