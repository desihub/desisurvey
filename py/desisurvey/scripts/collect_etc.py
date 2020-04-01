import os
import glob
import numpy as np
from astropy.io import fits
import desiutil.log

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
    maxexpid = np.max(start_from['expid'])
    return files[expid > maxmatch]


def scan_directory(dirname, simulate_donefrac=False, start_from=None):
    """Scan directory for spectra with ETC statistics to collect.

    Parameters
    ----------
    dirname : str
        directory path to scan.  All fits files under dirname are searched
        for ETC statistics.
        This needs to be updated to ~DESI files only, with more care given
        to where these keywords are actually found.
    simulate_donefrac : bool
        instead of tabulating DONEFRAC, tabulate EXPTIME / 1000 instead.
        this is useful when DONEFRAC is not being computed.
    start_from : str
        etc_stats file to start from.  Nights already in the etc_stats file
        will not be collected.  If a YYYMMDD string, look for etc_stats file
        in DESISURVEY_OUTPUT/YYYYMMDD/etc_stats-{YYYYMMDD}.fits
    """
    log.info('Scanning {} for desi exposures...'.format(dirname))
    files = glob.glob(os.path.join(dirname, '**/desi*.fits.fz'),
                      recursive=True)
    if start_from is not None:
        try:
            fn = os.path.join(os.environ['DESISURVEY_OUTPUT'], start_from,
                              'etc_stats-{}.fits'.format(start_from))
            start_exps = fits.getdata(fn, 'EXPS')
        except:
            try:
                start_exps = fits.getdata(start_from, 'EXPS')
            except:
                log.error('Could not find file {} to start from!'.format(
                    start_from))
                return
        files = cull_old_files(files, start_exps)
    else:
        start_exps = None
    
    log.info('Found {} files, extracting header information...'.format(
        len(files)))
    exps = np.zeros(len(files), dtype=[
        ('EXPID', 'i4'), ('FILENAME', 'U200'),
        ('TILEID', 'f4'), ('DONEFRAC_EXP_ETC', 'f4'),
        ('EXPTIME', 'f4'), ('MJD_OBS', 'f4'), ('FLAVOR', 'U80')])
    for i, fn in enumerate(files):
        if (i % 1000) == 0:
            log.info('Extracting headers from file {} of {}...'.format(
                i+1, len(files)))
        hdr = {}
        for ename in ['SPEC', 'SPS']:
            try:
                hdr = fits.getheader(fn, ename)
                break
            except:
                continue
        exps['EXPID'][i] = hdr.get('EXPID', -1)
        exps['FILENAME'][i] = hdr.get('filename', fn)
        exps['TILEID'][i] = hdr.get('TILEID', -1)
        if not simulate_donefrac:
            exps['DONEFRAC_EXP_ETC'][i] = hdr.get('DONEFRAC', -1)
        else:
            exps['DONEFRAC_EXP_ETC'][i] = hdr.get('EXPTIME', -1)/1000
        exps['EXPTIME'][i] = hdr.get('EXPTIME', -1)
        exps['MJD_OBS'][i] = hdr.get('MJD-OBS', -1)
        exps['FLAVOR'][i] = hdr.get('FLAVOR', 'none')
    flavors = np.array([f.upper().strip() for f in exps['FLAVOR']])
    m = (exps['TILEID'] == -1) | (flavors != 'SCIENCE')
    if np.any(m):
        log.info('Ignoring {} files due to weird FLAVOR or TILEID'.format(
            np.sum(m)))
        exps = exps[~m]
    ntiles = len(np.unique(exps['TILEID']))
    tiles = np.zeros(ntiles, dtype=[
        ('TILEID', 'i4'), ('DONEFRAC_ETC', 'f4'), ('EXPTIME', 'f4'),
        ('LASTEXPID_ETC', 'i4'), ('NOBS_ETC', 'i4'), ('LASTMJD_ETC', 'f4')])
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
    exps = exps[np.argsort(exps['EXPID'])]
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
    from astropy.io import fits
    fits.writeto(fn, tiles, header=fits.Header(dict(EXTNAME='TILES')))
    fits.append(fn, exps, header=fits.Header(dict(EXTNAME='EXPS')))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Collect ETC statistics from spectra.',
        epilog='EXAMPLE: %(prog)s directory outfile')
    parser.add_argument('directory', type=str,
                        help='directory to scan for spectra')
    parser.add_argument('outfile', type=str,
                        help='file to write out')
    parser.add_argument('--start_from', type=str, defalt=None,
                        help='etc_stats file to start from')
    args = parser.parse_args()
    tiles, exps = scan_directory(args.directory)
    write_tile_exp(tiles, exps, args.outfile)
