import os
import glob
import numpy
from astropy.io import fits


class subslices:
    "Iterator for looping over subsets of an array"
    def __init__(self, data, uind=None):
        if uind is None:
            self.uind = numpy.unique(data, return_index=True)[1]
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


def scan_directory(dirname):
    """Scan directory for spectra with ETC statistics to collect.

    Parameters
    ----------
    dirname : str
        directory path to scan.  All fits files under dirname are searched
        for ETC statistics.
        This needs to be updated to ~DESI files only, with more care given
        to where these keywords are actually found.
    """
    files = list(glob.iglob(os.path.join(dirname, '**/*.fits'),
                            recursive=True))
    exps = numpy.zeros(len(files), dtype=[
        ('EXPID', 'i4'), ('FILENAME', 'U80'),
        ('TILEID', 'f4'), ('DONEFRAC_EXP_ETC', 'f4'),
        ('EXPTIME', 'f4'), ('MJD_OBS', 'f4'), ('FLAVOR', 'U80')])
    for i, fn in enumerate(files):
        hdr = fits.getheader(fn)
        exps['EXPID'][i] = hdr.get('EXPID', -1)
        exps['FILENAME'][i] = os.path.basename(fn)
        exps['TILEID'][i] = hdr.get('TILEID', -1)
        exps['DONEFRAC_EXP_ETC'][i] = hdr.get('DONEFRAC', -1)
        exps['EXPTIME'][i] = hdr.get('EXPTIME', -1)
        exps['MJD_OBS'][i] = hdr.get('MJD_OBS', -1)
        exps['FLAVOR'][i] = hdr.get('FLAVOR', 'none')
    flavors = numpy.array([f.upper().strip() for f in exps['FLAVOR']])
    m = (exps['TILEID'] == -1) | (flavors != 'SCIENCE')
    if numpy.any(m):
        print('Ignoring %d files due to weird FLAVOR or TILEID' % numpy.sum(m))
        exps = exps[~m]
    ntiles = len(numpy.unique(exps['TILEID']))
    tiles = numpy.zeros(ntiles, dtype=[
        ('TILEID', 'i4'), ('DONEFRAC_ETC', 'f4'), ('EXPTIME', 'f4'),
        ('LASTEXPID_ETC', 'f4'), ('NOBS_ETC', 'i4'), ('LASTMJD_ETC', 'f4')])
    s = numpy.argsort(exps['TILEID'])
    for i, (f, l) in enumerate(subslices(exps['TILEID'][s])):
        ind = s[f:l]
        tiles['TILEID'][i] = exps['TILEID'][ind[0]]
        tiles['DONEFRAC_ETC'][i] = numpy.sum(exps['DONEFRAC_EXP_ETC'][ind])
        tiles['EXPTIME'][i] = numpy.sum(exps['EXPTIME'][ind])
        tiles['LASTEXPID_ETC'][i] = numpy.max(exps['EXPID'][ind])
        tiles['NOBS_ETC'][i] = len(ind)
        tiles['LASTMJD_ETC'][i] = numpy.max(exps['MJD_OBS'][ind])
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
    args = parser.parse_args()
    tiles, exps = scan_directory(args.directory)
    write_tile_exp(tiles, exps, args.outfile)
