import os
import glob
import numpy as np
from desimodel import focalplane
from astropy.io import fits
from desitarget import targetmask
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table


basetiledtype = [
    ('TILEID', 'i4'), ('RA', 'f8'), ('DEC', 'f8'), ('PASS', 'i4'),
    ('IN_DESI', 'bool'), ('EBV_MED', 'f4'), ('AIRMASS', 'f4'),
    ('STAR_DENSITY', 'f4'), ('EXPOSEFAC', 'f4'),
    ('PROGRAM', 'U20'), ('OBSCONDITIONS', 'i4')]
addtiledtype = [
    ('BRIGHTRA', '3f8'), ('BRIGHTDEC', '3f8'), ('BRIGHTVTMAG', '3f4'),
    ('CENTERID', 'i4'), ('IMAGEFRAC_G', 'f4'), ('IMAGEFRAC_R', 'f4'),
    ('IMAGEFRAC_Z', 'f4'), ('IMAGEFRAC_GR', 'f4'), ('IMAGEFRAC_GRZ', 'f4'),
    ('IN_IMAGING', 'f4')]


def match2d(x1, y1, x2, y2, rad):
    """Find all matches between x1, y1 and x2, y2 within radius rad."""
    from scipy.spatial import cKDTree
    xx1 = np.stack([x1, y1], axis=1)
    xx2 = np.stack([x2, y2], axis=1)
    tree1 = cKDTree(xx1)
    tree2 = cKDTree(xx2)
    res = tree1.query_ball_tree(tree2, rad)
    lens = [len(r) for r in res]
    m1 = np.repeat(np.arange(len(x1), dtype='i4'), lens)
    if sum([len(r) for r in res]) == 0:
        m2 = m1.copy()
    else:
        m2 = np.concatenate([r for r in res if len(r) > 0])
    d12 = np.sqrt(np.sum((xx1[m1, :]-xx2[m2, :])**2, axis=1))
    return m1, m2, d12


def lb2uv(r, d):
    return tp2uv(*lb2tp(r, d))


def uv2lb(uv):
    return tp2lb(*uv2tp(uv))


def uv2tp(uv):
    norm = np.sqrt(np.sum(uv**2., axis=1))
    uv = uv / norm.reshape(-1, 1)
    t = np.arccos(uv[:, 2])
    p = np.arctan2(uv[:, 1], uv[:, 0])
    return t, p


def lb2tp(l, b):
    return (90.-b)*np.pi/180., l*np.pi/180.


def tp2lb(t, p):
    return p*180./np.pi % 360., 90.-t*180./np.pi


def tp2uv(t, p):
    z = np.cos(t)
    x = np.cos(p)*np.sin(t)
    y = np.sin(p)*np.sin(t)
    return np.concatenate([q[..., np.newaxis] for q in (x, y, z)],
                          axis=-1)


def match_radec(r1, d1, r2, d2, rad=1./60./60., nneighbor=0, notself=False):
    """Match r1, d1, to r2, d2, within radius rad."""
    if notself and nneighbor > 0:
        nneighbor += 1
    uv1 = lb2uv(r1, d1)
    uv2 = lb2uv(r2, d2)
    from scipy.spatial import cKDTree
    tree = cKDTree(uv2)
    dub = 2*np.sin(np.radians(rad)/2)
    if nneighbor > 0:
        d12, m2 = tree.query(uv1, nneighbor, distance_upper_bound=dub)
        if nneighbor > 1:
            m2 = m2.reshape(-1)
            d12 = d12.reshape(-1)

        m1 = np.arange(len(r1)*nneighbor, dtype='i4') // nneighbor
        d12 = 2*np.arcsin(np.clip(d12 / 2, 0, 1))*180/np.pi
        m = m2 < len(r2)
    else:
        tree1 = cKDTree(uv1)
        res = tree.query_ball_tree(tree1, dub)
        lens = [len(r) for r in res]
        m2 = np.repeat(np.arange(len(r2), dtype='i4'), lens)
        res = [r for r in res if len(r) > 0]
        if len(res) > 0:
            m1 = np.concatenate(res)
        else:
            m1 = np.zeros(0, dtype='i4')
        d12 = gc_dist(r1[m1], d1[m1], r2[m2], d2[m2])
        m = np.ones(len(m1), dtype='bool')
    if notself:
        m = m & (m1 != m2)
    return m1[m], m2[m], d12[m]


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

    def next(self):
        if self.ind == len(self.uind):
            raise StopIteration
        if self.ind == len(self.uind)-1:
            last = self.length
        else:
            last = self.uind[self.ind+1]
        first = self.uind[self.ind]
        self.ind += 1
        return first, last

    def __next__(self):
        return self.next()


def render(ra, dec, tilera, tiledec, fiberposfile=None, oneperim=False,
           excludebad=False):
    """Return number of possible observations of ra, dec, given focal
    plane centers tilera, tiledec."""
    out = np.zeros_like(ra, dtype='i4')
    mg, mt, dgt = match_radec(ra, dec, tilera, tiledec, 1.65)
    s = np.argsort(mt)
    if fiberposfile is None:
        fiberposfile = os.path.join(os.environ['DESIMODEL'], 'data',
                                    'focalplane', 'fiberpos.fits')
    fpos = fits.getdata(fiberposfile)
    if excludebad:
        import desimodel.io
        _, _, state, _ = desimodel.io.load_focalplane()
        _, ms, mf = np.intersect1d(state['LOCATION'], fpos['LOCATION'],
                                   return_indices=True)
        keep = np.zeros(len(fpos), dtype='bool')
        keep[mf] = state['STATE'][ms] == 0
        fpos = fpos[keep]
    for f, l in subslices(mt[s]):
        tileno = mt[s[f]]
        ind = mg[s[f:l]]
        x, y = focalplane.radec2xy(tilera[tileno], tiledec[tileno],
                                   ra[ind], dec[ind])
        mx, mf, dxf = match2d(x, y, fpos['x'], fpos['y'], 6)
        if oneperim:
            mx = np.unique(mx)
        # much slower than my custom-rolled version!
        out += np.bincount(ind[mx], minlength=len(out))
    return out


def render_simple(ra, dec, tilera, tiledec):
    out = np.zeros_like(ra, dtype='i4')
    mg, mt, dgt = match_radec(ra, dec, tilera, tiledec, 1.63)
    out += np.bincount(mg, minlength=len(out))
    return out


def adjacency_matrix(tilera, tiledec, fiberposfile=None):
    """Overlap area matrix between slit blocks and radial bins, given
    tile ras and decs."""
    # compute x & y on each tile to ra & dec
    # match ras and decs together at some fiducial size
    # (this ignores ellipticity, etc., but the slite blocks are pretty big)
    #
    if fiberposfile is None:
        fiberposfile = os.path.join(os.environ['DESIMODEL'], 'data',
                                    'focalplane', 'fiberpos.fits')
    fpos = fits.getdata(fiberposfile)
    # really slow, not vectorized.
    pos = [[focalplane.xy2radec(tra, tdec, fx, fy)
            for fx, fy in zip(fpos['x'], fpos['y'])]
           for tra, tdec in zip(tilera, tiledec)]
    pos = np.array(pos)
    ras = pos[:, :, 0].ravel()
    decs = pos[:, :, 1].ravel()
    slitno = np.tile(fpos['slitblock']+fpos['petal']*20, len(tilera))
    radbin = np.floor(np.sqrt(fpos['x']**2+fpos['y']**2)/20).astype('i4')
    radbin = np.tile(radbin, len(tilera))
    expnum = np.repeat(np.arange(len(tilera)), len(fpos))
    rad = 1.4/60
    m1, m2, d12 = match_radec(ras, decs, ras, decs, rad,
                              notself=True)
    m = expnum[m1] != expnum[m2]
    m1 = m1[m]
    m2 = m2[m]
    d12 = d12[m]
    # area of intersection of two equal-size circles?
    # area: 2r^2 arccos(d/2r)-0.5 d sqrt((2r-d)(2r+d))
    area = (2*rad**2*np.arccos(d12/2./rad) -
            0.5*d12*np.sqrt((2*rad-d12)*(2*rad+d12)))
    nslitno = np.max(slitno)+1
    nradbin = np.max(radbin)+1
    adj = np.zeros(nslitno**2, dtype='f4')
    adjr = np.zeros(nradbin**2, dtype='f4')
    ind = slitno[m1]*nslitno+slitno[m2]
    indr = radbin[m1]*nradbin+radbin[m2]
    adj += np.bincount(ind, weights=area[m1], minlength=len(adj))
    adj = adj.reshape(nslitno, nslitno)
    adjr += np.bincount(indr, weights=area[m1], minlength=len(adjr))
    adjr = adjr.reshape(nradbin, nradbin)
    return adj, adjr


def simpleradecoffscheme(ras, decs, dx=0.6, ang=42):
    """Box ra and dec scheme, given a base tiling.

    Dithers the base tiling by fixed offsets in ra/cos(dec) and dec.  Initial
    dither direction is given by ang.  Dithers are dx in length, and
    the dither direction is rotated 90 degrees after each dither.  The
    fifth dither is placed at the center of the square formed by the four
    previous dithers.  This final set of five dithers is then duplicated,
    to make 10 dithers.

    Args:
       ras (ndarray[N]): base tile ras
       decs (ndarray[N]): base tile decs
       dx (float, degrees): amount to dither
       ang (float, degrees): position angle of initial dither

    Returns:
       ras (ndarray[N*10]): dithered tile ras
       decs (ndarray[N*10]): dithered tile decs
    """
    # take a single covering
    # define 4 sets of offsets
    # start with something minimal: need to cover:
    # central bullseyes: 0.2 deg
    # GFA gaps: up to 0.4 deg

    from numpy import sin, cos
    ang = np.radians(ang)
    dang = np.pi/2
    dithers = [[0, 0],
               [dx*sin(ang+0*dang), dx*cos(ang+0*dang)],
               [dx*sin(ang+1*dang), dx*cos(ang+1*dang)],
               [dx*sin(ang+2*dang), dx*cos(ang+2*dang)]]
    dithers = np.cumsum(np.array(dithers), axis=0)
    dithers = list(dithers) + [[np.mean([d[0] for d in dithers]),
                                np.mean([d[1] for d in dithers])]]
    fac = 1./np.cos(np.radians(decs))
    fac = np.clip(fac, 1, 360*5)  # confusion near celestial pole.
    newras = np.concatenate([ras+d[0]*fac for d in dithers])
    newdecs = np.concatenate([decs+d[1] for d in dithers])
    newdecs = np.clip(newdecs, -np.inf, 90.)
    newras = newras % 360
    newras = np.concatenate([newras, newras])
    newdecs = np.concatenate([newdecs, newdecs])
    return newras, newdecs


def logradecoffscheme(ras, decs, dx=0.6, ang=24, verbose=False,
                      firstyearoptimized=True):
    """Log spiraly ra and dec dither scheme, given a base tiling.

    Dithers the base tiling by offsets in ra/cos(dec) and dec, by
    increasing amounts.  Initial dither direction is given by ang of
    length dx.  Subsequent dithers are rotated by 90 degrees and
    increased in length by a factor of exp(1/3).  The fifth dither is
    placed at the center of the quadrilateral formed by the four
    previous dithers.  This final set of five dithers is then
    duplicated, to make 10 dithers.

    Args:
       ras (ndarray[N]): base tile ras
       decs (ndarray[N]): base tile decs
       dx (float, degrees): amount to dither
       ang (float, degrees): position angle of initial dither
       firstyearoptimized (bool): optimize dither order for first year

    Returns:
       ras (ndarray[N*10]): dithered tile ras
       decs (ndarray[N*10]): dithered tile decs
    """
    dx = dx*np.exp(np.arange(4)/3.)

    from numpy import sin, cos
    ang = np.radians(ang)
    dang = np.pi/2
    dithers = [[0, 0],
               [dx[0]*sin(ang+0*dang), dx[0]*cos(ang+0*dang)],
               [dx[1]*sin(ang+1*dang), dx[1]*cos(ang+1*dang)],
               [dx[2]*sin(ang+2*dang), dx[2]*cos(ang+2*dang)]]
    dithers = np.cumsum(np.array(dithers), axis=0)
    dithers -= np.mean(dithers, axis=0).reshape(1, -1)
    dithers = [[0, 0]] + list(dithers)
    if verbose:
        for dra, ddec in dithers:
            print(r'%6.3f  &  %6.3f \\' % (dra, ddec))
    fac = 1./np.cos(np.radians(decs))
    fac = np.clip(fac, 1, 360*5)  # confusion near celestial pole.
    newras = np.concatenate([ras+d[0]*fac for d in dithers])
    newdecs = np.concatenate([decs+d[1] for d in dithers])
    m = newdecs > 90
    newdecs[m] = 90-(newdecs[m]-90)
    newras[m] += 180.
    m = newdecs < -90
    newdecs[m] = -90+(-90-newdecs[m])
    newras[m] += 180.
    if np.any((newdecs > 90) | (newdecs < -90)):
        raise ValueError('Something is wrong!')
    newras = newras % 360
    newras2 = np.concatenate([
            newras[len(ras):], newras[:len(ras)]])
    newdecs2 = np.concatenate([
            newdecs[len(ras):], newdecs[:len(ras)]])
    # for duplicate 5 passes, change order slightly.
    # the zeroth and ninth passes are now the passes that are at the centers
    # of the other 4 passes.  This makes passes 1-4 and passes 5-7 somewhat
    # better optimized for complete coverage.
    newras = np.concatenate([newras, newras2])
    newdecs = np.concatenate([newdecs, newdecs2])
    if firstyearoptimized:
        permute = {0: 2, 1: 3, 2: 4, 3: 0, 4: 1}
        rasyr1 = newras.copy()
        decsyr1 = newdecs.copy()
        ntile = len(ras)
        for npass, opass in permute.items():
            rasyr1[npass*ntile:(npass+1)*ntile] = (
                newras[opass*ntile:(opass+1)*ntile])
            decsyr1[npass*ntile:(npass+1)*ntile] = (
                newdecs[opass*ntile:(opass+1)*ntile])
        newras = rasyr1
        newdecs = decsyr1
    return newras, newdecs


def gc_dist(lon1, lat1, lon2, lat2):
    from numpy import sin, cos, arcsin, sqrt

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    return np.degrees(
        2*arcsin(sqrt((sin((lat1-lat2)*0.5))**2 +
                      cos(lat1)*cos(lat2)*(sin((lon1-lon2)*0.5))**2)))


def qa(desitiles, nside=1024, npts=1000, compare=False,
       npass=5, makenew=True, oneperim=False):
    """Make tiling QA plots; demonstrate usage."""
    import healpy
    theta, phi = healpy.pix2ang(nside, np.arange(12*nside**2))
    la, ba = phi*180./np.pi, 90-theta*180./np.pi
    if (('CENTERID' in desitiles.dtype.names) and
            np.any(desitiles['TILEID'] > 0)):
        m0 = desitiles['CENTERID'] == desitiles['TILEID']
        m5pass = (desitiles['PASS'] < npass)
    else:
        m0 = (desitiles['PASS'] == 0) & (desitiles['PROGRAM'] == 'dark')
        m5pass = ((desitiles['PASS'] < npass) &
                  (desitiles['PROGRAM'] == 'dark'))
    if makenew:
        ran, decn = logradecoffscheme(desitiles['RA'][m0],
                                      desitiles['DEC'][m0], dx=0.6, ang=24)
    else:
        ran, decn = desitiles['RA'], desitiles['DEC']
    tilerd = {}
    if compare:
        tilerd['default'] = (desitiles['RA'], desitiles['DEC'])
    tilerd['Tiles v3'] = (ran, decn)
    ims = {name: render(la, ba, rd[0][m5pass], rd[1][m5pass],
                        oneperim=oneperim) for name, rd in tilerd.items()}
    pseudoindesi = ((gc_dist(la, ba, 180, 30) < 40)
                    | (gc_dist(la, ba, 0, 5) < 30))
    from matplotlib import pyplot as p

    p.figure('One Footprint')
    setup_print((5, 4), scalefont=1.2)
    p.subplots_adjust(left=0.15, bottom=0.15)
    delt = 1.8
    dg, rg = np.meshgrid(np.linspace(-delt, delt, npts),
                         np.linspace(-delt, delt, npts))
    dpts = 4./(npts - 1)
    p.clf()
    tim = render(rg.ravel(), dg.ravel(), np.zeros(1), np.zeros(1),
                 oneperim=oneperim)
    tim = tim.reshape(rg.shape)
    p.imshow(tim.T, cmap='binary', origin='lower',
             extent=[-delt-dpts/2, delt+dpts/2, -delt-dpts/2, delt+dpts/2])
    p.gca().set_aspect('equal')
    p.xlabel(r'$\alpha$ ($\degree$)')
    p.ylabel(r'$\delta$ ($\degree$)')
    p.savefig('onefootprint.png', dpi=200)
    p.savefig('onefootprint.pdf', dpi=200)

    p.figure('One Center')
    delt = 2.3
    setup_print((5, 4), scalefont=1.2)
    p.subplots_adjust(left=0.15, bottom=0.15)
    zzind = np.argmin(
        gc_dist(desitiles['RA'][m0], desitiles['DEC'][m0], 0, 0))
    monecen = (m5pass &
               (desitiles['CENTERID'] == desitiles['CENTERID'][m0][zzind]))
    xcen = desitiles['RA'][m0][zzind]
    ycen = desitiles['DEC'][m0][zzind]
    dg, rg = np.meshgrid(np.linspace(-delt+ycen, delt+ycen, npts),
                         np.linspace(-delt+xcen, delt+xcen, npts))
    dpts = 4./(npts - 1)
    p.clf()
    tim = render(rg.ravel(), dg.ravel(), ran[monecen], decn[monecen],
                 oneperim=oneperim)
    tim = tim.reshape(rg.shape)
    p.imshow(tim.T, cmap='binary', origin='lower',
             extent=[-delt-dpts/2+xcen, delt+dpts/2+xcen,
                     -delt-dpts/2+ycen, delt+dpts/2+ycen])
    p.scatter(((ran[monecen]+180) % 360)-180, decn[monecen],
              c=desitiles['PASS'][monecen])
    p.xlim((-delt+xcen, delt+xcen))
    p.ylim((-delt+ycen, delt+ycen))
    p.gca().set_aspect('equal')
    p.xlabel(r'$\alpha$ ($\degree$)')
    p.ylabel(r'$\delta$ ($\degree$)')
    p.savefig('onecenter.png', dpi=200)
    p.savefig('onecenter.pdf', dpi=200)

    p.figure('Several Footprints')
    setup_print((5, 4), scalefont=1.2, )
    p.subplots_adjust(left=0.15, bottom=0.15)
    delt = 10
    dg, rg = np.meshgrid(np.linspace(-delt, delt, npts),
                         np.linspace(-delt, delt, npts))
    dpts = 4./(npts - 1)
    p.clf()
    tim = render(rg.ravel(), dg.ravel(), ran[m0], decn[m0], oneperim=oneperim)
    tim = tim.reshape(rg.shape)
    p.imshow(tim.T, cmap='binary', origin='lower',
             extent=[-delt-dpts/2, delt+dpts/2, -delt-dpts/2, delt+dpts/2],
             vmin=0, vmax=3)
    p.plot(((ran[m0]+180.) % 360)-180, decn[m0], 'r+')
    p.xlim(-delt, delt)
    p.ylim(-delt, delt)
    p.gca().set_aspect('equal')
    p.xlabel(r'$\alpha$ ($\degree$)')
    p.ylabel(r'$\delta$ ($\degree$)')
    p.savefig('onepass.png', dpi=200)
    p.savefig('onepass.pdf', dpi=200)

    p.figure('Full Sky')
    setup_print((8, 5), scalefont=1.2)
    p.subplots_adjust(left=0.125, bottom=0.1)
    p.clf()
    tim, xx, yy = heal2cart(ims['Tiles v3'], interp=False, return_pts=True)
    p.imshow(tim, cmap='binary', origin='lower', extent=[360, 0, -90, 90],
             vmin=0, vmax=9)
    p.gca().set_aspect('equal')
    p.xlabel(r'$\alpha$ ($\degree$)')
    p.ylabel(r'$\delta$ ($\degree$)')
    p.savefig('allpass.png', dpi=200)
    p.savefig('allpass.pdf', dpi=200)

    p.figure('Coverage Histogram')
    setup_print((5, 4), scalefont=1.2)
    p.subplots_adjust(left=0.15, bottom=0.15)
    p.clf()
    for name, im in ims.items():
        p.hist(im[pseudoindesi].reshape(-1), range=[-0.5, 15.5], bins=16,
               histtype='step', label=name, density=True)
        print(r'# coverings  & fraction of area \\')
        for count in range(16):
            frac = (np.sum(im[pseudoindesi] == count) /
                    1./np.sum(pseudoindesi))
            print(r'%4d  &  %7.3e \\' % (count, frac))
    p.xlim(-0.5, 8.5)
    p.ylim(1e-7, 1e0)
    p.xlabel('# of coverings')
    p.ylabel('fraction of area')
    p.gca().set_yscale('log')
    if len(ims) > 1:
        p.legend()
    p.savefig('covhist.png', dpi=200)
    p.savefig('covhist.pdf', dpi=200)

    adjs = {}
    for name, rd in tilerd.items():
        # just build adjacency matrix from small region, but needs
        # to at least cover several pointings
        m = m5pass & (gc_dist(rd[0], rd[1], 0, 0) < 6.)
        adjs[name] = adjacency_matrix(rd[0][m], rd[1][m])

    p.figure('Adjacency Matrices')
    setup_print((8, 5), scalefont=1.2)
    p.subplots_adjust(left=0.125, bottom=0.1)
    p.clf()
    for i, (name, tadj) in enumerate(adjs.items()):
        p.subplot(len(adjs), 2, 2*i+1)
        p.imshow(tadj[0], origin='lower', aspect='equal',
                 cmap='binary', vmax=0.2)
        p.title('%s: slit block' % name)
        p.xlabel('slit block #')
        p.subplot(len(adjs), 2, 2*i+2)
        # scale by area of each bin
        angsizes = np.arange(21).astype('f4')+0.5
        angsizes = angsizes.reshape(1, -1)*angsizes.reshape(-1, 1)
        p.imshow(tadj[1]/angsizes, origin='lower', aspect='equal',
                 cmap='binary', vmax=0.1, extent=[0, 400, 0, 400])
        p.title('%s: radial' % name)
        p.xlabel('radius (mm)')
    p.savefig('adjmatrix.png', dpi=200)
    p.savefig('adjmatrix.pdf', dpi=200)


def add_info_fields(tiles, gaiadensitymapfile,
                    tycho2file, covfile):
    import healpy
    nside = 512
    theta, phi = healpy.pix2ang(nside, np.arange(12*nside**2))
    la, ba = phi*180./np.pi, 90-theta*180./np.pi
    try:
        from desiutil import dust
        ebva = dust.ebv(la, ba, frame='galactic',
                        mapdir=os.getenv('DUST_DIR')+'/maps', scaling=1)
    except Exception:
        import dust
        ebva = dust.getval(la, ba, map='sfd')
    if isinstance(tiles, Table):
        ra = tiles['RA'].data
        dec = tiles['DEC'].data
    else:
        ra = tiles['RA']
        dec = tiles['DEC']
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg,
                     frame='icrs')
    coordgal = coord.galactic
    lt, bt = coordgal.l.value, coordgal.b.value
    uvt = lb2uv(lt, bt)
    gaiadens = fits.getdata(gaiadensitymapfile).copy()
    cov = fits.getdata(covfile).copy()
    fprad = 1.605
    for i in range(len(tiles)):
        ind = healpy.query_disc(nside, uvt[i], fprad*np.pi/180.)
        tiles['EBV_MED'][i] = np.median(ebva[ind])
        tiles['STAR_DENSITY'][i] = np.median(gaiadens[ind])

    brightstars = fits.getdata(tycho2file)
    mb, mt, dbt = match_radec(brightstars['RA'], brightstars['DEC'],
                              ra, dec, fprad)
    s = np.lexsort((brightstars['VTMAG'][mb], mt))
    mt, mb, dbt = mt[s], mb[s], dbt[s]
    add = np.zeros(len(tiles), dtype=addtiledtype)
    add['BRIGHTVTMAG'] = 999.
    for f, l in subslices(mt):
        end = np.clip(l-f, 0, 3)
        l = np.clip(l, f, f+3)
        ind = mt[f]
        add['BRIGHTRA'][ind, 0:end] = brightstars['ra'][mb[f:l]]
        add['BRIGHTDEC'][ind, 0:end] = brightstars['dec'][mb[f:l]]
        add['BRIGHTVTMAG'][ind, 0:end] = brightstars['vtmag'][mb[f:l]]

    uvt = lb2uv(ra, dec)
    for i in range(len(tiles)):
        ind = healpy.query_disc(nside, uvt[i], fprad*np.pi/180.)
        for f in ['G', 'R', 'Z', 'GR', 'GRZ']:
            val = np.mean(cov['%s_COVERAGE' % f][ind])
            add['IMAGEFRAC_%s' % f][i] = val
    add['IN_IMAGING'] = add['IMAGEFRAC_GRZ'] > 0.9
    return add


def maketilefile(desitiles, gaiadensitymapfile, tycho2file, covfile,
                 firstyearoptimized=True):
    """Make tile file.

    Args:
        desitiles: original DESI tile file
        gaiadensitymapfile: file name of healpix map of density of Gaia
            stars brighter than 19th mag.
        tycho2file: file name of list of ra, dec, bt, vt mags of Tycho-2
            stars.
        covfile: file name of healpix coverage maps
        firstyearoptimized: bool, use scheme optimized for early full depth
          coverage.
    """
    m0 = desitiles['PASS'] == 0
    ran, decn = logradecoffscheme(desitiles['RA'][m0],
                                  desitiles['DEC'][m0], dx=0.6, ang=24,
                                  firstyearoptimized=firstyearoptimized)
    # stupid way to make copy of an array, but most other things I tried
    # ended up with the dtype of desitilesnew being a reference to the dtype
    # of desitiles, which I didn't want.
    # dtype munging below needed to make sure that we get things as proper
    # python unicode strings rather than byte arrays, though this approach
    # is fragile.
    dtype = []
    for name, dt in desitiles.dtype.descr:
        if name != 'PROGRAM':
            dtype.append((name, dt))
        else:
            dtype.append((name, 'U6'))
    desitilesnew = np.zeros(len(desitiles), dtype=dtype)
    for n in desitilesnew.dtype.names:
        desitilesnew[n] = desitiles[n]
    desitilesnew.dtype.names = [n.lower() for n in desitilesnew.dtype.names]
    desitilesnew['RA'] = ran
    desitilesnew['DEC'] = decn
    cenpass = 3 if firstyearoptimized else 0
    mc = desitilesnew['PASS'] == cenpass
    # pass 0: centers in new scheme, no first year permutation
    # pass 3: centers in new scheme, if permuted to get more full depth in
    # first year
    desitilesnew['IN_DESI'] = np.concatenate(
        [desitilesnew['IN_DESI'][mc]]*10)
    # just repeat identically for each pass; all passes are close to
    # pass = 0 'centers'.

    desitilesnew_add = add_info_fields(desitilesnew, gaiadensitymapfile,
                                       tycho2file, covfile)
    from numpy.lib import recfunctions
    desitilesnew = recfunctions.merge_arrays((desitilesnew, desitilesnew_add),
                                             flatten=True)

    p = desitilesnew['PASS']
    desitilesnew['PROGRAM'][p == 0] = 'GRAY'
    desitilesnew['PROGRAM'][(p >= 1) & (p <= 4)] = 'DARK'
    desitilesnew['PROGRAM'][(p >= 5) & (p <= 7)] = 'BRIGHT'
    desitilesnew['PROGRAM'][(p >= 8)] = 'EXTRA'
    obscondmapping = {'EXTRA': 0, 'DARK': 1, 'GRAY': 2, 'BRIGHT': 4}
    for program, obscond in obscondmapping.items():
        m = desitilesnew['PROGRAM'] == program
        desitilesnew['OBSCONDITIONS'][m] = obscond

    desitilesnew['AIRMASS'] = airmass(
        np.ones(len(desitilesnew), dtype='f4')*15., desitilesnew['DEC'],
        31.96)
    signalfac = 10.**(2.165*desitilesnew['EBV_MED']/2.5)
    desitilesnew['EXPOSEFAC'] = signalfac**2 * desitilesnew['AIRMASS']**1.25
    desitilesnew['CENTERID'] = np.concatenate(
        [desitilesnew['TILEID'][mc]]*10)
    centerind = desitilesnew['CENTERID'] - 1
    desitilesnew['IN_IMAGING'] = desitilesnew['IMAGEFRAC_GRZ'][centerind] > 0.9
    dc = desitilesnew['DEC'][centerind]

    coord = SkyCoord(ra=desitilesnew['RA']*u.deg,
                     dec=desitilesnew['DEC']*u.deg, frame='icrs')
    coordgal = coord.galactic
    lt, bt = coordgal.l.value, coordgal.b.value
    lc = lt[centerind]
    bc = bt[centerind]
    desitilesnew['IN_DESI'] = (
        (desitilesnew['IN_IMAGING'] != 0) & (dc >= -18) & (dc <= 77.7) &
        ((bc > 0) | (dc < 32.2)) &
        (((np.abs(bc) > 22) & ((lc < 90) | (lc > 270))) |
         ((np.abs(bc) > 20) & (lc > 90) & (lc < 270))))
    return desitilesnew


def writefiles(tiles, fnbase, overwrite=False, viewer=False):
    if viewer:
        tilesviewer = tiles[(tiles['CENTERID'] == tiles['TILEID']) &
                            (tiles['IN_IMAGING'] != 0)]
        tiles_add = np.zeros(len(tilesviewer), dtype=[
            ('NAME', 'a20'), ('RADIUS', 'f4'), ('COLOR', 'a20')])
        tiles_add['RADIUS'] = 5832
        m = tilesviewer['IN_DESI'] != 0
        tiles_add['COLOR'][m] = 'green'
        tiles_add['COLOR'][~m] = 'red'
        tiles_add['NAME'] = [str(tid) for tid in tilesviewer['TILEID']]
        from numpy.lib import recfunctions
        tilesviewer = recfunctions.merge_arrays((tilesviewer, tiles_add),
                                                flatten=True)
        fits.writeto(fnbase+'-viewer.fits', tilesviewer, overwrite=True)

    tiles.dtype.names = [n.upper() for n in tiles.dtype.names]
    tilestab = Table(tiles, meta={'EXTNAME': 'TILES'})
    metadata = {'tileid': ('', 'Unique tile ID'),
                'ra': ('deg', 'Right ascension'),
                'dec': ('deg', 'Declination'),
                'pass': ('', 'DESI layer'),
                'in_desi': ('', '1=within DESI footprint; 0=outside'),
                'ebv_med': ('mag', 'Median Galactic E(B-V) extinction in tile'),
                'airmass': ('', 'Airmass if observed at hour angle 15 deg'),
                'star_density': ('deg^-2', 'median number density of Gaia stars brighter than 19.5 mag in tile'),
                'exposefac': ('', 'Multiplicative exposure time factor from airmass and E(B-V)'),
                'program': ('', 'DARK, GRAY, BRIGHT, or EXTRA'),
                'obsconditions': ('', '1 for DARK, 2 for GRAY, 4 for BRIGHT, 0 for EXTRA'),
                'brightra': ('deg', 'RAs of 3 brightest Tycho-2 stars in tile'),
                'brightdec': ('deg', 'Decs of 3 brightest Tycho-2 stars in tile'),
                'brightvtmag': ('mag', 'V_T magnitudes of 3 brightest Tycho-2 stars in tile'),
#                 'centerid': ('', 'Unique tile ID of pass 0 tile corresponding to this tile'),
                'imagefrac_g': ('', 'Fraction of this tile within 1.605 deg with g imaging'),
                'imagefrac_r': ('', 'Fraction of this tile within 1.605 deg with r imaging'),
                'imagefrac_z': ('', 'Fraction of this tile within 1.605 deg with z imaging'),
                'imagefrac_gr': ('', 'Fraction of this tile within 1.605 deg with gr imaging'),
                'imagefrac_grz': ('', 'Fraction of this tile within 1.605 deg with grz imaging'),
                'in_imaging': ('', 'Central tile has imagefrac_grz > 0.9'),
                }
    metadatacaps = {k.upper(): v for k, v in metadata.items()}
    unitdict = {'': None, 'deg': u.deg, 'mag': u.mag, 'deg^-2': 1/u.deg/u.deg}
    for name in tilestab.dtype.names:
        tilestab[name].unit = unitdict[metadatacaps[name][0]]
        tilestab[name].description = metadatacaps[name][1]
    tilestab2 = tilestab.copy()
    tilestab2.remove_columns(['BRIGHTRA', 'BRIGHTDEC', 'BRIGHTVTMAG'])
    tilestab2.write(fnbase+'.ecsv', overwrite=overwrite)
    tilestab.write(fnbase+'.fits', format='fits', overwrite=overwrite)


def extraqa(tiles):
    from matplotlib import pyplot as p
    p.figure('Exposure Factor')
    p.clf()
    setup_print((5, 4), scalefont=1.2)
    p.subplots_adjust(left=0.15, bottom=0.15)
    m = tiles['IN_DESI'] != 0
    if 'EXPOSEFAC' in tiles.dtype.names:
        exposefac = tiles['EXPOSEFAC']
    else:
        signalfac = 10.**(2.165*tiles['EBV_MED']/2.5)
        tairmass = airmass(
            np.ones(len(tiles), dtype='f4')*15.,
            tiles['DEC'], 31.96)
        exposefac = signalfac**2 * tairmass**1.25
    _ = p.hist(exposefac[m], range=[1, 3.5], bins=25, histtype='step')
    p.ylabel('Number of tiles')
    p.xlabel('EXPOSEFAC')
    p.savefig('exposefac.png')
    p.savefig('exposefac.pdf')

    p.figure('Imaging coverage')
    p.clf()
    setup_print((8, 5), scalefont=1.2)
    p.subplots_adjust(left=0.125, bottom=0.1)
    # m0 = tiles['CENTERID'] == tiles['TILEID']
    m0 = (tiles['PASS'] == 0) & (tiles['PROGRAM'] == 'dark')
    mdesi = tiles['IN_DESI'] != 0
    p.scatter(((tiles['RA'][m0 & mdesi]+60) % 360)-60,
              tiles['DEC'][m0 & mdesi],
              c=tiles['IMAGEFRAC_GRZ'][m0 & mdesi], marker='s', s=8,
              vmin=0, vmax=1)
    p.scatter(((tiles['RA'][m0 & ~mdesi]+60) % 360)-60,
              tiles['DEC'][m0 & ~mdesi],
              c=tiles['IMAGEFRAC_GRZ'][m0 & ~mdesi], marker='^', s=4, vmin=0,
              vmax=1)
    cbar = p.colorbar()
    cbar.set_label('$grz$ coverage fraction')
    p.xlabel(r'$\alpha$ ($\degree$)')
    p.ylabel(r'$\delta$ ($\degree$)')
    # p.title('DESI Imaging Coverage')
    p.ylim(-30, 90)
    p.xlim(300, -60)
    p.savefig('imagingcoverage.pdf')
    p.savefig('imagingcoverage.png')

    p.figure('Tile centers')
    p.clf()
    setup_print((8, 5), scalefont=1.2)
    p.subplots_adjust(left=0.125, bottom=0.1)
    m5 = ((tiles['PASS'] < 7) & (tiles['IN_DESI'] != 0) &
          (tiles['PROGRAM'] == 'dark'))
    # mef = tiles['EXPOSEFAC'] < 2.5
    print('Mean, median exposefac:',
          np.mean(exposefac[m5]), np.median(exposefac[m5]))

    p.scatter((((tiles['ra']+60) % 360)-60)[m5], tiles['dec'][m5],
              c=exposefac[m5] > 2.5,
              s=5*(exposefac[m5] > 2.5)+1,
              cmap='bwr')
    p.xlabel(r'$\alpha$ ($\degree$)')
    p.ylabel(r'$\delta$ ($\degree$)')
    p.xlim(300, -60)
    p.ylim(-20, 80)
    p.text(280, -13, 'Red where EXPOSEFAC > 2.5')
    p.text(280, -18, '%d tiles in 5 passes' % np.sum(m5))
    p.savefig('tilerd.png')
    p.savefig('tilerd.pdf')

    p.figure('Stellar Density')
    p.clf()
    setup_print((5, 4), scalefont=1.2)
    p.subplots_adjust(left=0.15, bottom=0.15)
    coord = SkyCoord(ra=tiles['RA']*u.deg, dec=tiles['DEC']*u.deg,
                     frame='icrs')
    coordgal = coord.galactic
    lt, bt = coordgal.l.value, coordgal.b.value
    m0 = (tiles['PASS'] == 0) & (tiles['PROGRAM'] == 'dark')
    mindesi = (tiles['IN_DESI'] != 0)
    p.plot(bt[m0 & ~mindesi], tiles['STAR_DENSITY'][m0 & ~mindesi], 'ro',
           markersize=0.5)
    p.plot(bt[m0 & mindesi], tiles['STAR_DENSITY'][m0 & mindesi], 'ko',
           markersize=0.5)
    p.xlabel(r'$b$ ($\degree$)')
    p.ylabel(r'Number of Gaia stars per sq. deg.')
    p.xlim(-90, 90)
    p.yscale('log')
    p.ylim(1e3, 1e6)
    p.savefig('stardensity.png')
    p.savefig('stardensity.pdf')
    npts = 10**6
    uv = np.random.randn(npts, 3)
    rr, dd = uv2lb(uv)
    mn = bt > 0
    ms = bt < 0
    mrn, mtn, drtn = match_radec(rr, dd, tiles['RA'][m5 & mn],
                                 tiles['DEC'][m5 & mn], 1.6)
    mrs, mts, drts = match_radec(rr, dd, tiles['RA'][m5 & ms],
                                 tiles['DEC'][m5 & ms], 1.6)
    ncovern = np.bincount(mrn, minlength=len(rr))
    ncovers = np.bincount(mrs, minlength=len(rr))
    print('North area:', (np.sum(ncovern >= 4)*4*np.pi *
                          (180/np.pi)**2/npts))
    print('South area:', (np.sum(ncovers >= 4)*4*np.pi *
                          (180/np.pi)**2/npts))

    mr, mt, drt = match_radec(rr, dd, tiles['RA'][m5], tiles['DEC'][m5], 1.6)
    ncover = np.bincount(mr, minlength=len(rr))

    for i in range(10):
        print(r'%4d  &  %6.0f  \\' %
              (i, np.sum(ncover >= i)*4*np.pi*(180./np.pi)**2/npts))

    m0 = ((tiles['IN_DESI'] != 0) &
          (tiles['PROGRAM'] == 'dark') & (tiles['PASS'] == 0))
    mrc, mtc, drtc = match_radec(rr, dd, tiles['RA'][m0], tiles['DEC'][m0],
                                 1.6)
    print('Average number of coverings for points within 1.6 deg of pass=0:',
          np.mean(ncover[mrc]))


def airmass(ha, dec, lat):
    az, alt = rotate(ha, dec, 0., lat)
    sinalt = np.clip(np.sin(np.radians(alt)), 1e-2, np.inf)
    return 1./sinalt


def rotate(l, b, l0, b0, phi0=0.):
    l = np.radians(l)
    b = np.radians(b)
    l0 = np.radians(l0)
    b0 = np.radians(b0)
    ce = np.cos(b0)
    se = np.sin(b0)
    phi0 = np.radians(phi0)

    cb, sb = np.cos(b), np.sin(b)
    cl, sl = np.cos(l-l0+np.pi/2.), np.sin(l-l0+np.pi/2.)

    ra = np.arctan2(cb*cl, sb*ce-cb*se*sl) + phi0
    dec = np.arcsin(cb*ce*sl + sb*se)

    ra = ra % (2*np.pi)

    ra = np.degrees(ra)
    dec = np.degrees(dec)

    return ra, dec


def rotate2(l, b, l0, b0, phi0=0.):
    return rotate(l, b, phi0, b0, phi0=l0)


def heal2cart(heal, interp=True, return_pts=False):
    import healpy
    nside = healpy.get_nside(heal)  #*(2 if interp else 1)
    owidth = 8*nside
    oheight = 4*nside-1
    dm, rm = np.mgrid[0:oheight, 0:owidth]
    rm = 360.-(rm+0.5) / float(owidth) * 360.
    dm = -90. + (dm+0.5) / float(oheight) * 180.
    t, p = lb2tp(rm.ravel(), dm.ravel())
    if interp:
        map = healpy.get_interp_val(heal, t, p)
    else:
        pix = healpy.ang2pix(nside, t, p)
        map = heal[pix]
    map = map.reshape((oheight, owidth))
    if return_pts:
        map = (map, np.sort(np.unique(rm)), np.sort(np.unique(dm)))
    return map


def setup_print(size=None, keys=None, scalefont=1., **kw):
    from matplotlib import pyplot
    params = {'backend': 'ps',
              'axes.labelsize': 12*scalefont,
              'font.size': 12*scalefont,
              'legend.fontsize': 10*scalefont,
              'xtick.labelsize': 10*scalefont,
              'ytick.labelsize': 10*scalefont,
              'axes.titlesize': 18*scalefont,
              }
    for key in kw:
        params[key] = kw[key]
    if keys is not None:
        for key in keys:
            params[key] = keys[key]
    oldparams = dict(pyplot.rcParams.items())
    pyplot.rcParams.update(params)
    if size is not None:
        pyplot.gcf().set_size_inches(*size, forward=True)
    return oldparams


def firstyeartiles(tiles, orig=True):
    racen, deccen = (tiles['RA'][tiles['CENTERID']-1],
                     tiles['DEC'][tiles['CENTERID']-1])
    # m1 = (deccen > -9.5) & (deccen < 9.5) & (tiles['pass'] == 3)
    # m2 = (deccen > -9.5) & (deccen < 7) & (tiles['pass'] == 2)
    # m3 = (deccen > -7) & (deccen < 7) & (tiles['pass'] == 0)
    # m4 = (deccen > -7) & (deccen < 7) & (tiles['pass'] == 1)
    passes = [3, 4, 0, 1] if orig else [1, 2, 3, 4]
    print('passes', passes)
    m1 = (deccen > -6.5) & (deccen < 6.5) & (tiles['PASS'] == passes[0])
    m2 = (deccen > -3.9) & (deccen < 6.5) & (tiles['PASS'] == passes[1])
    m3 = (deccen > -3.9) & (deccen < 3.9) & (tiles['PASS'] == passes[2])
    m4 = (deccen > -3.9) & (deccen < 3.9) & (tiles['PASS'] == passes[3])
    rabounds = (((racen > 148) & (racen < 213)) | (racen > 148+180) |
                (racen < 213-180))
    return [m1 & rabounds, m2 & rabounds, m3 & rabounds, m4 & rabounds]


def render_tiles_and_edges(tiles, selections):
    rr = np.linspace(0, 360, 3600*2, endpoint=False)
    dd = np.linspace(-8, 8, 160*2, endpoint=False)
    ra = rr.reshape(-1, 1)*np.ones((1, len(dd)))
    da = dd.reshape(1, -1)*np.ones((len(rr), 1))
    passes = []
    for m in selections:
        pi = render(ra.ravel(), da.ravel(),
                    tiles['RA'][m], tiles['DEC'][m])
        medge = edgetiles(tiles, m)
        po = render(ra.ravel(), da.ravel(),
                    tiles['RA'][medge].ravel(), tiles['DEC'][medge].ravel())
        passes.append([pi.reshape(ra.shape), po.reshape(ra.shape)])
    return passes


def edgetiles(tiles, sel):
    r0, d0 = tiles['RA'][sel], tiles['DEC'][sel]
    if not np.all(tiles['PASS'][sel] == tiles['PASS'][sel][0]):
        raise ValueError('all tiles in selection must be in the same pass.')
    pass0 = tiles['PASS'][sel][0]
    mp = np.flatnonzero(tiles['PASS'] == pass0)
    rad = 1.605*2*1.2
    m0, mt, d0t = match_radec(r0, d0, tiles['RA'][mp], tiles['DEC'][mp], rad)
    edges = np.zeros(len(tiles), dtype='bool')
    edges[mp[mt]] = 1
    edges[sel] = 0
    return edges


def make_tiles_from_fiberassignfn(fafn, tiletabs, exposures):
    fafn = sorted(fafn)[::-1]
    tileid = dict()
    import re
    rgx = re.compile(r'fiberassign-(\d{6})\.fits(\.gz)?')
    fafntokeep = []
    for fn in fafn:
        match = rgx.match(os.path.basename(fn))
        if match:
            tileid0 = int(match.group(1))
            if not tileid.get(tileid0):
                fafntokeep.append(fn)
                tileid[tileid0] = True
        else:
            print(f'failed to match {fn}')
    dat = np.zeros(len(fafntokeep),
                   dtype=[('TILEID', 'i4'), ('RA', 'f8'), ('DEC', 'f8'),
                          ('PROGRAM', 'U20'), ('EBV_MED', 'f4'),
                          ('FAFLAVOR', 'U20')])
    for i, fn in enumerate(fafntokeep):
        hdr = fits.getheader(fn)
        if 'TILEID' not in hdr:
            hdr = fits.getheader(fn, 'FIBERASSIGN')
        dat['TILEID'][i] = hdr['TILEID']
        dat['RA'][i] = hdr['TILERA']
        dat['DEC'][i] = hdr['TILEDEC']
        dat['PROGRAM'][i] = hdr.get('FAPRGRM', 'UNKNOWN')
        dat['FAFLAVOR'][i] = hdr.get('FAFLAVOR', 'UNKNOWN')
    alreadyusedtileid = np.unique(
        np.concatenate([tab['TILEID'] for tab in tiletabs]))
    m = ~np.isin(dat['TILEID'], alreadyusedtileid)
    dat = dat[m]
    from copy import deepcopy
    colarr = []
    for i in range(len(tiletabs[0].columns)):
        c = deepcopy(tiletabs[0].columns[i])
        c.resize(len(dat))
        colarr.append(c)
    out = Table(colarr)
    out['TILEID'][:] = dat['TILEID']
    out['PASS'][:] = 0
    out['RA'][:] = dat['RA']
    out['DEC'][:] = dat['DEC']
    out['PROGRAM'] = dat['FAFLAVOR']
    # out['PROGRAM'][:] = dat['FAFLAVOR']
    out['IN_DESI'][:] = False
    out['PRIORITY'][:] = 1.0
    out['STATUS'][:] = 'unobs'
    observed = np.isin(out['TILEID'], np.unique(exposures['TILEID']))
    out['STATUS'][observed] = 'obsstart'
    out['DONEFRAC'][:] = 0
    out['AVAILABLE'][:] = False
    out['PRIORITY_BOOSTFAC'][:] = 1
    out = out[np.argsort(out['TILEID'])]
    return out


def make_tiles_from_fiberassign(dirname, gaiadensitymapfile,
                                tycho2file, covfile):
    fn = glob.glob(os.path.join(dirname, '**/fiberassign*.fits*'),
                   recursive=True)
    fn = sorted(fn)
    tiles = np.zeros(len(fn), dtype=basetiledtype)
    for i, fn0 in enumerate(fn):
        h = fits.getheader(fn0)
        if 'TILEID' not in h:
            tiles['TILEID'][i] = -1
            continue
        tiles['TILEID'][i] = h['TILEID']
        tiles['RA'][i] = h['TILERA']
        tiles['DEC'][i] = h['TILEDEC']
        isdither = False
        try:
            dat = fits.getdata(fn0, 'EXTRA')
            if 'UNDITHER_RA' in dat.dtype.names:
                isdither = True
        except Exception:
            isdither = False
        progstr = h.get('FAFLAVOR', 'unknown').strip()
        fa_surv = h['FA_SURV'].strip()
        if progstr[:len(fa_surv)] != fa_surv:
            progstr = fa_surv + '_' + progstr
        if isdither:
            progstr += '_dith'
        tiles['PROGRAM'][i] = progstr
        if 'OBSCON' in h:
            tiles['OBSCONDITIONS'] = targetmask.obsconditions.mask(h['OBSCON'])
        else:
            tiles['OBSCONDITIONS'] = 2**31-1
    tiles = tiles[tiles['TILEID'] != -1]
    tiles['AIRMASS'] = airmass(
        np.ones(len(tiles), dtype='f4')*15., tiles['DEC'], 31.96)
    tiles_add = add_info_fields(tiles, gaiadensitymapfile,
                                tycho2file, covfile)
    from numpy.lib import recfunctions
    tiles = recfunctions.merge_arrays((tiles, tiles_add), flatten=True)
    signalfac = 10.**(2.165*tiles['EBV_MED']/2.5)
    tiles['EXPOSEFAC'] = signalfac**2 * tiles['AIRMASS']**1.25
    tiles['CENTERID'] = tiles['TILEID']
    for i, prog in enumerate(np.unique(tiles['PROGRAM'])):
        m = tiles['PROGRAM'] == prog
        tiles['PASS'][m] = i
    coord = SkyCoord(ra=tiles['RA']*u.deg,
                     dec=tiles['DEC']*u.deg, frame='icrs')
    coordgal = coord.galactic
    lt, bt = coordgal.l.value, coordgal.b.value
    dt = tiles['DEC']
    tiles['IN_DESI'] = (
        (tiles['IN_IMAGING'] != 0) & (dt >= -18) & (dt <= 77.7) &
        ((bt > 0) | (dt < 32.2)) &
        (((np.abs(bt) > 22) & ((lt < 90) | (lt > 270))) |
         ((np.abs(bt) > 20) & (lt > 90) & (lt < 270))))

    return tiles


def decorate_djs_tilefile(fn, gaiafn, tychofn, coveragefn,
                          rewritecenterid=False):
    from astropy import table
    tf = Table.read(fn)
    tf['EBV_MED'] = np.zeros(len(tf), dtype='f4')
    tf['STAR_DENSITY'] = np.zeros(len(tf), dtype='f4')
    if rewritecenterid:
        tf['CENTERID'] = np.arange(len(tf), dtype='i4')+1
    add = add_info_fields(tf, gaiafn, tychofn, coveragefn)
    addtab = Table(add)
    addtab.remove_column('CENTERID')
    tfadd = table.hstack([tf, addtab])
    centerind = tfadd['CENTERID'] - 1
    dc = tfadd['DEC'][centerind]
    rc = tfadd['RA'][centerind]
    cc = SkyCoord(ra=rc*u.deg, dec=dc*u.deg)
    lc = cc.galactic.l.to(u.deg).value
    bc = cc.galactic.b.to(u.deg).value
    tfadd['IN_IMAGING'] = tfadd['IMAGEFRAC_GRZ'][centerind] > 0.9
    tfadd['IN_DESI'] = (
        (tfadd['IN_IMAGING'] != 0) & (dc >= -18) & (dc <= 77.7) &
        ((bc > 0) | (dc < 32.2)) &
        (((np.abs(bc) > 22) & ((lc < 90) | (lc > 270))) |
         ((np.abs(bc) > 20) & (lc > 90) & (lc < 270))))
    return tfadd


def tenkfootprint(tiles):
    from matplotlib import path
    northpoly = np.array(
        [[272.32117571,  25.10016439],
         [247.80111764,  19.15590789],
         [245.20050542,  13.58316742],
         [231.82592829,   7.26739489],
         [226.25318782,  -15],
         [123.71476317,  -15],
         [102.16683335,  36.24564533],
         [139.68995251,  49.62022246],
         [186.50097247,  49.62022246],
         [231.82592829,  51.47780262],
         [264.89085508,  64.48086372],
         [272.69269174,  64.85237975],
         [289.78242918,  55.19296293],
         [279.00846427,  27.70077661]])

    southpoly = np.array(
        [[50.15458896,   6.52436283],
         [49.78307293,  -9.82234255],
         [-4.45826765, -10.19385858],
         [-5.20129972,  -6.47869827],
         [-47.92564332,  -5.36415018],
         [-47.92564332,  16.55529567],
         [-32.69348604,  29.55835677],
         [41.98123627,  31.0444209],
         [33.43636755,  21.38500408],
         [39.00910802,   7.63891092],
         [58.32794165,   6.52436283]])
    coord = SkyCoord(ra=tiles['RA']*u.deg, dec=tiles['DEC']*u.deg)
    coordgal = coord.galactic
    l, b = coordgal.l.to(u.deg).value, coordgal.b.to(u.deg).value
    north = b > 0
    south = b < 0
    northpath = path.Path(list(northpoly)+list(northpoly)[0:1])
    southpath = path.Path(list(southpoly)+list(southpoly)[0:1])
    northmask = north & northpath.contains_points(
        np.array([tiles['RA'], tiles['DEC']]).T)
    southmask = south & southpath.contains_points(
        np.array([((tiles['RA']+180) % 360)-180, tiles['DEC']]).T)
    return northmask | southmask
