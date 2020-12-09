import os
import glob
import numpy
from desimodel import focalplane
from astropy.io import fits
from desitarget import targetmask

basetiledtype = [
    ('tileid', 'i4'), ('ra', 'f8'), ('dec', 'f8'), ('pass', 'i4'),
    ('in_desi', 'bool'), ('ebv_med', 'f4'), ('airmass', 'f4'),
    ('star_density', 'f4'), ('exposefac', 'f4'),
    ('program', 'U20'), ('obsconditions', 'i4')]
addtiledtype = [
    ('brightra', '3f8'), ('brightdec', '3f8'), ('brightvtmag', '3f4'),
    ('centerid', 'i4'), ('imagefrac_g', 'f4'), ('imagefrac_r', 'f4'),
    ('imagefrac_z', 'f4'), ('imagefrac_gr', 'f4'), ('imagefrac_grz', 'f4'),
    ('in_imaging', 'f4')]


def match2d(x1, y1, x2, y2, rad):
    """Find all matches between x1, y1 and x2, y2 within radius rad."""
    from scipy.spatial import cKDTree
    xx1 = numpy.stack([x1, y1], axis=1)
    xx2 = numpy.stack([x2, y2], axis=1)
    tree1 = cKDTree(xx1)
    tree2 = cKDTree(xx2)
    res = tree1.query_ball_tree(tree2, rad)
    lens = [len(r) for r in res]
    m1 = numpy.repeat(numpy.arange(len(x1), dtype='i4'), lens)
    if sum([len(r) for r in res]) == 0:
        m2 = m1.copy()
    else:
        m2 = numpy.concatenate([r for r in res if len(r) > 0])
    d12 = numpy.sqrt(numpy.sum((xx1[m1, :]-xx2[m2, :])**2, axis=1))
    return m1, m2, d12


def lb2uv(r, d):
    return tp2uv(*lb2tp(r, d))


def uv2lb(uv):
    return tp2lb(*uv2tp(uv))


def uv2tp(uv):
    norm = numpy.sqrt(numpy.sum(uv**2., axis=1))
    uv = uv / norm.reshape(-1, 1)
    t = numpy.arccos(uv[:,2])
    p = numpy.arctan2(uv[:,1], uv[:,0])
    return t, p


def lb2tp(l, b):
    return (90.-b)*numpy.pi/180., l*numpy.pi/180.


def tp2lb(t, p):
    return p*180./numpy.pi % 360., 90.-t*180./numpy.pi


def tp2uv(t, p):
    z = numpy.cos(t)
    x = numpy.cos(p)*numpy.sin(t)
    y = numpy.sin(p)*numpy.sin(t)
    return numpy.concatenate([q[...,numpy.newaxis] for q in (x, y, z)],
                             axis=-1)



def match_radec(r1, d1, r2, d2, rad=1./60./60., nneighbor=0, notself=False):
    """Match r1, d1, to r2, d2, within radius rad."""
    if notself and nneighbor > 0:
        nneighbor += 1
    uv1 = lb2uv(r1, d1)
    uv2 = lb2uv(r2, d2)
    from scipy.spatial import cKDTree
    tree = cKDTree(uv2)
    dub = 2*numpy.sin(numpy.radians(rad)/2)
    if nneighbor > 0:
        d12, m2 = tree.query(uv1, nneighbor, distance_upper_bound=dub)
        if nneighbor > 1:
            m2 = m2.reshape(-1)
            d12 = d12.reshape(-1)

        m1 = numpy.arange(len(r1)*nneighbor, dtype='i4') // nneighbor
        d12 = 2*numpy.arcsin(numpy.clip(d12 / 2, 0, 1))*180/numpy.pi
        m = m2 < len(r2)
    else:
        tree1 = cKDTree(uv1)
        res = tree.query_ball_tree(tree1, dub)
        lens = [len(r) for r in res]
        m2 = numpy.repeat(numpy.arange(len(r2), dtype='i4'), lens)
        res = [r for r in res if len(r) > 0]
        if len(res) > 0:
            m1 = numpy.concatenate(res)
        else:
            m1 = numpy.zeros(0, dtype='i4')
        d12 = gc_dist(r1[m1], d1[m1], r2[m2], d2[m2])
        m = numpy.ones(len(m1), dtype='bool')
    if notself:
        m = m & (m1 != m2)
    return m1[m], m2[m], d12[m]


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


def render(ra, dec, tilera, tiledec, fiberposfile=None):
    """Return number of possible observations of ra, dec, given focal
    plane centers tilera, tiledec."""
    out = numpy.zeros_like(ra, dtype='i4')
    mg, mt, dgt = match_radec(ra, dec, tilera, tiledec, 1.65)
    s = numpy.argsort(mt)
    if fiberposfile is None:
        fiberposfile = os.path.join(os.environ['DESIMODEL'], 'data',
                                    'focalplane', 'fiberpos.fits')
    fpos = fits.getdata(fiberposfile)
    for f, l in subslices(mt[s]):
        tileno = mt[s[f]]
        ind = mg[s[f:l]]
        x, y = focalplane.radec2xy(tilera[tileno], tiledec[tileno],
                                   ra[ind], dec[ind])
        mx, mf, dxf = match2d(x, y, fpos['x'], fpos['y'], 6)
        # much slower than my custom-rolled version!
        out += numpy.bincount(ind[mx], minlength=len(out))
    return out


def render_simple(ra, dec, tilera, tiledec):
    out = numpy.zeros_like(ra, dtype='i4')
    mg, mt, dgt = match_radec(ra, dec, tilera, tiledec, 1.63)
    out += numpy.bincount(mg, minlength=len(out))
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
    pos = numpy.array(pos)
    ras = pos[:, :, 0].ravel()
    decs = pos[:, :, 1].ravel()
    slitno = numpy.tile(fpos['slitblock']+fpos['petal']*20, len(tilera))
    radbin = numpy.floor(numpy.sqrt(fpos['x']**2+fpos['y']**2)/20).astype('i4')
    radbin = numpy.tile(radbin, len(tilera))
    expnum = numpy.repeat(numpy.arange(len(tilera)), len(fpos))
    rad = 1.4/60
    m1, m2, d12 = match_radec(ras, decs, ras, decs, rad,
                                       notself=True)
    m = expnum[m1] != expnum[m2]
    m1 = m1[m]
    m2 = m2[m]
    d12 = d12[m]
    # area of intersection of two equal-size circles?
    # area: 2r^2 arccos(d/2r)-0.5 d sqrt((2r-d)(2r+d))
    area = (2*rad**2*numpy.arccos(d12/2./rad) -
            0.5*d12*numpy.sqrt((2*rad-d12)*(2*rad+d12)))
    nslitno = numpy.max(slitno)+1
    nradbin = numpy.max(radbin)+1
    adj = numpy.zeros(nslitno**2, dtype='f4')
    adjr = numpy.zeros(nradbin**2, dtype='f4')
    ind = slitno[m1]*nslitno+slitno[m2]
    indr = radbin[m1]*nradbin+radbin[m2]
    adj += numpy.bincount(ind, weights=area[m1], minlength=len(adj))
    adj = adj.reshape(nslitno, nslitno)
    adjr += numpy.bincount(indr, weights=area[m1], minlength=len(adjr))
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
    ang = numpy.radians(ang)
    dang = numpy.pi/2
    dithers = [[0, 0],
               [dx*sin(ang+0*dang), dx*cos(ang+0*dang)],
               [dx*sin(ang+1*dang), dx*cos(ang+1*dang)],
               [dx*sin(ang+2*dang), dx*cos(ang+2*dang)]]
    dithers = numpy.cumsum(numpy.array(dithers), axis=0)
    dithers = list(dithers) + [[numpy.mean([d[0] for d in dithers]),
                                numpy.mean([d[1] for d in dithers])]]
    fac = 1./numpy.cos(numpy.radians(decs))
    fac = numpy.clip(fac, 1, 360*5)  # confusion near celestial pole.
    newras = numpy.concatenate([ras+d[0]*fac for d in dithers])
    newdecs = numpy.concatenate([decs+d[1] for d in dithers])
    newdecs = numpy.clip(newdecs, -numpy.inf, 90.)
    newras = newras % 360
    newras = numpy.concatenate([newras, newras])
    newdecs = numpy.concatenate([newdecs, newdecs])
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
    dx = dx*numpy.exp(numpy.arange(4)/3.)

    from numpy import sin, cos
    ang = numpy.radians(ang)
    dang = numpy.pi/2
    dithers = [[0, 0],
               [dx[0]*sin(ang+0*dang), dx[0]*cos(ang+0*dang)],
               [dx[1]*sin(ang+1*dang), dx[1]*cos(ang+1*dang)],
               [dx[2]*sin(ang+2*dang), dx[2]*cos(ang+2*dang)]]
    dithers = numpy.cumsum(numpy.array(dithers), axis=0)
    dithers -= numpy.mean(dithers, axis=0).reshape(1, -1)
    dithers = [[0, 0]] + list(dithers)
    if verbose:
        for dra, ddec in dithers:
            print(r'%6.3f  &  %6.3f \\' % (dra, ddec))
    fac = 1./numpy.cos(numpy.radians(decs))
    fac = numpy.clip(fac, 1, 360*5)  # confusion near celestial pole.
    newras = numpy.concatenate([ras+d[0]*fac for d in dithers])
    newdecs = numpy.concatenate([decs+d[1] for d in dithers])
    m = newdecs > 90
    newdecs[m] = 90-(newdecs[m]-90)
    newras[m] += 180.
    m = newdecs < -90
    newdecs[m] = -90+(-90-newdecs[m])
    newras[m] += 180.
    if numpy.any((newdecs > 90) | (newdecs < -90)):
        raise ValueError('Something is wrong!')
    newras = newras % 360
    newras2 = numpy.concatenate([
            newras[len(ras):], newras[:len(ras)]])
    newdecs2 = numpy.concatenate([
            newdecs[len(ras):], newdecs[:len(ras)]])
    # for duplicate 5 passes, change order slightly.
    # the zeroth and ninth passes are now the passes that are at the centers
    # of the other 4 passes.  This makes passes 1-4 and passes 5-7 somewhat
    # better optimized for complete coverage.
    newras = numpy.concatenate([newras, newras2])
    newdecs = numpy.concatenate([newdecs, newdecs2])
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

    lon1 = numpy.radians(lon1); lat1 = numpy.radians(lat1)
    lon2 = numpy.radians(lon2); lat2 = numpy.radians(lat2)

    return numpy.degrees(
        2*arcsin(sqrt( (sin((lat1-lat2)*0.5))**2 +
                       cos(lat1)*cos(lat2)*(sin((lon1-lon2)*0.5))**2 )));


def qa(desitiles, nside=1024, npts=1000, compare=False):
    """Make tiling QA plots; demonstrate usage."""
    import healpy
    theta, phi = healpy.pix2ang(nside, numpy.arange(12*nside**2))
    la, ba = phi*180./numpy.pi, 90-theta*180./numpy.pi
    m5pass = (desitiles['pass'] <= 4)
    m0 = desitiles['centerid'] == desitiles['tileid']
    ran, decn = logradecoffscheme(desitiles['ra'][m0],
                                  desitiles['dec'][m0], dx=0.6, ang=24)
    tilerd = {}
    if compare:
        tilerd['default'] = (desitiles['ra'], desitiles['dec'])
    tilerd['Tiles v3'] = (ran, decn)
    ims = {name: render(la, ba, rd[0][m5pass], rd[1][m5pass])
           for name, rd in tilerd.items()}
    pseudoindesi = ((gc_dist(la, ba, 180, 30) < 40)
                    | (gc_dist(la, ba, 0, 5) < 30))
    from matplotlib import pyplot as p

    p.figure('One Footprint')
    setup_print((5, 4), scalefont=1.2)
    p.subplots_adjust(left=0.15, bottom=0.15)
    delt = 1.8
    dg, rg = numpy.meshgrid(numpy.linspace(-delt, delt, npts),
                            numpy.linspace(-delt, delt, npts))
    dpts = 4./(npts - 1)
    p.clf()
    tim = render(rg.ravel(), dg.ravel(), numpy.zeros(1), numpy.zeros(1))
    tim = tim.reshape(rg.shape)
    p.imshow(tim.T, cmap='binary', origin='lower',
             extent=[-delt-dpts/2, delt+dpts/2, -delt-dpts/2, delt+dpts/2])
    p.gca().set_aspect('equal')
    p.xlabel(r'$\alpha$ ($\degree$)')
    p.ylabel(r'$\delta$ ($\degree$)')
    p.savefig('onefootprint.png', dpi=200)
    p.savefig('onefootprint.pdf', dpi=200)

    p.figure('One Center')
    setup_print((5, 4), scalefont=1.2)
    p.subplots_adjust(left=0.15, bottom=0.15)
    dg, rg = numpy.meshgrid(numpy.linspace(-delt, delt, npts),
                            numpy.linspace(-delt, delt, npts))
    dpts = 4./(npts - 1)
    p.clf()
    monecen = m5pass & (gc_dist(0, 0, ran, decn) < 1.2)
    tim = render(rg.ravel(), dg.ravel(), ran[monecen], decn[monecen])
    tim = tim.reshape(rg.shape)
    p.imshow(tim.T, cmap='binary', origin='lower',
             extent=[-delt-dpts/2, delt+dpts/2, -delt-dpts/2, delt+dpts/2])
    p.scatter(((ran[monecen]+180)%360)-180, decn[monecen],
              c=desitiles['pass'][monecen])
    p.xlim((-delt, delt))
    p.ylim((-delt, delt))
    p.gca().set_aspect('equal')
    p.xlabel(r'$\alpha$ ($\degree$)')
    p.ylabel(r'$\delta$ ($\degree$)')
    p.savefig('onecenter.png', dpi=200)
    p.savefig('onecenter.pdf', dpi=200)


    p.figure('Several Footprints')
    setup_print((5, 4), scalefont=1.2, )
    p.subplots_adjust(left=0.15, bottom=0.15)
    delt = 10
    dg, rg = numpy.meshgrid(numpy.linspace(-delt, delt, npts),
                            numpy.linspace(-delt, delt, npts))
    dpts = 4./(npts - 1)
    p.clf()
    tim = render(rg.ravel(), dg.ravel(), ran[m0], decn[m0])
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
             vmin=0, vmax=14)
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
               histtype='step', label=name, normed=True)
        print(r'# coverings  & fraction of area \\')
        for count in range(16):
            frac = (numpy.sum(im[pseudoindesi] == count)/
                    1./numpy.sum(pseudoindesi))
            print(r'%4d  &  %7.3e \\' % (count, frac))
    p.xlim(-0.5, 15.5)
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
        angsizes = numpy.arange(21).astype('f4')+0.5
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
    theta, phi = healpy.pix2ang(nside, numpy.arange(12*nside**2))
    la, ba = phi*180./numpy.pi, 90-theta*180./numpy.pi
    try:
        from desiutil import dust
        ebva = dust.ebv(la, ba, frame='galactic',
                        mapdir=os.getenv('DUST_DIR')+'/maps', scaling=1)
    except:
        import dust
        ebva = dust.getval(la, ba, map='sfd')
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    coord = SkyCoord(ra=tiles['ra']*u.deg, dec=tiles['dec']*u.deg, frame='icrs')
    coordgal = coord.galactic
    lt, bt = coordgal.l.value, coordgal.b.value
    uvt = lb2uv(lt, bt)
    gaiadens = fits.getdata(gaiadensitymapfile).copy()
    cov = fits.getdata(covfile).copy()
    fprad = 1.605
    for i in range(len(tiles)):
        ind = healpy.query_disc(nside, uvt[i], fprad*numpy.pi/180.)
        tiles['ebv_med'][i] = numpy.median(ebva[ind])
        tiles['star_density'][i] = numpy.median(gaiadens[ind])

    brightstars = fits.getdata(tycho2file)
    mb, mt, dbt = match_radec(brightstars['ra'], brightstars['dec'],
                              tiles['ra'], tiles['dec'], fprad)
    s = numpy.lexsort((brightstars['vtmag'][mb], mt))
    mt, mb, dbt = mt[s], mb[s], dbt[s]
    add = numpy.zeros(len(tiles), dtype=addtiledtype)
    add['brightvtmag'] = 999.
    for f, l in subslices(mt):
        end = numpy.clip(l-f, 0, 3)
        l = numpy.clip(l, f, f+3)
        ind = mt[f]
        add['brightra'][ind, 0:end] = brightstars['ra'][mb[f:l]]
        add['brightdec'][ind, 0:end] = brightstars['dec'][mb[f:l]]
        add['brightvtmag'][ind, 0:end] = brightstars['vtmag'][mb[f:l]]

    uvt = lb2uv(tiles['ra'], tiles['dec'])
    for i in range(len(tiles)):
        ind = healpy.query_disc(nside, uvt[i], fprad*numpy.pi/180.)
        for f in ['g', 'r', 'z', 'gr', 'grz']:
            val = numpy.mean(cov['%s_coverage' % f][ind])
            add['imagefrac_%s' % f][i] = val
    add['in_imaging'] = add['imagefrac_grz'] > 0.9
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
    m0 = desitiles['pass'] == 0
    ran, decn = logradecoffscheme(desitiles['ra'][m0],
                                  desitiles['dec'][m0], dx=0.6, ang=24,
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
    desitilesnew = numpy.zeros(len(desitiles), dtype=dtype)
    for n in desitilesnew.dtype.names:
        desitilesnew[n] = desitiles[n]
    desitilesnew.dtype.names = [n.lower() for n in desitilesnew.dtype.names]
    desitilesnew['ra'] = ran
    desitilesnew['dec'] = decn
    cenpass = 3 if firstyearoptimized else 0
    mc = desitilesnew['pass'] == cenpass
    # pass 0: centers in new scheme, no first year permutation
    # pass 3: centers in new scheme, if permuted to get more full depth in
    # first year
    desitilesnew['in_desi'] = numpy.concatenate(
        [desitilesnew['in_desi'][mc]]*10)
    # just repeat identically for each pass; all passes are close to
    # pass = 0 'centers'.

    desitilesnew_add = add_info_fields(desitilesnew, gaiadensitymapfile,
                                       tycho2file, covfile)
    from numpy.lib import recfunctions
    desitilesnew = recfunctions.merge_arrays((desitilesnew, desitilesnew_add),
                                             flatten=True)

    p = desitilesnew['pass']
    desitilesnew['program'][p == 0] = 'GRAY'
    desitilesnew['program'][(p >= 1) & (p <= 4)] = 'DARK'
    desitilesnew['program'][(p >= 5) & (p <= 7)] = 'BRIGHT'
    desitilesnew['program'][(p >= 8)] = 'EXTRA'
    obscondmapping = {'EXTRA': 0, 'DARK': 1, 'GRAY': 2, 'BRIGHT': 4}
    for program, obscond in obscondmapping.items():
        m = desitilesnew['program'] == program
        desitilesnew['obsconditions'][m] = obscond

    desitilesnew['airmass'] = airmass(
        numpy.ones(len(desitilesnew), dtype='f4')*15., desitilesnew['dec'],
        31.96)
    signalfac = 10.**(3.303*desitilesnew['ebv_med']/2.5)
    desitilesnew['exposefac'] = signalfac**2 * desitilesnew['airmass']**1.25
    desitilesnew['centerid'] = numpy.concatenate(
        [desitilesnew['tileid'][mc]]*10)
    centerind = desitilesnew['centerid'] - 1
    desitilesnew['in_imaging'] = desitilesnew['imagefrac_grz'][centerind] > 0.9
    dc = desitilesnew['dec'][centerind]

    from astropy.coordinates import SkyCoord
    from astropy import units as u
    coord = SkyCoord(ra=desitilesnew['ra']*u.deg,
                     dec=desitilesnew['dec']*u.deg, frame='icrs')
    coordgal = coord.galactic
    lt, bt = coordgal.l.value, coordgal.b.value
    lc = lt[centerind]
    bc = bt[centerind]
    desitilesnew['in_desi'] = (
        (desitilesnew['in_imaging'] != 0) & (dc >= -18) & (dc <= 77.7) &
        ((bc > 0) | (dc < 32.2)) &
        (((numpy.abs(bc) > 22) & ((lc < 90) | (lc > 270))) |
         ((numpy.abs(bc) > 20) & (lc > 90) & (lc < 270))))
    return desitilesnew


def writefiles(tiles, fnbase, overwrite=False, viewer=False):
    from astropy.io import ascii
    from matplotlib.mlab import rec_drop_fields
    from astropy import table
    if viewer:
        tilesviewer = tiles[(tiles['centerid'] == tiles['tileid']) &
                            (tiles['in_imaging'] != 0)]
        tiles_add = numpy.zeros(len(tilesviewer), dtype=[
            ('name', 'a20'), ('radius', 'f4'), ('color', 'a20')])
        tiles_add['radius'] = 5832
        m = tilesviewer['in_desi'] != 0
        tiles_add['color'][m] = 'green'
        tiles_add['color'][~m] = 'red'
        tiles_add['name'] = [str(tid) for tid in tilesviewer['tileid']]
        from numpy.lib import recfunctions
        tilesviewer = recfunctions.merge_arrays((tilesviewer, tiles_add),
                                                 flatten=True)
        fits.writeto(fnbase+'-viewer.fits', tilesviewer, overwrite=True)

    tiles.dtype.names = [n.upper() for n in tiles.dtype.names]
    tilestab = table.Table(tiles, meta={'EXTNAME': 'TILES'})
    metadata = {'tileid': ('', 'Unique tile ID'),
                'ra': ('deg', 'Right ascension'),
                'dec': ('deg', 'Declination'),
                'pass': ('', 'DESI layer'),
                'in_desi': ('', '1=within DESI footprint; 0=outside'),
                'ebv_med':('mag', 'Median Galactic E(B-V) extinction in tile'),
                'airmass':('', 'Airmass if observed at hour angle 15 deg'),
                'star_density':('deg^-2', 'median number density of Gaia stars brighter than 19.5 mag in tile'),
                'exposefac':('', 'Multiplicative exposure time factor from airmass and E(B-V)'),
                'program':('', 'DARK, GRAY, BRIGHT, or EXTRA'),
                'obsconditions':('', '1 for DARK, 2 for GRAY, 4 for BRIGHT, 0 for EXTRA'),
                'brightra':('deg', 'RAs of 3 brightest Tycho-2 stars in tile'),
                'brightdec':('deg', 'Decs of 3 brightest Tycho-2 stars in tile'),
                'brightvtmag':('mag', 'V_T magnitudes of 3 brightest Tycho-2 stars in tile'),
                'centerid':('', 'Unique tile ID of pass 0 tile corresponding to this tile'),
                'imagefrac_g':('', 'Fraction of this tile within 1.605 deg with g imaging'),
                'imagefrac_r':('', 'Fraction of this tile within 1.605 deg with r imaging'),
                'imagefrac_z':('', 'Fraction of this tile within 1.605 deg with z imaging'),
                'imagefrac_gr':('', 'Fraction of this tile within 1.605 deg with gr imaging'),
                'imagefrac_grz':('', 'Fraction of this tile within 1.605 deg with grz imaging'),
                'in_imaging':('', 'Central tile has imagefrac_grz > 0.9'),
                }
    metadatacaps = {k.upper(): v for k, v in metadata.items()}
    from astropy import units as u
    unitdict = {'': None, 'deg': u.deg, 'mag': u.mag, 'deg^-2': 1/u.deg/u.deg}
    for name in tilestab.dtype.names:
        tilestab[name].unit = unitdict[metadatacaps[name][0]]
        tilestab[name].description = metadatacaps[name][1]
    tilestab.write(fnbase+'.fits', format='fits', overwrite=overwrite)
    tilestab.remove_columns(['BRIGHTRA', 'BRIGHTDEC', 'BRIGHTVTMAG'])
    ascii.write(tilestab, fnbase+'.ecsv', format='ecsv', overwrite=overwrite)


def extraqa(tiles):
    from matplotlib import pyplot as p
    p.figure('Exposure Factor')
    p.clf()
    setup_print((5, 4), scalefont=1.2)
    p.subplots_adjust(left=0.15, bottom=0.15)
    m = tiles['in_desi'] != 0
    _ = p.hist(tiles['exposefac'][m], range=[1, 3.5], bins=25, histtype='step')
    p.ylabel('Number of tiles')
    p.xlabel('EXPOSEFAC')
    p.savefig('exposefac.png')
    p.savefig('exposefac.pdf')

    p.figure('Imaging coverage')
    p.clf()
    setup_print((8, 5), scalefont=1.2)
    p.subplots_adjust(left=0.125, bottom=0.1)
    m0 = tiles['centerid'] == tiles['tileid']
    mdesi = tiles['in_desi'] != False
    p.scatter(((tiles['ra'][m0 & mdesi]+60)%360)-60, tiles['dec'][m0 & mdesi],
              c=tiles['imagefrac_grz'][m0 & mdesi], marker='s', s=8,
              vmin=0, vmax=1)
    p.scatter(((tiles['ra'][m0 & ~mdesi]+60)%360)-60, tiles['dec'][m0 & ~mdesi],
              c=tiles['imagefrac_grz'][m0 & ~mdesi], marker='^', s=4, vmin=0,
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
    m5 = (tiles['pass'] <= 4) & (tiles['in_desi'] != 0)
    mef = tiles['exposefac'] < 2.5
    p.scatter((((tiles['ra']+60)%360)-60)[m5], tiles['dec'][m5],
              c=tiles['exposefac'][m5] > 2.5,
              s=5*(tiles['exposefac'][m5] > 2.5)+1,
              cmap='bwr')
    p.xlabel(r'$\alpha$ ($\degree$)')
    p.ylabel(r'$\delta$ ($\degree$)')
    p.xlim(300, -60)
    p.ylim(-20, 80)
    p.text(280, -13, 'Red where EXPOSEFAC > 2.5')
    p.text(280, -18, '%d tiles in 5 passes' % numpy.sum(m5))
    p.savefig('tilerd.png')
    p.savefig('tilerd.pdf')

    p.figure('Stellar Density')
    p.clf()
    setup_print((5, 4), scalefont=1.2)
    p.subplots_adjust(left=0.15, bottom=0.15)
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    coord = SkyCoord(ra=tiles['ra']*u.deg, dec=tiles['dec']*u.deg,
                     frame='icrs')
    coordgal = coord.galactic
    lt, bt = coordgal.l.value, coordgal.b.value
    m0 = (tiles['centerid'] == tiles['tileid'])
    mindesi = (tiles['in_desi'] != 0)
    p.plot(bt[m0 & ~mindesi], tiles['star_density'][m0 & ~mindesi], 'ro',
           markersize=0.5)
    p.plot(bt[m0 & mindesi], tiles['star_density'][m0 & mindesi], 'ko',
           markersize=0.5)
    p.xlabel(r'$b$ ($\degree$)')
    p.ylabel(r'Number of Gaia stars per sq. deg.')
    p.xlim(-90, 90)
    p.yscale('log')
    p.ylim(1e3, 1e6)
    p.savefig('stardensity.png')
    p.savefig('stardensity.pdf')
    npts = 10**6
    uv = numpy.random.randn(npts, 3)
    rr, dd = uv2lb(uv)
    mn = bt > 0
    ms = bt < 0
    mrn, mtn, drtn = match_radec(rr, dd, tiles['ra'][m5 & mn],
                                 tiles['dec'][m5 & mn], 1.6)
    mrs, mts, drts = match_radec(rr, dd, tiles['ra'][m5 & ms],
                                 tiles['dec'][m5 & ms], 1.6)
    ncovern = numpy.bincount(mrn, minlength=len(rr))
    ncovers = numpy.bincount(mrs, minlength=len(rr))
    print('North area:', (numpy.sum(ncovern >= 4)*4*numpy.pi*
                          (180/numpy.pi)**2/npts))
    print('South area:', (numpy.sum(ncovers >= 4)*4*numpy.pi*
                          (180/numpy.pi)**2/npts))

    mr, mt, drt = match_radec(rr, dd, tiles['ra'][m5], tiles['dec'][m5], 1.6)
    ncover = numpy.bincount(mr, minlength=len(rr))

    for i in range(10):
        print(r'%4d  &  %6.0f  \\' %
              (i, numpy.sum(ncover >= i)*4*numpy.pi*(180./numpy.pi)**2/npts))

    m0 = (tiles['in_desi'] != 0) & (tiles['centerid'] == tiles['tileid'])
    mrc, mtc, drtc = match_radec(rr, dd, tiles['ra'][m0], tiles['dec'][m0], 1.6)
    print('Average number of coverings for points within 1.6 deg of pass=3:',
          numpy.mean(ncover[mrc]))


def airmass(ha, dec, lat):
    az, alt = rotate(ha, dec, 0., lat)
    sinalt = numpy.clip(numpy.sin(numpy.radians(alt)), 1e-2, numpy.inf)
    return 1./sinalt


def rotate(l, b, l0, b0, phi0=0.):
    l = numpy.radians(l)
    b = numpy.radians(b)
    l0 = numpy.radians(l0)
    b0 = numpy.radians(b0)
    ce = numpy.cos(b0)
    se = numpy.sin(b0)
    phi0 = numpy.radians(phi0)

    cb, sb = numpy.cos(b), numpy.sin(b)
    cl, sl = numpy.cos(l-l0+numpy.pi/2.), numpy.sin(l-l0+numpy.pi/2.)

    ra  = numpy.arctan2(cb*cl, sb*ce-cb*se*sl) + phi0
    dec = numpy.arcsin(cb*ce*sl + sb*se)

    ra = ra % (2*numpy.pi)

    ra = numpy.degrees(ra)
    dec = numpy.degrees(dec)

    return ra, dec


def rotate2(l, b, l0, b0, phi0=0.):
    return rotate(l, b, phi0, b0, phi0=l0)


def heal2cart(heal, interp=True, return_pts=False):
    import healpy
    nside = healpy.get_nside(heal)#*(2 if interp else 1)
    owidth = 8*nside
    oheight = 4*nside-1
    dm,rm = numpy.mgrid[0:oheight,0:owidth]
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
        map = (map, numpy.sort(numpy.unique(rm)), numpy.sort(numpy.unique(dm)))
    return map

def setup_print(size=None, keys=None, scalefont=1., **kw):
    from matplotlib import pyplot
    params = {'backend': 'ps',
              'axes.labelsize': 12*scalefont,
              'font.size':12*scalefont,
              'legend.fontsize': 10*scalefont,
              'xtick.labelsize': 10*scalefont,
              'ytick.labelsize': 10*scalefont,
              'axes.titlesize':18*scalefont,
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
    racen, deccen = (tiles['ra'][tiles['centerid']-1],
                     tiles['dec'][tiles['centerid']-1])
    # m1 = (deccen > -9.5) & (deccen < 9.5) & (tiles['pass'] == 3)
    # m2 = (deccen > -9.5) & (deccen < 7) & (tiles['pass'] == 2)
    # m3 = (deccen > -7) & (deccen < 7) & (tiles['pass'] == 0)
    # m4 = (deccen > -7) & (deccen < 7) & (tiles['pass'] == 1)
    passes = [3, 4, 0, 1] if orig else [1, 2, 3, 4]
    print('passes', passes)
    m1 = (deccen > -6.5) & (deccen < 6.5) & (tiles['pass'] == passes[0])
    m2 = (deccen > -3.9) & (deccen < 6.5) & (tiles['pass'] == passes[1])
    m3 = (deccen > -3.9) & (deccen < 3.9) & (tiles['pass'] == passes[2])
    m4 = (deccen > -3.9) & (deccen < 3.9) & (tiles['pass'] == passes[3])
    rabounds = (((racen > 148) & (racen < 213)) | (racen > 148+180) |
                (racen < 213-180))
    return [m1 & rabounds, m2 & rabounds, m3 & rabounds, m4 & rabounds]


def render_tiles_and_edges(tiles, selections):
    rr = numpy.linspace(0, 360, 3600*2, endpoint=False)
    dd = numpy.linspace(-8, 8, 160*2, endpoint=False)
    ra = rr.reshape(-1, 1)*numpy.ones((1, len(dd)))
    da = dd.reshape(1, -1)*numpy.ones((len(rr), 1))
    passes = []
    for m in selections:
        pi = render(ra.ravel(), da.ravel(),
                    tiles['ra'][m], tiles['dec'][m])
        medge = edgetiles(tiles, m)
        po = render(ra.ravel(), da.ravel(),
                    tiles['ra'][medge].ravel(), tiles['dec'][medge].ravel())
        passes.append([pi.reshape(ra.shape), po.reshape(ra.shape)])
    return passes


def edgetiles(tiles, sel):
    r0, d0 = tiles['ra'][sel], tiles['dec'][sel]
    if not numpy.all(tiles['pass'][sel] == tiles['pass'][sel][0]):
        raise ValueError('all tiles in selection must be in the same pass.')
    pass0 = tiles['pass'][sel][0]
    mp = numpy.flatnonzero(tiles['pass'] == pass0)
    rad = 1.605*2*1.2
    m0, mt, d0t = match_radec(r0, d0, tiles['ra'][mp], tiles['dec'][mp], rad)
    edges = numpy.zeros(len(tiles), dtype='bool')
    edges[mp[mt]] = 1
    edges[sel] = 0
    return edges


def make_tiles_from_fiberassign(dirname, gaiadensitymapfile,
                                tycho2file, covfile):
    fn = glob.glob(os.path.join(dirname, '**/fiberassign*.fits.gz'),
                   recursive=True)
    tiles = numpy.zeros(len(fn), dtype=basetiledtype)
    for i, fn0 in enumerate(fn):
        h = fits.getheader(fn0)
        tiles['tileid'][i] = h['TILEID']
        tiles['ra'][i] = h['TILERA']
        tiles['dec'][i] = h['TILEDEC']
        tiles['program'][i] = h['FA_SURV'].strip()+'_'+h['FLAVOR'].strip()
        if 'OBSCON' in h:
            tiles['obsconditions'] = targetmask.obsconditions.mask(h['OBSCON'])
        else:
            tiles['obsconditions'] = 2**31-1
    tiles['airmass'] = airmass(
        numpy.ones(len(tiles), dtype='f4')*15., tiles['dec'], 31.96)
    tiles_add = add_info_fields(tiles, gaiadensitymapfile,
                                tycho2file, covfile)
    from numpy.lib import recfunctions
    tiles = recfunctions.merge_arrays((tiles, tiles_add), flatten=True)
    signalfac = 10.**(3.303*tiles['ebv_med']/2.5)
    tiles['exposefac'] = signalfac**2 * tiles['airmass']**1.25
    tiles['centerid'] = tiles['tileid']
    for i, prog in enumerate(numpy.unique(tiles['program'])):
        m = tiles['program'] == prog
        tiles['pass'][m] = i
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    coord = SkyCoord(ra=tiles['ra']*u.deg,
                     dec=tiles['dec']*u.deg, frame='icrs')
    coordgal = coord.galactic
    lt, bt = coordgal.l.value, coordgal.b.value
    dt = tiles['dec']
    tiles['in_desi'] = (
        (tiles['in_imaging'] != 0) & (dt >= -18) & (dt <= 77.7) &
        ((bt > 0) | (dt < 32.2)) &
        (((numpy.abs(bt) > 22) & ((lt < 90) | (lt > 270))) |
         ((numpy.abs(bt) > 20) & (lt > 90) & (lt < 270))))

    return tiles
