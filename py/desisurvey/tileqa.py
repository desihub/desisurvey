import os
import numpy
from desimodel import focalplane
import util_efs
import pdb
import orthographic

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


def render(ra, dec, tilera, tiledec, fiberposfile=None):
    """Return number of possible observations of ra, dec, given focal
    plane centers tilera, tiledec."""
    out = numpy.zeros_like(ra, dtype='i4')
    mg, mt, dgt = util_efs.match_radec(ra, dec, tilera, tiledec, 1.65)
    s = numpy.argsort(mt)
    if fiberposfile is None:
        fiberposfile = os.path.join(os.environ['DESIMODEL'], 'data', 
                                    'focalplane', 'fiberpos.fits')
    from astropy.io import fits
    fpos = fits.getdata(fiberposfile)
    for f, l in util_efs.subslices(mt[s]):
        tileno = mt[s[f]]
        ind = mg[s[f:l]]
        x, y = focalplane.radec2xy(tilera[tileno], tiledec[tileno],
                                   ra[ind], dec[ind])
        mx, mf, dxf = match2d(x, y, fpos['x'], fpos['y'], 6)
        out = util_efs.add_arr_at_ind(out, numpy.ones(len(mx), dtype='i4'), 
                                      ind[mx])
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
    from astropy.io import fits
    fpos = fits.getdata(fiberposfile)
    # really slow
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
    m1, m2, d12 = util_efs.match_radec(ras, decs, ras, decs, rad, 
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
    adj = util_efs.add_arr_at_ind(adj, area[m1], ind)
    adj = adj.reshape(nslitno, nslitno)
    adjr = util_efs.add_arr_at_ind(adjr, area[m1], indr)
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


def logradecoffscheme(ras, decs, dx=0.6, ang=24):
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
    dithers =  list(dithers) + [[0, 0]]
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
    newras = numpy.concatenate([newras, newras])
    newdecs = numpy.concatenate([newdecs, newdecs])
    return newras, newdecs


def gc_dist(lon1, lat1, lon2, lat2):
    from numpy import sin, cos, arcsin, sqrt

    lon1 = numpy.radians(lon1); lat1 = numpy.radians(lat1)
    lon2 = numpy.radians(lon2); lat2 = numpy.radians(lat2)

    return numpy.degrees(
        2*arcsin(sqrt( (sin((lat1-lat2)*0.5))**2 + 
                       cos(lat1)*cos(lat2)*(sin((lon1-lon2)*0.5))**2 )));


def qa(desitiles, nside=1024):
    """Make tiling QA plots; demonstrate usage."""
    import healpy
    theta, phi = healpy.pix2ang(nside, numpy.arange(12*nside**2))
    la, ba = phi*180./numpy.pi, 90-theta*180./numpy.pi
    m4pass = desitiles['pass'] <= 3
    m0 = desitiles['pass'] == 0
    ran, decn = logradecoffscheme(desitiles['ra'][m0], 
                                  desitiles['dec'][m0], dx=0.6, ang=24)
    tilerd = {}
    tilerd['default'] = (desitiles['ra'], desitiles['dec'])
    tilerd['new'] = (ran, decn)
    ims = {name: render(la, ba, rd[0][m4pass], rd[1][m4pass])
           for name, rd in tilerd.items()}
    pseudoindesi = ((gc_dist(la, ba, 180, 30) < 40) 
                    | (gc_dist(la, ba, 0, 5) < 30))
    from matplotlib import pyplot as p
    p.figure('Coverage Histogram')
    p.clf()
    for name, im in ims.items():
        p.hist(im[pseudoindesi].reshape(-1), range=[0, 20], bins=20, 
               histtype='step', label=name, normed=True, )
    p.gca().set_yscale('log')
    p.legend()

    adjs = {}
    for name, rd in tilerd.items():
        # just build adjacency matrix from small region, but needs
        # to at least cover several pointings
        m = m4pass & (gc_dist(rd[0], rd[1], 0, 0) < 6.)
        adjs[name] = adjacency_matrix(rd[0][m], rd[1][m])
    
    p.figure('Adjacency Matrices')
    p.clf()
    for i, (name, adjs) in enumerate(adjs.items()):
        p.subplot(len(adjs), 2, 2*i+1)
        p.imshow(adjs[0], origin='lower', aspect='equal',
                 cmap='binary', vmax=0.2)
        p.title('%s: slit block' % name)
        p.subplot(len(adjs), 2, 2*i+2)
        # scale by area of each bin
        angsizes = numpy.arange(21).astype('f4')+0.5
        angsizes = angsizes.reshape(1, -1)*angsizes.reshape(-1, 1)
        p.imshow(adjs[1]/angsizes, origin='lower', aspect='equal',
                 cmap='binary', vmax=0.1)
        p.title('%s: radial' % name)
        
        
    
