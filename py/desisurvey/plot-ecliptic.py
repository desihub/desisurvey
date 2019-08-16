import  matplotlib           as      mpl

import  desisurvey
import  astropy.io.fits      as      fits
import  pylab                as      pl
import  numpy                as      np
import  ephem
import  astropy.units        as      u
import  matplotlib.pyplot    as      plt
import  matplotlib.cm        as      cm

from    astropy.table        import  Table

from    desisurvey.ephem     import  Ephemerides
from    desisurvey.utils     import  local_noon_on_date
from    datetime             import  datetime, date
from    astropy.time         import  Time
from    astropy.coordinates  import  SkyCoord, EarthLocation, AltAz
from    desitarget.geomask   import  circles

##
program          =  'Gray'
program2int      = {'Dark': 0, 'Gray': 1, 'Bright': 2}

tiles            = Table(fits.open('/global/cscratch1/sd/mjwilson/svdc2019c2/survey/basetiles/original/schlafly-tiles.fits')[1].data)
tiles            = tiles[tiles['IN_DESI'].quantity > 0]
tiles            = tiles[tiles['PASS'].quantity ==  0]  ##    0    
'''
tiles            = tiles[tiles['RA'].quantity >   90.]  ##  160.                                                                                                                                                                                                                                                           
tiles            = tiles[tiles['RA'].quantity <  290.]  ##  280.                                                                                                                                                                                                                                                           
tiles            = tiles[tiles['DEC'].quantity > -10.]  ##   -5.                                                                                                                                             
tiles            = tiles[tiles['DEC'].quantity <  80.]  ##   75.                                                                                                                                          
'''
'''
tiles.sort('CENTERID')

cmap   = plt.get_cmap('viridis')
norm   = mpl.colors.Normalize()

##

pl.clf()

nnights            = 1642
hrs_visible        = np.loadtxt('visibility/visibility-nofullmoon-{}-{}.txt'.format(nnights, program2int[program]))

normed_visibility  = np.sum(hrs_visible, axis=0) / nnights

##  Wrap ra. 
##  tiles['RA']       -= 90.
##  tiles['RA'][tiles['RA'] < 0.0] += 360.

circles(tiles['RA'].quantity, tiles['DEC'].quantity, s=1.67, lw=1., ec='k', c=normed_visibility, alpha=0.4, cmap=cmap, norm=norm)
'''
##  Plot ecliptic
moons = np.loadtxt('moons.txt')
pl.plot(moons[:,0], moons[:,1], 'ko', marker=2)
##  circles(moons[:,0], moons[:,1], s=50., fc=None, ec='k')

pl.xlim(10., 360.)
pl.ylim(-20.,  80.)

pl.title(program)
 
pl.xlabel(r'Right ascension [deg.]')
pl.ylabel(r'Declination [deg.]')

plt.tight_layout()
plt.gca().invert_xaxis()

pl.show()
  
##  pl.savefig('plots/visibility-nofullmoon-{}-{}.pdf'.format(nnights, program2int[program]))                                                                                                                                                                                                
