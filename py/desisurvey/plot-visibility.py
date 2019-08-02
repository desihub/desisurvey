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
int2program      = {0: 'Dark', 1: 'Gray', 2: 'Bright'}

##
tiles            = Table(fits.open('/global/cscratch1/sd/mjwilson/svdc2019c2/survey/basetiles/original/schlafly-tiles.fits')[1].data)
tiles            = tiles[tiles['IN_DESI'].quantity > 0]
tiles            = tiles[tiles['RA'].quantity > 160.]
tiles            = tiles[tiles['RA'].quantity < 280.]
tiles            = tiles[tiles['DEC'].quantity > -5.]
tiles            = tiles[tiles['DEC'].quantity < 75.]
tiles            = tiles[tiles['PASS'].quantity == 0]

tiles.sort('CENTERID')

cmap   = plt.get_cmap('viridis')
norm   = mpl.colors.Normalize()

##
for program in range(3):
  pl.clf()

  nnights            = 30
  hrs_visible        = np.loadtxt('visibility/visibility-{}-{}.txt'.format(nnights, program))

  normed_visibility  = np.sum(hrs_visible, axis=0) / nnights
  
  circles(tiles['RA'].quantity, tiles['DEC'].quantity, s=1.67, lw=1., ec='k', c=normed_visibility, alpha=0.4, cmap=cmap, norm=norm)

  pl.title(int2program[program])
 
  plt.colorbar()

  pl.xlabel(r'Right ascension [deg.]')
  pl.ylabel(r'Declination [deg.]')
  
  plt.tight_layout()
  plt.gca().invert_xaxis()

  pl.savefig('plots/visibility-{}-{}.pdf'.format(nnights, program))                                                                                                                                                                                                
