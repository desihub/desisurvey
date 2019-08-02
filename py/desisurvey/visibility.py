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


int2program = {0: 'Dark', 1: 'Gray', 2: 'Bright'}

def  whatprogram(mjd, _programs, _changes):
  changes  = list(_changes[_changes > 0.])
  programs = list(_programs[_programs > -1.]) 
  
  while (len(changes) != 0):
   if mjd >= changes[0]:  
     programs.pop(0)
     changes.pop(0)

   else:
     return  programs[0]
     
  return  programs[0]

##  Get Eddie's tiles -> remppaed to a format similar to the old tiles, e.g. in pass ordering.
##  Center ID defined for pass 5 (Gray) in this instance.  
##  tiles        = Table(fits.open('/global/cscratch1/sd/mjwilson/svdc2019c2/survey/tiles/schlafly/opall.fits')[1].data)

tiles            = Table(fits.open('/global/cscratch1/sd/mjwilson/svdc2019c2/survey/basetiles/original/schlafly-tiles.fits')[1].data)
tiles            = tiles[tiles['IN_DESI'].quantity > 0]
tiles            = tiles[tiles['RA'].quantity > 160.]
tiles            = tiles[tiles['RA'].quantity < 280.]
tiles            = tiles[tiles['DEC'].quantity > -5.]
tiles            = tiles[tiles['DEC'].quantity < 75.]
tiles            = tiles[tiles['PASS'].quantity == 0]

tiles.sort('CENTERID')

print(tiles)

##
cids             = np.unique(tiles['CENTERID'].quantity)

##  Write.                                                                                                                                                                                                     
np.savetxt('visibility/cids.txt', cids, fmt='%d')

##  ephem table duration. 
start            = date(year = 2019, month = 1,  day = 1)   ##  local_noon_on_date(datetime(year = 2020, month = 4, day = 16))
stop             = date(year = 2025, month = 12, day = 31)  ##  local_noon_on_date(datetime(year = 2020, month = 5, day = 16))

##  config derived constraints.                                                                                                                                                         
config           = desisurvey.config.Configuration(file_name='/global/cscratch1/sd/mjwilson/svdc2019c2/survey/config.yaml')
full_moon_nights = config.full_moon_nights()

first_day        = Time(config.first_day().isoformat(), format='iso').mjd
last_day         = Time(config.last_day().isoformat(), format='iso').mjd

min_altitude     = config.min_altitude().value

lat              = config.location.latitude()
lon              = config.location.longitude()
elv              = config.location.elevation()
 
avoid_bodies     = {}

for body in config.avoid_bodies.keys:
  avoid_bodies[body] = getattr(config.avoid_bodies, body)().to(u.deg)

##
mayall           = EarthLocation(lat=lat, lon=lon, height=elv)

##  Load ephem. 
dat              = Ephemerides(start, stop, restore='/global/cscratch1/sd/mjwilson/svdc2019c2/survey/ephem_2019-01-01_2025-12-31.fits')
 
##  print(dat._table.columns)
##  print(dat._table)

##  Program hours for each night. 
hrs              = dat.get_program_hours(include_twilight=True)

##
mjd0             = Time(datetime(1899, 12, 31, 12, 0, 0)).mjd
t_obj            = np.linspace(0., 1., 25)

print('\n\nHours up and hours visible for each CENTERID and noon-to-noon day.\n\n')

nnights      = 0
hrs_visible  = np.zeros(3 * len(tiles['RA'].quantity) * len(dat._table['noon'].quantity), dtype=np.float).reshape(len(dat._table['noon'].quantity), len(tiles['RA'].quantity), 3)

##  Write.                                                                                                                                                                                                      
np.savetxt('visibility/noons.txt', dat._table['noon'].quantity)

for i, noon in enumerate(dat._table['noon'].quantity):
 isonoon     = Time(noon.value, format='mjd').iso.split(' ')[0]
 
 if (noon >= first_day) & (noon <= last_day):
  nnights   += 1

  midnight   = noon.value + 0.5
  
  programs   = dat._table['programs'].quantity[i]
  changes    = dat._table['changes'].quantity[i]

  bdusk      = dat._table['brightdusk'].quantity[i]
  bdawn      = dat._table['brightdawn'].quantity[i]
  
  for j, t in enumerate(t_obj):
    mjd      = noon.value + t
    
    if (mjd < bdusk) or (mjd > bdawn):      
      continue

    program  = whatprogram(mjd, programs, changes)
    
    time     = Time(mjd, format='mjd')
    
    pos      = [SkyCoord(ra = ra * u.degree, dec = dec * u.degree, frame='icrs').transform_to(AltAz(obstime=time, location=mayall)) for ra, dec in zip(tiles['RA'], tiles['DEC'])]

    ra       = tiles['RA'].quantity
    dec      = tiles['DEC'].quantity

    az       = np.array([x.az.value   for x in pos])
    alt      = np.array([x.alt.value  for x in pos])
    airmass  = np.array([x.secz.value for x in pos])

    isin     = np.ones_like(tiles['RA'].quantity, dtype=np.float)
    
    isin[alt < min_altitude] = 0.0
    
    for body in avoid_bodies:
      bra, bdec = desisurvey.ephem.get_object_interpolator(dat._table[i], body, altaz=False)(mjd)
      too_close = desisurvey.utils.separation_matrix([bra] * u.deg, [bdec] * u.deg, ra * u.deg, dec * u.deg, avoid_bodies[body])[0]
      
      isin[too_close] = 0.0
      
    hrs_visible[i, :, program] += np.array(isin)
  
  print('\n\n', isonoon, '\n', '\n'.join('{}'.format(hrs_visible[i, :, x].astype(np.int)) for x in range(3))) 
  
##  
normed_visibility  = np.sum(hrs_visible, axis=0) / nnights

cmap   = plt.get_cmap('viridis')
norm   = mpl.colors.Normalize()

for program in range(3):
  pl.clf()

  ##  print(normed_visibility[:,program])
  
  circles(tiles['RA'].quantity, tiles['DEC'].quantity, s=1.67, lw=1., ec='k', c=normed_visibility[:,program], alpha=0.4, cmap=cmap, norm=norm)

  pl.title(int2program[program])
  
  plt.colorbar()

  pl.xlabel(r'Right ascension [deg.]')
  pl.ylabel(r'Declination [deg.]')

  plt.tight_layout()
  plt.gca().invert_xaxis()
  
  ##  pl.show()
  
  ##  Write.
  np.savetxt('visibility/visibility-{}-{}.txt'.format(nnights, program), hrs_visible[:, :, program], fmt='%.3lf')
  
print('\n\nDone.\n\n')
