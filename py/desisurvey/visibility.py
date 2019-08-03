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

def whatprogram(mjd, _programs, _changes):
  changes  = list(_changes[_changes > 0.])
  programs = list(_programs[_programs > -1.]) 
  
  while (len(changes) != 0):
   if mjd >= changes[0]:  
     programs.pop(0)
     changes.pop(0)

   else:
     return  programs[0]
     
  return  programs[0]


verbose          = True

##  Get Eddie's tiles -> remppaed to a format similar to the old tiles, e.g. in pass ordering.
##  Center ID defined for pass 5 (Gray) in this instance.  
##  tiles        = Table(fits.open('/global/cscratch1/sd/mjwilson/svdc2019c2/survey/tiles/schlafly/opall.fits')[1].data)

tiles            = Table(fits.open('/global/cscratch1/sd/mjwilson/svdc2019c2/survey/basetiles/original/schlafly-tiles.fits')[1].data)
tiles            = tiles[tiles['IN_DESI'].quantity > 0]
tiles            = tiles[tiles['RA'].quantity > 160.]  ##  160.
tiles            = tiles[tiles['RA'].quantity < 280.]  ##  280.
tiles            = tiles[tiles['DEC'].quantity > -5.]  ##   -5.
tiles            = tiles[tiles['DEC'].quantity < 75.]  ##   75.
tiles            = tiles[tiles['PASS'].quantity == 0]  ##    0

tiles.sort('CENTERID')

print(tiles)

##
cids             = np.unique(tiles['CENTERID'].quantity)

##  Write.                                                                                                                                                                                                     
np.savetxt('visibility/cids.txt', cids, fmt='%d')

##  config derived constraints.                                                                                                                                                         
config           = desisurvey.config.Configuration(file_name='/global/cscratch1/sd/mjwilson/svdc2019c2/survey/config.yaml')
full_moon_nights = config.full_moon_nights()

first_day        = config.first_day().isoformat()
last_day         = config.last_day().isoformat()

first_mjd        = Time(first_day, format='iso').mjd
last_mjd         = Time(last_day,  format='iso').mjd

min_altitude     = config.min_altitude().value

lat              = config.location.latitude()
lon              = config.location.longitude()
elv              = config.location.elevation()
 
avoid_bodies     = {}
bodies           = list(config.avoid_bodies.keys)

for body in bodies:
  avoid_bodies[body] = getattr(config.avoid_bodies, body)().to(u.deg)

##  You want this as the moon is the body that excludes the greatest number of tiles. 
assert  bodies[0] == 'moon'
  
##
mayall           = EarthLocation(lat=lat, lon=lon, height=elv)

##  ephem table duration.                                                                                                                                                                                                                                                                                                                                              
start            = date(year = 2019, month =  1,  day = 1)
stop             = date(year = 2025, month = 12, day = 31)

##  Load ephem. 
dat              = Ephemerides(start, stop, restore='/global/cscratch1/sd/mjwilson/svdc2019c2/survey/ephem_2019-01-01_2025-12-31.fits')
 
##  print(dat._table.columns)
##  print(dat._table)

##  Program hours for each night. 
hrs              = dat.get_program_hours(include_twilight=False)

##  Choose same times as those solved for in ephem. 
N                = 96
dt               = 24. / N
t_obj            = np.linspace(0., 1., N + 1)

print('\n\nHours visible for each CENTERID, program and noon-to-noon day.\n\n')

nnights          = 0
hrs_visible      = np.zeros(3 * len(tiles['RA'].quantity) * len(dat._table['noon'].quantity), dtype=np.float).reshape(len(dat._table['noon'].quantity), len(tiles['RA'].quantity), 3)
output           = []
isonoons         = []

for i, noon in enumerate(dat._table['noon'].quantity):
 isonoon     = Time(noon.value, format='mjd').iso.split(' ')[0]

 fullmoon    = dat.is_full_moon(desisurvey.utils.get_date(isonoon))
 moonsoon    = desisurvey.utils.is_monsoon(desisurvey.utils.get_date(isonoon))
 
 if (isonoon >= first_day) & (isonoon <= last_day):
  if fullmoon | moonsoon:
    print('\n\n', isonoon, '\nFULLMOON')
    continue

  output.append(i)
  isonoons.append(isonoon)
  
  midnight    = noon.value + 0.5
  
  programs    = dat._table['programs'].quantity[i]
  changes     = dat._table['changes'].quantity[i]

  dusk        = dat._table['dusk'].quantity[i]
  dawn        = dat._table['dawn'].quantity[i]
  
  ##  Includes twilight. 
  bdusk       = dat._table['brightdusk'].quantity[i]
  bdawn       = dat._table['brightdawn'].quantity[i]

  nnights    += 1
  
  for j, t in enumerate(t_obj):
    mjd       = noon.value + t
    
    if (mjd < dusk) or (mjd > dawn):      
      continue

    program   = whatprogram(mjd, programs, changes)    
    time      = Time(mjd, format='mjd') ##  UTC.

    if verbose:
      print(isonoon, time, program)  
    
    pos       = [SkyCoord(ra = ra * u.degree, dec = dec * u.degree, frame='icrs').transform_to(AltAz(obstime=time, location=mayall)) for ra, dec in zip(tiles['RA'], tiles['DEC'])]

    ra        = tiles['RA'].quantity
    dec       = tiles['DEC'].quantity

    az        = np.array([x.az.value   for x in pos])
    alt       = np.array([x.alt.value  for x in pos])  ##  airmass = np.array([x.secz.value for x in pos])

    isin      = np.zeros_like(tiles['RA'].quantity, dtype=np.float)

    sel       = alt > min_altitude
    indices   = np.where(sel == True)[0]
    
    for body in bodies:
      bra, bdec    = desisurvey.ephem.get_object_interpolator(dat._table[i], body, altaz=False)(mjd)
      too_close    = desisurvey.utils.separation_matrix([bra] * u.deg, [bdec] * u.deg, ra[indices] * u.deg, dec[indices] * u.deg, avoid_bodies[body])[0]
    
      indices      = indices[~too_close]

    ##  
    isin[indices]  = 1.0
      
    hrs_visible[i, :, program] += np.array(isin) * dt
    
  print('\n\n', isonoon, '\n', '\n'.join('{}'.format(hrs_visible[i, :, x].astype(np.int)) for x in range(3))) 
  '''
  for program in range(3):
    np.savetxt('visibility/visibility-nofullmoon-{}-{}.txt'.format(nnights, program), hrs_visible[output, :, program], fmt='%.3lf')
  '''
  
##  
normed_visibility  = np.sum(hrs_visible, axis=0) / nnights
'''
with open('visibility/noons.txt', 'w') as f:
    for item in isonoons:
        f.write("%s\n" % item)
'''  
print('\n\nDone.\n\n')
