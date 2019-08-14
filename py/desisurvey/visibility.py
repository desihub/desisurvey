import  ephem
import  matplotlib           as      mpl 
import  desisurvey
import  astropy.io.fits      as      fits
import  pylab                as      pl
import  numpy                as      np
import  astropy.units        as      u
import  matplotlib.pyplot    as      plt
import  matplotlib.cm        as      cm

from    astropy.table        import  Table

from    desisurvey.ephem     import  Ephemerides
from    desisurvey.utils     import  local_noon_on_date, get_date
from    datetime             import  datetime, date
from    astropy.time         import  Time
from    astropy.coordinates  import  SkyCoord, EarthLocation, AltAz
from    desitarget.geomask   import  circles
from    desisurvey           import  plan


def whatprogram(mjd, programs, changes):
  index = 0 

  while mjd >= changes[index + 1]:
    index += 1
     
  return  programs[index]

prefix           = 'op'
##  prefix       = 'y5'

verbose          = True

##  Get Eddie's tiles -> remppaed to a format similar to the old tiles, e.g. in pass ordering.
##  Center ID defined for pass 5 (Gray) in this instance.  
##  tiles        = Table(fits.open('/global/cscratch1/sd/mjwilson/svdc2019c2/survey/tiles/schlafly/opall.fits')[1].data)

tiles            = Table(fits.open('/global/cscratch1/sd/mjwilson/svdc2019c2/survey/basetiles/original/schlafly-tiles.fits')[1].data)
tiles            = tiles[tiles['IN_DESI'].quantity > 0]
tiles            = tiles[tiles['PASS'].quantity ==  0]  ##    0
'''
##  Select single pass (on which CENTERID is defined) in potential one percent area. 
tiles            = tiles[tiles['RA'].quantity >   90.]  ##  160.
tiles            = tiles[tiles['RA'].quantity <  290.]  ##  280.
tiles            = tiles[tiles['DEC'].quantity > -10.]  ##   -5.
tiles            = tiles[tiles['DEC'].quantity <  80.]  ##   75.
'''
tiles.sort('CENTERID')

##  print(tiles)

##
cids             = np.unique(tiles['CENTERID'].quantity)

##  Write.                                                                                                                                                                                                     
np.savetxt('visibility-op/cids.txt', cids, fmt='%d')

##  design_hrangle = plan.load_design_hourangle()               

##  config derived constraints.                                                                                                                                                         
config           = desisurvey.config.Configuration(file_name='/global/cscratch1/sd/mjwilson/svdc2019c2/survey/config.yaml')
full_moon_nights = config.full_moon_nights()

first_day        = config.first_day().isoformat()
last_day         = config.last_day().isoformat()

##  Five-year rewrite.
##  first_day    = '2020-06-01'   
##  last_day     = '2025-12-31'

first_mjd        = Time(first_day, format='iso').mjd
last_mjd         = Time(last_day,  format='iso').mjd

print(first_mjd)
print(last_mjd)

min_altitude     = config.min_altitude().value

lat              = config.location.latitude()
lon              = config.location.longitude()
elv              = config.location.elevation()
 
avoid_bodies     = {}
bodies           = list(config.avoid_bodies.keys)

for body in bodies:
  avoid_bodies[body] = getattr(config.avoid_bodies, body)().to(u.deg)

##  Planet exclusion evaluated at midnight, moon exclusion at each mjd. 
bodies.remove('moon')
  
##
##  mayall       = EarthLocation(lat=lat, lon=lon, height=elv)
mayall           = EarthLocation.from_geodetic(lat=lat, lon=lon, height=elv)

##
ra               = tiles['RA'].quantity
dec              = tiles['DEC'].quantity

# Calculate the maximum |HA| in degrees allowed for each tile to stay above the survey minimum altitude (plus a 5 deg padding).
# cosz_min       = np.cos(90. * u.deg - (config.min_altitude() + 5. * u.deg))
# cosHA_min      = ((cosz_min - np.sin(dec * u.deg) * np.sin(lat)) / (np.cos(dec * u.deg) * np.cos(lat))).value
# max_abs_ha     = np.degrees(np.arccos(cosHA_min))

##  ephem table duration.
start            = date(year = 2019, month =  1,  day = 1)
stop             = date(year = 2025, month = 12, day = 31)

##  Load ephem. 
dat              = Ephemerides(start, stop, restore='/global/cscratch1/sd/mjwilson/svdc2019c2/survey/v1/ephem_2019-01-01_2025-12-31.fits')
 
##  print(dat._table.columns)
##  print(dat._table)

##  Survey sim derived program hours for each night. 
hrs              = dat.get_program_hours(include_twilight=True)

for i, _hrs in enumerate(hrs.T):
  print('Night  {}:  {} Dark;  {}  Gray;  {}  Bright.'.format(i, _hrs[0], _hrs[1], _hrs[2]))

##  Choose same times as those solved for in ephem, but more finely sample than 1/hr due to twilight.   
N                = 96
dt               = 24. / N
t_obj            = np.linspace(0., 1., N + 1)

print('\n\nHours visible for each CENTERID, program and noon-to-noon day.\n\n')

nnights          = 0

##  For each tile, and each night, record the hrs visible in each program. 
hrs_visible      = np.zeros(3 * len(tiles['RA'].quantity) * len(dat._table['noon'].quantity), dtype=np.float).reshape(len(dat._table['noon'].quantity), len(tiles['RA'].quantity), 3)
moons            = []

output           = []
isonoons         = []

for i, noon in enumerate(dat._table['noon'].quantity):
 isonoon     = Time(noon.value, format='mjd').iso.split(' ')[0]

 fullmoon    = dat.is_full_moon(get_date(isonoon))         
 monsoon     = desisurvey.utils.is_monsoon(get_date(isonoon))  

 if (isonoon >= first_day) & (isonoon <= last_day):
  if fullmoon:
    print('\n\n', isonoon, '\nFULLMOON')
    continue

  if monsoon:
    print('\n\n', isonoon, '\nMONSOON')
    continue

  ##  Count the nights that made it. 
  nnights += 1
  
  ##  This ephem row made the cut. 
  output.append(i)

  ##  Isonoon for this row. 
  isonoons.append([i, isonoon, nnights])

  ##  MJD for midnight on this date. 
  midnight    = noon.value + 0.5
  
  ##  programs, changes = dat._table['programs'].quantity[i],  dat._table['changes'].quantity[i]  
  programs, changes = dat.get_night_program(get_date(isonoon), include_twilight=True, program_as_int=True)  ##  Augmented with (b)dusk, (b)dawn at either end.

  dusk        = dat._table['dusk'].quantity[i]
  dawn        = dat._table['dawn'].quantity[i]
  
  ##  Includes twilight. 
  bdusk       = dat._table['brightdusk'].quantity[i]
  bdawn       = dat._table['brightdawn'].quantity[i]

  MJD0, MJD1  = bdusk, bdawn
  LST0, LST1  = dat._table['brightdusk_LST'][i], dat._table['brightdawn_LST'][i]
  dLST        = (LST1 - LST0) / (MJD1 - MJD0)

  indices     = np.array(range(len(ra)))

  ##  Planet exclusion evaluated at midnight, moon exclusion at each mjd. 
  for body in bodies:
    bdec, bra = desisurvey.ephem.get_object_interpolator(dat._table[i], body, altaz=False)(midnight)
    too_close = desisurvey.utils.separation_matrix([bra] * u.deg, [bdec] * u.deg, ra[indices] * u.deg, dec[indices] * u.deg, avoid_bodies[body])[0]
    
    indices   = indices[~too_close]
    
  ##  On this day, enumerate over the time samples and determine which CENTERIDs are visible at each time. 
  for j, t in enumerate(t_obj):
    mjd       = noon.value + t

    ##  Include twilight currently. 
    if (mjd < bdusk) or (mjd > bdawn):      
      continue

    program   = whatprogram(mjd, programs, changes)    
    time      = Time(mjd, format='mjd') ##  UTC.
    
    pos       = SkyCoord(ra = ra[indices] * u.degree, dec = dec[indices] * u.degree, frame='icrs').transform_to(AltAz(obstime=time, location=mayall))
    alt       = pos.alt.degree  
       
    ##  az    = np.array([x.az.value  for x in pos]);  airmass = np.array([x.secz.value for x in pos])  
    ##  alt   = np.array([x.alt.value for x in pos])

    ishigh    = alt > min_altitude
    isin      = indices[ishigh]
    
    ##  Calculate the local apparent sidereal time in degrees.                                                                                                                                                 
    LST       = LST0 + dLST * (mjd - MJD0)
    hourangle = LST - ra
    
    for body in ['moon']:
      bdec, bra    = desisurvey.ephem.get_object_interpolator(dat._table[i], body, altaz=False)(mjd)
      too_close    = desisurvey.utils.separation_matrix([bra] * u.deg, [bdec] * u.deg, ra[isin] * u.deg, dec[isin] * u.deg, avoid_bodies[body])[0]
    
      isin         = isin[~too_close]

      moons.append([bra, bdec])
      
    ##
    hrs_visible[i, isin, program] += dt
    
  print('\n\n', isonoon, '\n', '\n'.join('{}'.format(hrs_visible[i, :, x].astype(np.int)) for x in range(3))) 
  
  for program in range(3):
    np.savetxt('visibility-{}/visibility-nofullmoon-{}-{}.txt'.format(prefix, nnights, program), hrs_visible[output, :, program])
    
np.savetxt('visibility-{}/moons.txt'.format(prefix), np.array(moons), fmt='%.4lf')

with open('visibility-{}/noons.txt'.format(prefix), 'w') as f:
    for item in isonoons:
        f.write("%s\n" % item)

print('\n\nDone.\n\n')
