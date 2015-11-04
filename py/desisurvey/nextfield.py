#- Provided by Data Systems to be called by DOS

import math
import time
import numpy as np
#from astropy.time import Time
#from astropy.coordinates import SkyCoord
#from astropy.coordinates import ICRS, FK5, AltAz, EarthLocation
#from astropy.coordinates import Angle, Latitude, Longitude
from astropy.table import Table
#import astropy.units as u

def get_next_field(dateobs, skylevel, seeing, transparency, obsplan,
    programname=None):
    """
    Returns structure with information about next field to observe.
    
    Args:
        dateobs (float): start time of observation in UTC (TAI).
            Could be past, present, or future.
        skylevel: current sky level [counts/s/cm^2/arcsec^2]
        seeing: current astmospheric seeing PSF FWHM [arcsec]
        transparency: current atmospheric transparency
        obsplan: filename containing the nights observing plan
        programname (string, optional): if given, the output result will be for
            that program.  Otherwise, next_field_selector() chooses the
            program based upon the current conditions.

    Returns dictionary with keys
        tileid: tile ID [integer]
            --> DOS should just add this to the raw data header
        programname: DESI (or other) program name, e.g. "Dark Time Survey",
            "Bright Galaxy Survey", etc.
            --> DOS should just add this to the raw data header
        telera, teledec: telescope central pointing RA, dec [J2000 degrees]
        exptime: expected exposure time [seconds]
        maxtime: maximum allowable exposure time [seconds]
        fibers: dictionary with the following keys, each of which contains
            a list of 5000 values for each of the positioners
            - ra: RA for each fiber [J2000 degrees]
            - dec: dec for each fiber [J2000 degrees]
            - lambdaref: wavelength to optimize each positioner [Angstrom]
        gfa: dictionary with the following keys, each of which contains
            a list of values for objects detectecable by the GFAs, including
            border regions in RA,dec to assist with acquisition
            - id : ID of GFA for this object
            - ra, dec : RA and dec for each object [J2000 degrees]
            - objtype : 'point', 'extended', 'sky'
                --> point sources with okguide=True can be used for guiding;
                    knowledge of the existence of extended sources may help
                    with acquisition; sky locations are large enough to be
                    used for estimating sky backgrounds.
            - okguide : True if good for guiding
            - mag : magnitude [SDSS r-band AB magnitude]
                --> or a flux instead?

        Additional keys may be present and should be ignored
        
    e.g. result['fibers']['ra'] gives the 5000 RA locations for the fibers
    
    Notes:
      * get_next_field() will calculate the LST and moon phase/location
        based upon the input datetime.
      * skylevel, seeing, and transparency are in the filter of the guider.
      * The contents of the returned dictionary should be *everything* needed
        as input to point the telescope and take an exposure.  If that isn't
        true, we need to add more.  An ancillary/test/commissioning program
        that defines all of these quantities (e.g. in a JSON file)
        should be sufficient to take observations.
      * previoustiles is a required input rather than having get_next_field()
        query ObsDB to get the history.  Two reasons:
        - Easier to test without requiring live database
        - Decouples code dependencies
      * result['fibers'] will be pre-calculated by fiber assignment;
        DOS shouldn't care as long as get_next_field is fast (<1 sec).
      * Current expectation is the ObsDB/DOS only tracks the past, i.e. what
        observations were taken, but not which observations we would like to
        take in the future.  As such, get_next_field() will need to look up
        the DESI tiling (currently in desimodel/data/footprint/desi-tiles.*)
        and a list of overrides for tiles that were observed by deemed bad
        and need to be redone (details TBD).
        DOS shouldn't care about those details.
        
    TBD:
      * Error handling: if the request is impossible (e.g. the sun is up),
        should this raise an exception?  Or return a default zenith answer
        with some calib programname?  Or?
    """
    
    """ 
        Below is an algorithm for calculating the local apparent sidereal time. 
        astropy should be able to this (I think), but I keep getting an error when I try 
        to use those functions (and I'm pretty sure I'm importing all the necessary 
        components of astropy). A bunch of lines have been commented out which were 
        used for testing purposes. 
    """
                
    #tobs = Time(dateobs, format='jd', scale='ut1')
    
    #Find the Julian date of the previous midnight
    if (dateobs-math.floor(dateobs) >= 0.5):
        jd_0 = math.floor(dateobs)+0.5
    elif (dateobs-math.floor(dateobs) < 0.5):
        jd_0 = math.floor(dateobs)-0.5
        
    d_0 = jd_0-2451545.0 #Difference between last Julian midnight and J2000
    d = dateobs-2451545.0 #Difference between observation date and J2000
    t = d_0/36525 #Fraction of Julian century that's past since J2000
    
    #Calculate the sideral time in Greenwich for the last Julian midnight
    gmst_0 = 100.4606184+36000.77005361*t+0.00038793*t**2-2.6E-08*t**3
    
    #Add correction for the number of hours that have past since midnight
    gmst = gmst_0+0.25068447733746215*(dateobs-jd_0)*24*60
    
    #Calculate the equation of equinoxes
    l = 280.47+0.98565*d
    omega = 125.04-0.052954*d
    del_psi = -0.000319*math.sin(omega*math.pi/180)-0.00024*math.sin(2*l*math.pi/180)
    e = 23.4393-0.0000004*d
    
    eqeq = del_psi*math.cos(e*math.pi/180)
    
    #Correct with the equation of equinoxes to get the current apparent sidereal time
    gast = gmst+eqeq
    
    #Add the longitude of the observatory to get the local sidereal time
    last = gast-111.5984796
        
    #Shift the local sidereal time into the range of 0 to 360 degrees
    if last >= 360:
        n = math.floor(last/360)
        last = last-360*n
        
    #- Find the position of the Moon using pyephem. After the compute statement below,
    #- many attributes of the Moon can be accessed including
    #-      1. Right Ascension/Declination (epoch of date) - moon.g_ra, moon.g_dec
    #-         Right Ascension/Declination (epoch specified) - moon.a_ra, moon.a_dec
    #-      2. Phase - moon.phase (percent illimunation)
    #- In order to calculate the Moon's attribute for dateobs, it is necessary to 
    #- convert to the Dublin Julian date which can be done by subtracting 2415020 from
    #- the Julian date.
    #moon = ephem.Moon() #- Setup the Moon object
    #moon.compute(dateobs-2415020.0, epoch=dateobs-2415020.0) #- Compute for dateobs
    
    #Loads the tiles
    tiles_array = Table.read(obsplan, hdu=1)

    nextfield = 0
    
    #- Perform coarse trim of tiles with mismatched coordinates
    igood = np.where( (last-5 <= tiles_array['BEG_OBS']) & (tiles_array['BEG_OBS'] <= last+5) )[0]
    tiles_array = tiles_array[igood]
    
    #- Setup astropy SkyCoord objects
    #tiles = SkyCoord(ra=tiles_array['RA']*u.deg, dec=tiles_array['DEC']*u.deg, frame='icrs')
    
    #- Determine the current epoch from the input date and store as string
    #epoch = "J" + str(round(tobs.decimalyear, 3))
    
    #- Transform the RA and Dec to JNow write the transformed data back to the shorthand
    #tiles = tiles.transform_to(FK5(equinox=epoch))

    #- Trim tiles_array to those within 15 degrees of the meridian
    #igood = np.where( (last-15 <= tiles.ra.value) & (tiles.ra.value <= last+15) )[0]
    #tiles_array = tiles_array[igood]
    #tiles = tiles[igood]
    
    #- Remove previously observed tiles
    #notobs = np.in1d(tiles_array['TILEID'], previoustiles, invert=True)
    #inotobs = np.where(obs == False)
    #tiles_array = tiles_array[notobs]

    #- will need to explicitly handle the case of running out of tiles later
    assert len(tiles_array) > 0
        
    #- shorthand    
    #ra = tiles.ra.value
    #dec = tiles.dec.value

    #- calculate the hour angle for those tiles
    #ha = (last - ra + 360) % 360
    #assert np.min(ha) >= 0
    #assert np.max(ha) <= 360.0

    #alt = (np.arcsin(np.sin(dec*math.pi/180)
    #                 *np.sin(31.9614929*math.pi/180)
    #                 +np.cos(dec*math.pi/180)
    #                 *np.cos(31.9614929*math.pi/180)
    #                 *np.cos(ha*math.pi/180)))*(180/math.pi)

    #- Find the lowest dec tile; this could also be done faster with
    #- array calculations instead of a loop
    ibest = -1
    priority = 100000
    for i in range(len(tiles_array)):
        if tiles_array[i]['PRIORITY'] < priority:
            ibest = i
            priority = tiles_array[i]['PRIORITY']
            
    assert ibest >= 0
                
    #Create dictionary with information that is needed to point the telescope.
    #Currently the exptime and maxtime are just place holder values and fibers and gfa
    #dictionaries are just empty.
    results = {
        'tileid':int(tiles_array[ibest]['TILEID']),
        'programname':'DESI',
        'telera':float(tiles_array[ibest]['RA']),
        'teledec':float(tiles_array[ibest]['DEC']),
        'exptime':tiles_array[ibest]['OBSTIME'],
        'maxtime':(tiles_array[ibest]['END_OBS']-last)*3600/15,
        'fibers':{},
        'gfa':{},
        }
    
    print last
    #Return the dictionary
    return results

""" The lines below allow the function to be tested by itself with the user
inputting a Julian date of observation. They also calculate the execution time for
purposes of optimizing."""
        
            

#dateobs = float(raw_input('Enter the date of observation: '))
#skylevel = 0
#seeing = 1.1
#transparency = 0
#obsplan = 'toyplan.fits'
#programname = 'DESI'
#start_time = time.time()
#next_field = get_next_field(dateobs, skylevel, seeing, transparency, obsplan, programname)

#print("Total execution time: %s seconds" % (time.time()-start_time))
#print next_field
