#- Provided by Data Systems to be called by DOS

import math
import time
from PyAstronomy import pyasl
from astropy import coordinates
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, FK5, AltAz, EarthLocation
from astropy.coordinates import Angle, Latitude, Longitude
import astropy.units as u

def get_next_field(dateobs, skylevel, seeing, transparency, previoustiles,
    programname=None):
    """
    Returns structure with information about next field to observe.
    
    Args:
        dateobs (float): start time of observation in UTC (TAI).
            Could be past, present, or future.
        skylevel: current sky level [counts/s/cm^2/arcsec^2]
        seeing: current astmospheric seeing PSF FWHM [arcsec]
        transparency: current atmospheric transparency
        previoustiles: list of tile IDs previously observed.
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
    
    tobs = Time(dateobs, format='jd', scale='utc')
    
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
        
    #hour = math.floor(LAST/15)
    #minute = math.floor((LAST/15-hour)*60)
    #second = ((LAST/15-hour)*60-minute)*60
    
    #print( str(hour) + "h " + str(minute) + "m " + str(second) +"s")

    #print("dateobs = " + str(dateobs))
    #print("JD_0 = " + str(JD_0))
    #print("D_0 = " + str(D_0))
    #print("D = " + str(D))
    #print("T = " + str(T))
    #print("H = " + str(H))
    #print("GMST = " + str(GMST))
    #print("LAST = " + str(LAST))
    
    #Use PyAstronomy to calculate the position of the Sun.
    #pos_sun = pyasl.sunpos(dateobs,full_output=True)
    pos_sun = coordinates.get_sun(tobs)
    
    print pos_sun
    
    print pos_sun.ra.value
    print pos_sun.dec.value
    
    #print("ra_sun = " + str(float(pos_sun[1])) + " dec_sun = " + str(float(pos_sun[2])))
    
    #Check to see if the Sun is up.
    ra_sun = pos_sun.ra.value
    dec_sun = pos_sun.dec.value
    
    print ra_sun
    print dec_sun
    
    ha_sun = last - ra_sun
    if (ha_sun < 0):
        ha_sun = ha_sun+360
    if (ha_sun > 360):
        ha_sun = ha_sun-360
    #print("ha_sun = " + str(ha_sun))
    
    #Calculate the altitude of the Sun to determine if it is up
    alt_sun = (math.asin(math.sin(dec_sun*math.pi/180)
                         *math.sin(31.9614929*math.pi/180)
                         +math.cos(dec_sun*math.pi/180)
                         *math.cos(31.9614929*math.pi/180)
                         *math.cos(ha_sun*math.pi/180)))*(180/math.pi)
    
    #print("alt_sun = " + str(alt_sun))
    
    #Print warning if the Sun is up. We may decide this should do more than just warn
    if (alt_sun >= 0):
        print("WARNING: The sun is currently up!")
        
    #Find the position of the Moon using PyAstronomy
    pos_moon = pyasl.moonpos(dateobs)
    
    ra_moon = float(pos_moon[0])
    dec_moon = float(pos_moon[1])
    
    print("ra_moon = " + str(ra_moon) + " dec_moon = " + str(dec_moon))
    
    #Find the phase of the moon using PyAstronomy
    moon_phase = pyasl.moonphase(dateobs)
    phase = float(moon_phase)
    
    print(phase)
    
    #Opens a file containing information about all of the DESI tiles. Not sure if
    #this is the best way to get that information?
    tiles_file = open("desi-tiles-full.par.txt","r")
    
    #Declare some variables to store data from the file
    tiles = []
    idnum = []
    ra = []
    dec = []
    passnum = []
    in_desi = []
    ebv_med = []
    airmass = []
    exposefac = []
    i = 0
    j = 0
    possibletiles = 0
    impossibletiles = 0
    mindec = 100.0
    nextfield = 0
    #Read the data from the file and find the next field
    for line in tiles_file:
        tiles.append(line)
        c = tiles[j].split(" ")
        if (c[0] == "STRUCT1"):
            idnum.append(int(c[1]))
            ra.append(float(c[2]))
            dec.append(float(c[3]))
            passnum.append(int(c[4]))
            in_desi.append(int(c[5]))
            ebv_med.append(float(c[6]))
            airmass.append(float(c[7]))
            exposefac.append(float(c[8]))
            ha = last-ra[i]
            if ha < 0:
                ha = ha + 360
            if ha > 360:
                ha = ha - 360
            alt = (math.asin(math.sin(dec[i]*math.pi/180)
                             *math.sin(31.9614929*math.pi/180)
                             +math.cos(dec[i]*math.pi/180)
                             *math.cos(31.9614929*math.pi/180)
                             *math.cos(ha*math.pi/180)))*(180/math.pi)
            if (alt >= 0):
                if (dec[i] < mindec and ra[i] >= last-1 and ra[i] <= last+1):
                    mindec = dec[i]
                    nextfield = idnum[i]
            i = i + 1
        j = j + 1
    
    tiles_file.close()    
    
    #raise NotImplementedError
    
    #Create dictionary with information that is needed to point the telescope
    next_field = {'tileid':idnum[nextfield-1], 'programname':'DESI', 'telera':ra[nextfield-1], 'teledec':dec[nextfield-1], 'exptime':1800, 'maxtime':2000, 'fibers':{}, 'gfa':{}}
    
    #Return the dictionary
    return next_field

""" The lines below allow the function to be tested by itself with the user
inputting a Julian date of observation. They also calculate the execution time for
purposes of optimizing."""

dateobs = float(raw_input('Enter the date of observation: '))
skylevel = 0
transparency = 0
previoustiles = []
programname = 'DESI'
start_time = time.time()
next_field = get_next_field(dateobs, skylevel, transparency, previoustiles, programname)

print("Total execution time: %s seconds" % (time.time()-start_time))
print next_field
