#- Provided by Data Systems to be called by DOS

import math
import time
from PyAstronomy import pyasl
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
    if dateobs-math.floor(dateobs) >= 0.5:
        JD_0 = math.floor(dateobs)+0.5
    elif dateobs-math.floor(dateobs) < 0.5:
        JD_0 = math.floor(dateobs)-0.5
        
    D_0 = JD_0-2451545.0
    D = dateobs-2451545.0
    T = D_0/36525
    H = 24*(dateobs-math.floor(dateobs))
    H_min = H*60
    
    GMST_0 = 100.4606184+36000.77005361*T+0.00038793*T**2-2.6E-08*T**3
    
    GMST = GMST_0+0.25068447733746215*(dateobs-JD_0)*24*60
    
    L = 280.47+0.98565*D
    Omega = 125.04-0.052954*D
    delPsi = -0.000319*math.sin(Omega*math.pi/180)-0.00024*math.sin(2*L*math.pi/180)
    E = 23.4393-0.0000004*D
    
    eqeq = delPsi*math.cos(E*math.pi/180)
    
    GAST = GMST+eqeq
    
    LAST = GAST-111.5984796
        
    if LAST >= 360:
        n = math.floor(LAST/360)
        LAST = LAST-360*n
        
    hour = math.floor(LAST/15)
    minute = math.floor((LAST/15-hour)*60)
    second = ((LAST/15-hour)*60-minute)*60
    
    print( str(hour) + "h " + str(minute) + "m " + str(second) +"s")

    #print("dateobs = " + str(dateobs))
    #print("JD_0 = " + str(JD_0))
    #print("D_0 = " + str(D_0))
    #print("D = " + str(D))
    #print("T = " + str(T))
    #print("H = " + str(H))
    #print("GMST = " + str(GMST))
    #print("LAST = " + str(LAST))
    
    pos_sun = pyasl.sunpos(dateobs,full_output=True)
    
    #print("RA_sun = " + str(float(pos_sun[1])) + " Dec_sun = " + str(float(pos_sun[2])))
    
    RA_sun = float(pos_sun[1])
    Dec_sun = float(pos_sun[2])
    HA_sun = LAST - RA_sun
    if (HA_sun < 0):
        HA_sun = HA_sun+360
    if (HA_sun > 360):
        HA_sun = HA_sun-360
    #print("HA_sun = " + str(HA_sun))
    ALT_sun = (math.asin(math.sin(Dec_sun*math.pi/180)*math.sin(31.9614929*math.pi/180)+math.cos(Dec_sun*math.pi/180)*math.cos(31.9614929*math.pi/180)*math.cos(HA_sun*math.pi/180)))*(180/math.pi)
    #print("ALT_sun = " + str(ALT_sun))
    
    if (ALT_sun >= 0):
        print("WARNING: The sun is currently up!")
        
    pos_moon = pyasl.moonpos(dateobs)
    
    RA_moon = float(pos_moon[0])
    Dec_moon = float(pos_moon[1])
    
    print("RA_moon = " + str(RA_moon) + " Dec_moon = " + str(Dec_moon))
    
    moonPhase = pyasl.moonphase(dateobs)
    phase = float(moonPhase)
    
    print(phase)
    
    tiles_file = open("desi-tiles-full.par.txt","r")
    #uptiles_file = open("PossibleTiles.txt","w")
    #downtiles_file = open("ImpossibleTiles.txt","w")
    
    Tiles = []
    idnum = []
    ra = []
    dec = []
    passnum = []
    in_desi = []
    ebv_med = []
    airmass = []
    exposefac = []
    i=0
    j=0
    possibletiles = 0
    impossibletiles = 0
    mindec = 100.0
    nextfield = 0
    for line in tiles_file:
        Tiles.append(line)
        c = Tiles[j].split(" ")
        if (c[0] == "STRUCT1"):
            idnum.append(int(c[1]))
            ra.append(float(c[2]))
            dec.append(float(c[3]))
            passnum.append(int(c[4]))
            in_desi.append(int(c[5]))
            ebv_med.append(float(c[6]))
            airmass.append(float(c[7]))
            exposefac.append(float(c[8]))
            HA = LAST-ra[i]
            if HA < 0:
                HA = HA + 360
            if HA > 360:
                HA = HA - 360
            ALT = (math.asin(math.sin(dec[i]*math.pi/180)*math.sin(31.9614929*math.pi/180)+math.cos(dec[i]*math.pi/180)*math.cos(31.9614929*math.pi/180)*math.cos(HA*math.pi/180)))*(180/math.pi)
            if (ALT >=0):
                #possibletiles = possibletiles + 1
                #uptiles_file.write(str(idnum[i]) + " " + str(ra[i]) + " " + str(dec[i]) + " " + str(HA) + " " + str(ALT) + "\n")
                if (dec[i] < mindec and ra[i] >= LAST-1 and ra[i] <= LAST-1):
                    mindec = dec[i]
                    nextfield = idnum[i]
            #else:
                #impossibletiles = impossibletiles + 1
                #downtiles_file.write(str(idnum[i]) + " " + str(ra[i]) + " " + #str(dec[i]) + " " + str(HA) + " " + str(ALT) + "\n")
            i = i + 1
        j = j + 1
    
    tiles_file.close()
    
    #print("Total number of tiles in the file: " + str(i))
    
    #print("The number of tiles that are possible to observe is: " + str(possibletiles))
    #print("The number of tiles that are not possible to observer is: " + str(impossibletiles))
    
    #print("The next field that should be observed is: " + str(nextfield) + "\n")
    #print(str(idnum[nextfield-1]) + " " + str(ra[nextfield-1]) + " " + str(dec[nextfield-1]) + " " + str(passnum[nextfield-1]) + " " + str(in_desi[nextfield-1]) + " " + str(ebv_med[nextfield-1]) + " " + str(airmass[nextfield-1]) + " " + str(exposefac[nextfield-1]))
    #raise NotImplementedError
    next_field = {'tileid':idnum[nextfield-1], 'programname':'DESI', 'telera':ra[nextfield-1], 'teledec':dec[nextfield-1], 'exptime':1800, 'maxtime':2000, 'fibers':{}, 'gfa':{}}
    
    #c = SkyCoord(ra=ra[nextfield-1]*u.deg,dec=dec[nextfield-1]*u.deg,equinox="J2000")
    
    #print c
    
    #c = c.transform_to(FK5(equinox="J2015.5"))
    
    #print c
    
    #TeleLoc = EarthLocation.from_geodetic(lat=31.9614929*u.deg,lon=-111.5984796*u.deg)
    
    #print TeleLoc
    
    #print c.transform_to(AltAz(dateobs, TeleLoc))
    
    #c = c.transform_to(FK5(equinox="J2000"))
    
    #print c
    
    return next_field

dateobs = float(raw_input('Enter the date of observation: '))
skylevel = 0
transparency = 0
previoustiles = []
programname = 'DESI'
start_time = time.time()
next_field = get_next_field(dateobs, skylevel, transparency, previoustiles, programname)

print("Total execution time: %s seconds" % (time.time()-start_time))
print next_field
