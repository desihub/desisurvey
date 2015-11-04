import numpy as np
from pyslalib import slalib
import atmoModel
import seeingModel
import skyModel
import nextfield

class rs(object):
    """
    Usage:
        rs=runSurvey.rs(jdBegin =  2458728.)  # create a survey
        rs.runDay()                           # one one nights worth
    Arguments:
        jdBegin =  2458728.   # beginning julian date
        ndays = 1             # how many days to run
    Returns:
        weatherfile: an ascii file containing :
            mjd, lst,
            analytic sky, seeing, transmission
            actual sky, seeing, transmission
    Notes:
    TBD:
            jdBegin =  2458728.  #- Sept 1 2019 

    This code relies on pyslalib.
    Instructions for installation:
# pyslalib: python over industrial grade spherical astronomy code slalib
# https://github.com/scottransom/pyslalib
# % wget  https://github.com/scottransom/pyslalib/archive/master.zip
# % unzip master.zip
# % cd pyslalib-master
# % make
# % python setup.py install --home=$WORK_DIR/python-home-stash/
# % python test/test_slalib.py
# % PYTHONPATH=$PYTHONPATH:$WORK_DIR/python-home-stash/ ;export PYTHONPATH
    """
    def __init__(self, jdBegin =  2458728., ndays = 1., 
            weatherfile= "../../examples/weather.txt", verbose=False) :
        self.verbose      = verbose
        kpno_lat          =   31.963
        kpno_lon          = -111.600
        kpno_height       = 2120.

        # MJD = JD - 2400000.5
        self.mjd          = jdBegin -  2400000.5
        self.date         = slalib.sla_djcl(self.mjd)

        self.degToRad     = 0.0174532925199
        self.lat          = kpno_lat*self.degToRad
        self.lon          = kpno_lon*self.degToRad
        self.height       = kpno_height

        # observational astronomy
        self.lst          = self.mjdToLST(self.mjd, self.lon)
        self.sunData      = self.getSolarPosition (self.mjd, self.lon, self.lat, self.lst)
        self.sunZD        = self.sunData[2]
        self.moonData     = self.getLunarPosition (self.mjd, self.lon, self.lat, self.lst)
        self.moonZD       = self.moonData[2]
        self.moonPhase    = self.getLunarPhase()
        self.moonSep      = np.nan
        self.ha,self.zd   = np.nan, np.nan
        self.airmass      = np.nan

        # weather file 
        self.weatherfile  = weatherfile


    def runDay (self) :
        # obs plan is to be created by afternoon work
        self.obsplan      = "../../examples/toyplan.fits"
        fd = open(self.weatherfile,"w")
        fd.write("# mjd lst in_sky in_see in_trans ")
        fd.write("real_sky real_see real_trans\n")

        afterSunset = False
        startTime = self.mjd
        endTime  = np.trunc(startTime + 1.)+0.5  ;# end of this day
        readyToExpose = startTime

        for time in np.arange(startTime, endTime, 1./24./60.) :
            if time < readyToExpose : continue
            self.mjd     = time
            self.lst     = self.mjdToLST(self.mjd, self.lon)
            sunIsUp      = self.sunCalculations () 
            if sunIsUp: continue

            if not afterSunset :
                print "do the afternoon work"
                afterSunset = True

            transparency, seeing, skylevel = self.guiderModel()
            next = nextfield.get_next_field(self.mjd, \
                    skylevel, seeing, transparency, self.obsplan)
            foundtile = next["foundtile"] 
            if foundtile :
                ra = next["telera"] 
                dec = next["teledec"] 
                exptime = next["exptime"]
                maxtime = next["maxtime"]
                self.setTimeAndPosition(self.mjd, ra, dec)
                print "fake exposure at ra,dec, exptime",
                print ra,dec,exptime, "at hour ",(time-startTime)*24.

                real_transparency, real_seeing, real_skylevel = \
                        self.guiderModel()
                lst = self.lst*360./2/np.pi
                if lst >= 360: lst = lst-360.
                fd.write("{:.2f} {:.2f}  {:.1f} {:.2f} {:.2f} ".format(
                    self.mjd, lst, skylevel, seeing,transparency))
                fd.write("{:.1f} {:.2f} {:.2f} \n".format(
                    real_skylevel, real_seeing, real_transparency))
            else :
                exptime = 60.
                self.setTimeAndPosition(self.mjd, self.lst, self.lat)
            readyToExpose = self.mjd + exptime/(3600.*24.)
        fd.close()
        print "done"

    def get_next_field(self, mjd, sky, seeing, transparancy, file) :
        next = dict()
        next["telera"] = 0.0
        next["teledec"] = 0.0
        next["exptime"] = 1800.
        next["maxtime"] = 3600.
        return next

    # change to a new time
    def setTimeAndPosition (self, mjd, ra, dec) :
        self.mjd          = mjd
        self.ra, self.dec = ra,dec
        self.lst          = self.mjdToLST(self.mjd, self.lon)
        self.ha,self.zd   = self.equatorialToObservational (self.ra, self.dec, self.lst, self.lat) 
        self.airmass      = self.airmassModel(self.zd) 
        sunIsUp           = self.sunCalculations()
        moonSep           = self.moonCalculations()
        # neither of these two returns are needed here.

    def sunCalculations (self ) :
        self.sunData      = self.getSolarPosition (self.mjd, \
                self.lon, self.lat, self.lst)
        self.sunZD        = self.sunData[2]
        sunIsUp           = self.sunBrightnessModel (self.sunZD )
        return sunIsUp
    def moonCalculations (self ) :
        self.moonData     = self.getLunarPosition (self.mjd, \
                self.lon, self.lat, self.lst)
        self.moonZD       = self.moonData[2]
        self.moonSep      = self.getLunarSeparation(\
                self.ra, self.dec, self.moonData)
        return self.moonSep

    def guiderModel(self, filter="g") :
        verbose = False
        mjd       = self.mjd
        zd        = self.zd
        sun_zd    = self.sunZD
        moon_zd   = self.moonZD
        moon_phase= self.moonPhase
        moon_sep  = self.moonSep
        airmass   = self.airmass
        if verbose: 
            print mjd, zd, sun_zd, moon_zd, moon_phase, moon_sep, airmass
        atmo    = self.atmosphereTransmission(zd, airmass, filter, moon_sep)
        seeing  = self.seeing(airmass, filter, seeingAtZenith=0.9)
        sky     = self.skyBrightness(zd, moon_zd, moon_sep, moon_phase, filter) 
        return atmo, seeing, sky

    def atmosphereTransmission(self, zd,airmass,filter,moon_sep,refAirmass=1.3) :
        if self.verbose: print "\t ... atmosphere"
        atransmission = atmoModel.transmission(airmass, filter, refAirmass)
        if self.verbose: print "\t ... earth"
        dtransmission = atmoModel.dirtTransmission(zd)
        if self.verbose: print "\t ... moon"
        mtransmission = atmoModel.lunarDirtTransmission(moon_sep)
        transmission = atransmission*dtransmission*mtransmission
        return transmission

    def seeing(self, airmass, wavelength=775., seeingAtZenith=1.0) :
        if self.verbose: print "\t ... seeing"
        seeing = seeingModel.seeingWithAirmassAndLambda(airmass, wavelength, seeingAtZenith)
        return seeing

    def skyBrightness(self, zd, moon_zd, moon_sep, moon_phase, filter) :
        if self.verbose: print "\t ... sky brightness"
        sky = skyModel.sky_brightness_at_time( filter, zd, moon_zd, moon_sep, moon_phase) 
        return sky

    def skyBrightnessFid(self, filter) :
        skyfiducial = skyModel.skyFiducial(filter)
        return skyfiducial

    #
    # First, let's turn RA Dec into local sideral time, the start of all observations
    #
    # equation of Equinoxes is an ~1 second effect
    # "east longitude", where positive numbers increase as one moves east
    #   recall that lat, long is in radians
    def mjdToLST (self, mjd, eastLongitude) :
        gmst        = slalib.sla_gmst(mjd)
        eqEquinoxes = slalib.sla_eqeqx(mjd)
        lst         = gmst + eqEquinoxes + eastLongitude
        return lst 

    #
    # Now we turn LST and dec into Hour Angle and zenith distance
    #
    # Technically this is apparent ha,dec; i.e. we ignore diurnal abberation
    #   recall that lat, long is in radians
    # The best way to view this output is as hexbin(ra,dec,zd) or ha
    def equatorialToObservational (self, ra, dec, lst, latitude) :
        if self.verbose: print "\t LST to HA,zenith distance"
        ha = lst - ra
        zd = self.zenithDistance(ha, dec, latitude)
        return ha,zd

    #
    # Calculating zenith distance
    #
    # sin(HC) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(LHA)
    def zenithDistance(self, ha, dec, latitude) :
        degToRad = 2.*np.pi/360.
        sinAltRad = np.sin(latitude)*np.sin(dec) + np.cos(latitude)*np.cos(dec)*np.cos(ha)
        altRad = np.arcsin(sinAltRad)
        zenithDist = 90*degToRad - altRad
        return zenithDist

    #
    # Calculating airmass
    #
    # from Young 1994 "Air mass and refraction", Applied Optics 33:1108-1110
    # max error = 0.0037 airmasses at zd=90 degrees. (!)
    def airmassModel(self, zd) :
        if self.verbose: print "\t zenith distance to airmass"
        coszd = np.cos(zd)
        coszdsq = coszd*coszd
        numerator = 1.002432*coszdsq + 0.148386*coszd + 0.0096467
        denominator = coszdsq*coszd + 0.149864*coszdsq + 0.0102963*coszd + 0.000303978
        airmass = numerator/denominator
        return airmass
        

    #
    # check lunar position
    #
    def getLunarPosition (self, mjd, eastLongitude, latitude, lst ) :
        ra, dec, diam = slalib.sla_rdplan(mjd, 3, eastLongitude, latitude)
        ha = lst - ra
        zd = self.zenithDistance(ha, dec, latitude)
        return ra,dec,zd

    #
    # returns moon phase in degrees where 0 is full, 180 is new
    #
    def getLunarPhase (self) :
        moon_ra, moon_dec = self.moonData[0], self.moonData[1]
        sun_ra, sun_dec   = self.sunData[0], self.sunData[1]
        # moon is full when elongation = 180, new when elongation = 0
        moon_elongation = self.gc_separation(sun_ra, sun_dec, moon_ra, moon_dec)
        # K&S want moon phase as angle in degrees, 
        #   where 0 = full, 90 equals half, and  180 = new
        phase = (180*(2*np.pi/360.) - moon_elongation)*360./2/np.pi
        return phase

    def getLunarSeparation(self, ra, dec, moonData) :
        moon_ra, moon_dec = self.moonData[0], self.moonData[1]
        moon_sep = self.gc_separation(ra, dec, moon_ra, moon_dec)
        return moon_sep

    #
    # check solar position
    #
    def getSolarPosition (self, mjd, eastLongitude, latitude, lst ) :
        ra, dec, diam = slalib.sla_rdplan(mjd, 0, eastLongitude, latitude)
        ha = lst - ra
        zd = self.zenithDistance(ha, dec, latitude)
        return ra,dec,zd
    #
    # check to see if sun near the horizen
    #
    def sunBrightnessModel (self, sunZD) :
        twilight = 100.*2*np.pi/360. 
        if sunZD <= twilight :
            bright = True
        else :
            bright = False
        return bright
    
    #
    # great circle separations
    #
    def gc_separation(self, ra1, dec1, ra2, dec2) :
        delDec = dec1-dec2
        delRa = ra1-ra2
        dhav = self.haversine(delDec)
        rhav = self.haversine(delRa)
        hav = dhav + np.cos(dec1)*np.cos(dec2)*rhav
        gc_distance = self.ahaversine(hav)
        return gc_distance

    def haversine(self, theta) :
        hav = np.sin(theta/2.)**2
        return hav
    def ahaversine(self, x) :
        ahav = 2*np.arcsin(np.sqrt(x))
        return ahav

