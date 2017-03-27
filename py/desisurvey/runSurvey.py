import numpy as np
from pyslalib import slalib
import atmoModel
import seeingModel
import desMoonSkyModel
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

        # MJD = JD - 2400000.5
        self.mjd          = jdBegin -  2400000.5
        self.date         = slalib.sla_djcl(self.mjd)
        self.ra, self.dec = 0.0, 0.0

        self.degToRad     = 0.0174532925199

        # weather file 
        self.weatherfile  = weatherfile

        # observatory parameters
        lon_kpno  =  -111.600
        lat_kpno =     31.963
        ele_kpno  =  2120.
        lon_ctio  =   -70.8125
        lat_ctio  =   -30.16527778
        ele_ctio  =  2215.
        self.lon = lon_kpno
        self.lat = lat_kpno
        self.ele = ele_kpno


    def runDay (self) :
        # obs plan is to be created by afternoon work
        self.obsplan      = "../../examples/toyplan.fits"
        weather_fd        = self.weatherfileCreate (file) 

        filter = "i"
        afterSunset = False
        startTime = self.mjd
        endTime  = np.trunc(startTime + 1.)+0.5  ;# end of this day
        readyToExpose = startTime

        for time in np.arange(startTime, endTime, 1./24./60.) :
            if time < readyToExpose : continue
            self.mjd     = time
            default_ra   = self.mjd_to_lst(time)
            default_dec  = 0.0
            sky_model    = desMoonSkyModel.GeneralMoonSkyModel(
                    time, default_ra, default_dec, filter,
                    self.lon, self.lat, self.ele)
            sunIsUp      = self.sunUp (sky_model) 
            if sunIsUp: continue

            if not afterSunset :
                print "do the afternoon work"
                afterSunset = True

            transparency, seeing, skylevel = self.guiderModel()
            next = nextfield.get_next_field(time, \
                    skylevel, seeing, transparency, self.obsplan)
            foundtile = next["foundtile"] 
            if foundtile :
                ra = next["telera"] 
                dec = next["teledec"] 
                exptime = next["exptime"]
                maxtime = next["maxtime"]
                print "fake exposure at ra,dec, exptime",
                print ra,dec,exptime, "at hour ",(time-startTime)*24.

                self.ra, self.dec = ra,dec
                self.mjd = time
                lst = self.mjd_to_lst(time)
                real_transparency, real_seeing, real_skylevel = \
                        self.guiderModel()
                self.weatherfileUpdate( weather_fd, lst,
                        skylevel, seeing, transparency,
                        real_skylevel, real_seeing, real_transparency) 
            else :
                ra   = self.mjd_to_lst(time)
                dec  = 0.0
                exptime = 0.
                self.ra, self.dec = ra,dec
                self.mjd = time
            readyToExpose = self.mjd + exptime/(3600.*24.)
        weather_fd.close()
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

    def sunUp (self, sky_model ) :
        lon = sky_model.longitude
        lat = sky_model.latitude
        mjd = sky_model.mjd
        sunZD = desMoonSkyModel.sunZd(lat, lon, mjd)
        sunIsUp = True
        if sunZD < 108.0 : sunIsUp = False

    def guiderModel(self, filter="g") :
        verbose = False
        mjd       = self.mjd
        ra,dec    = self.ra, self.dec
        sky_model = desMoonSkyModel.GeneralMoonSkyModel(mjd, ra, dec, filter,
                    self.lon, self.lat, self.ele)
        lon       = sky_model.longitude
        lat       = sky_model.latitude
        lst       = self.mjd_to_lst(mjd)
        ha        = lst - self.ra
        zd        = desMoonSkyModel.zd( ha, dec*np.pi/180., 
                        lat*np.pi/180.0)*180.0/np.pi
        airmass   = desMoonSkyModel.airmass(zd)
        moon_sep  = ((180.0/np.pi)*np.arccos(
            desMoonSkyModel.moon_cosrho(mjd, ra, dec, lat, lon)))

        atmo    = self.atmosphereTransmission(zd, airmass, filter, moon_sep)
        seeing  = self.seeing(airmass, filter, seeingAtZenith=0.9)
        sky     = sky_model.skymag
        return atmo, seeing, sky

    def atmosphereTransmission(self, 
            zd,airmass,filter,moon_sep,refAirmass=1.3) :
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
        seeing = seeingModel.seeingWithAirmassAndLambda(
                airmass, wavelength, seeingAtZenith)
        return seeing

    #
    # Turn RA Dec into local sideral time, the start of all observations
    #
    # equation of Equinoxes is an ~1 second effect
    # "east longitude", where positive numbers increase as one moves east
    def mjd_to_lst (self, mjd, eastLongitude=-111.600) :
        gmst        = slalib.sla_gmst(mjd)
        eqEquinoxes = slalib.sla_eqeqx(mjd)
        lst         = gmst + eqEquinoxes + eastLongitude*2*np.pi/360.
        lst         = lst*360./2/np.pi
        if lst >= 360: lst = lst-360.
        return lst 

    def weatherfileCreate (self, file) :
        fd = open(self.weatherfile,"w")
        fd.write("# mjd lst in_sky in_see in_trans ")
        fd.write("real_sky real_see real_trans\n")
        return fd
    def weatherfileUpdate(self, fd, lst, skylevel, seeing, transparency,
            real_skylevel, real_seeing, real_transparency) :
        fd.write("{:.2f} {:.2f}  {:.1f} {:.2f} {:.2f} ".format(
            self.mjd, lst, skylevel, seeing,transparency))
        fd.write("{:.1f} {:.2f} {:.2f} \n".format(
            real_skylevel, real_seeing, real_transparency))

