import numpy as np
import astropy.io.fits as pyfits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from pkg_resources import resource_filename
from operator import itemgetter
from desisurvey.utils import radec2altaz, mjd2lst, equ2gal_J2000, sort2arr, inLSTwindow
from desitarget.targetmask import obsconditions as obsbits
from desisurvey.exposurecalc import airMassCalculator
import copy

import warnings
warnings.simplefilter('error', RuntimeWarning)

class surveyPlan:
    """
    Main class for survey planning
    """
    
    def __init__(self, MJDstart, MJDend, surveycal, tilesubset=None):
        """Initialises survey by reading in the file desi_tiles.fits
        and populates the class members.

        Arguments:
            MJDstart: day of the (re-)start of the survey
            MJDend: day of the end of the survey
            surveycal: list of dictionnaries with times of sunset, twilight, etc

        Optional:
            tilesubset: array of integer tileids to use; ignore others
        """

        self.surveycal = surveycal
        # Read in DESI tile data
        # Columns are:
        #   'TILEID'; format = 'J'
        #   'RA'; format = 'D'
        #   'DEC'; format = 'D'
        #   'PASS'; format = 'I'
        #   'IN_DESI'; format = 'I'
        #   'EBV_MED'; format = 'E'
        #   'AIRMASS'; format = 'E'
        #   'STAR_DENSITY'; format = 'E'
        #   'EXPOSEFAC'; format = 'E'
        #   'PROGRAM'; format = '6A'
        #   'OBSCONDITIONS'; format = 'J'
        #hdulist0 = pyfits.open(resource_filename('desimodel', 'data/footprint/desi-tiles.fits'))
        hdulist0 = pyfits.open('/Users/mlandriau/DESI/desihub/desimodel/py/desimodel/data/footprint/desi-tiles.fits')
        tiledata0 = hdulist0[1].data
        # This works because original table has index = tileID.
        if tilesubset is not None:
            tiledata1 = tiledata0[tilesubset]
        else:
            tiledata1 = tiledata0
        # Only keep DESI tiles
        in_desi = np.where(tiledata1.field('IN_DESI')==1)
        tb_temp = tiledata1[in_desi]
        # Ignore EXTRA tiles (passes 8 and 9)
        no_extra = np.where(tb_temp.field('PROGRAM') != 'EXTRA')
        tiledata = tb_temp[no_extra]
        hdulist0.close()

        # Dummy initilisations
        priority = 0
        status = 0
        ha = 0.0
        lstmin = 0.0
        lstmax = 0.0
        explen = 0.0
        
        # Loop over elements in table
        self.numtiles = len(tiledata)
        self.tiles = np.recarray((self.numtiles,),
                                 names = ('TILEID', 'RA', 'DEC', 'PASS', 'EBV_MED', 'PROGRAM', 'OBSCONDITIONS', 'GAL_CAP', 'SUBLIST', 'PRIORITY', 'STATUS', 'HA', 'LSTMIN', 'LSTMAX', 'EXPLEN'),
                                 formats = ['i4', 'f8', 'f8', 'i4', 'f4', 'a6', 'i2', 'i4', 'i4', 'i4', 'i4', 'f8', 'f8', 'f8', 'f8'])
        for i in range(len(tiledata)):
            #l, b = equ2gal_J2000(tiledata[i].field('RA'), tiledata[i].field('DEC'))
            #cap = b / np.abs(b)
            if tiledata[i].field('RA') > 75.0 and tiledata[i].field('RA') < 300.0:
                cap = 1.0
            else:
                cap = -1.0
            if tiledata[i].field('PASS') == 0:
                if self.inFirstYearFullDepthField(tiledata[i].field('DEC'), cap, True):
                    sublist = 0
                else:
                    sublist = 8
            elif tiledata[i].field('PASS') == 1:
                if self.inFirstYearFullDepthField(tiledata[i].field('DEC'), cap, False):
                    sublist = 1
                else:
                    sublist = 9
            elif tiledata[i].field('PASS') == 2:
                if self.inFirstYearFullDepthField(tiledata[i].field('DEC'), cap, False):
                    sublist = 2
                else:
                    sublist = 10
            elif tiledata[i].field('PASS') == 3:
                if self.inFirstYearFullDepthField(tiledata[i].field('DEC'), cap, False):
                    sublist = 3
                else:
                    sublist = 11
            elif tiledata[i].field('PASS') == 4:
                if self.inFirstYearFullDepthField(tiledata[i].field('DEC'), cap, True):
                    sublist = 4
                else:
                    sublist = 12
            elif tiledata[i].field('PASS') == 5:
                if self.inFirstYearFullDepthField(tiledata[i].field('DEC'), cap, True):
                    sublist = 5
                else:
                    sublist = 13
            elif tiledata[i].field('PASS') == 6:
                if self.inFirstYearFullDepthField(tiledata[i].field('DEC'), cap, False):
                    sublist = 6
                else:
                    sublist = 14
            elif tiledata[i].field('PASS') == 7:
                if self.inFirstYearFullDepthField(tiledata[i].field('DEC'), cap, False):
                    sublist = 7
                else:
                    sublist = 15
            self.tiles[i] = (tiledata[i].field('TILEID'),
                              tiledata[i].field('RA'),
                              tiledata[i].field('DEC'),
                              tiledata[i].field('PASS'),
                              tiledata[i].field('EBV_MED'),
                              tiledata[i].field('PROGRAM'),
                              tiledata[i].field('OBSCONDITIONS'),
                              cap,
                              sublist,
                              priority,
                              status,
                              ha,
                              lstmin,
                              lstmax,
                              explen)

        self.LSTres = 2.5 # bin width in degrees, starting with 10 minutes for now
        self.nLST = int(np.floor(360.0/self.LSTres))
        self.LSTbins = np.zeros(self.nLST)
        for i in range(self.nLST):
            self.LSTbins[i] = (float(i) + 0.5) * self.LSTres

        self.assignHA(MJDstart, MJDend)
        self.tiles.sort(axis=0, order=('SUBLIST', 'DEC'))
                
    def assignHA(self, MJDstart, MJDend, compute=False):
        """Assigns optimal hour angles for the DESI tiles;
        can be re-run at any point during the survey to
        reoptimise the schedule.
        """

        if compute:
            f = open("ha_check.dat", 'w')
            obs_dark = self.plan_ha(MJDstart, MJDend, False)
            obs_bright = self.plan_ha(MJDstart, MJDend, True)
            for tile in self.tiles:
                if (tile['OBSCONDITIONS'] & obsbits.mask('GRAY')) != 0 or (tile['OBSCONDITIONS'] & obsbits.mask('DARK')) != 0:
                    j=0
                    while j < len(obs_dark):
                        if tile['TILEID'] == obs_dark['tileid'][j]:
                            tile['HA'] = obs_dark['ha'][j]
                            tile['LSTMIN'] = obs_dark['beginobs'][j]
                            tile['LSTMAX'] = obs_dark['endobs'][j]
                            tile['EXPLEN'] = obs_dark['obstime'][j]
                            break
                        else:
                            j += 1
                else:
                    i=0
                    while i < len(obs_bright):
                        if tile['TILEID'] == obs_bright['tileid'][i]:
                            tile['HA'] = obs_bright['ha'][i]
                            tile['LSTMIN'] = obs_bright['beginobs'][i]
                            tile['LSTMAX'] = obs_bright['endobs'][i]
                            tile['EXPLEN'] = obs_bright['obstime'][i]
                            break
                        else:
                            i += 1
                line = str(tile['TILEID']) +" "+ str(tile['HA']) +" "+ str(tile['LSTMIN']) +" "+ str(tile['LSTMAX']) +" "+ str(tile['EXPLEN']) +" " + str(tile['PROGRAM']) +"\n"
                f.write(line)
            f.close()
        else:
            # Reads in the pre-computed HA a LSTbegin and LSTend
            hdulist0 = pyfits.open(resource_filename('surveysim', 'data/tile-info.fits'))
            tiledata0 = hdulist0[1].data
            j = 0
            for tile in self.tiles:
                while j < len(tiledata0):
                    if tile['TILEID'] == tiledata0[j][0]:
                        tile['HA'] = tiledata0[j][8]
                        tile['LSTMIN'] = tiledata0[j][16]
                        tile['LSTMAX'] = tiledata0[j][17]
                        tile['EXPLEN'] = tiledata0[j][14]
                        break
                    else:
                        j += 1

    def inFirstYearFullDepthField(self, dec, bgal, first_pass):
        """Checks whether the given field centre is within the
        area to be observed at full depth during the first year.
        The current field characteristics are:
        15 < dec < 25 for all the tiles in the NGC, buffered by
        3degs for the first pass in each program.

        Args:
            ra: right ascension of the field centre in degrees
            dec: declination of the field centre in degrees
            first_pass: bool set to True if first pass of main survey or BGS

        Returns:
            True if it is within the area, False otherwise.
        """

        if (first_pass):
            decmin = 12.0
            decmax = 28.0
        else:
            decmin = 15.0
            decmax = 25.0
        
        answer = False
        if (dec >= decmin and dec <= decmax and bgal > 0.0):
            answer = True

        return answer

    def afternoonPlan(self, day_stats, tiles_observed):
        """Main decision making method

        Args:
            day_stats: dictionnary containing the following keys:
                       'MJDsunset', 'MJDsunrise', 'MJDetwi', 'MJDmtwi', 'MJDe13twi',
                       'MJDm13twi', 'MJDmoonrise', 'MJDmoonset', 'MoonFrac', 'dirName'
            tiles_observed: table with follwing columns: tileID, status

        Returns:
            string containg the filename for today's plan; it has the format
            obsplanYYYYMMDD.fits
        """

        # Update status
        finalTileList = []
        nto = len(tiles_observed)
        for j in range(self.numtiles):
            for i in range(nto):
                if self.tiles[j]['TILEID'] == tiles_observed['TILEID'][i]:
                    self.tiles[j]['STATUS'] = tiles_observed['STATUS'][i]
                    break
            if self.tiles[j]['STATUS']==0:
                finalTileList.append(self.tiles[j])

        # Assign tiles to LST bins
        planList0 = []
        lst15evening = mjd2lst(day_stats['MJDetwi'])
        lst15morning = mjd2lst(day_stats['MJDmtwi'])
        lst13evening = mjd2lst(day_stats['MJDe13twi'])
        lst13morning = mjd2lst(day_stats['MJDe13twi'])
        LSTmoonrise = mjd2lst(day_stats['MJDmoonrise'])
        LSTmoonset = mjd2lst(day_stats['MJDmoonset'])
        LSTbrightstart = mjd2lst(day_stats['MJD_bright_start'])
        LSTbrightend = mjd2lst(day_stats['MJD_bright_end'])

        # Loop over LST bins
        for i in range(self.nLST):
            # DARK time
            if ( inLSTwindow(self.LSTbins[i], lst15evening, lst15morning) and
                 not inLSTwindow(self.LSTbins[i], LSTmoonrise, LSTmoonset) ):
                nfields = 0
                for tile in finalTileList:
                    tileLST = tile['RA'] + tile['HA']
                    if tileLST < 0.0:
                        tileLST += 360.0
                    if tileLST > 360.0:
                        tileLST -= 360.0
                    if ( tile['STATUS']<2 and
                         tileLST >= self.LSTbins[i] - 0.5*self.LSTres and
                         tileLST <= self.LSTbins[i] + 0.5*self.LSTres and
                         (tile['OBSCONDITIONS'] & obsbits.mask('DARK')) != 0 ):
                        tile['PRIORITY'] = nfields + 3
                        #tile['LSTMIN'] = self.LSTbins[i] - 0.5*self.LSTres
                        #tile['LSTMAX'] = self.LSTbins[i] + 0.5*self.LSTres
                        planList0.append(tile)
                        nfields += 1
                    if nfields == 5:
                        break
                    else:
                        continue
                if nfields < 5: # If fewer than 5 dark tiles fall within this window, pad with grey tiles 
                    for tile in finalTileList:
                        tileLST = tile['RA'] + tile['HA']
                        if tileLST < 0.0:
                            tileLST += 360.0
                        if tileLST > 360.0:
                            tileLST -= 360.0
                        if ( tile['STATUS']<2 and
                            tileLST >= self.LSTbins[i] - 0.5*self.LSTres and
                            tileLST <= self.LSTbins[i] + 0.5*self.LSTres and
                            (tile['OBSCONDITIONS'] & obsbits.mask('GRAY')) != 0 ):
                            tile['PRIORITY'] = nfields + 3
                            #tile['LSTMIN'] = self.LSTbins[i] - 0.5*self.LSTres
                            #tile['LSTMAX'] = self.LSTbins[i] + 0.5*self.LSTres
                            planList0.append(tile)
                            nfields += 1
                        if nfields == 5:
                            break
                        else:
                            continue
                if nfields < 5: # If fewer than 5 dark or grey tiles fall within this window, pad with bright tiles 
                    for tile in finalTileList:
                        tileLST = tile['RA'] + tile['HA']
                        if tileLST < 0.0:
                            tileLST += 360.0
                        if tileLST > 360.0:
                            tileLST -= 360.0
                        if ( tile['STATUS']<2 and
                            tileLST >= self.LSTbins[i] - 0.5*self.LSTres and
                            tileLST <= self.LSTbins[i] + 0.5*self.LSTres and
                            (tile['OBSCONDITIONS'] & obsbits.mask('BRIGHT')) != 0 ):
                            tile['PRIORITY'] = nfields + 3
                            #tile['LSTMIN'] = self.LSTbins[i] - 0.5*self.LSTres
                            #tile['LSTMAX'] = self.LSTbins[i] + 0.5*self.LSTres
                            planList0.append(tile)
                            nfields += 1
                        if nfields == 5:
                            break
                        else:
                            continue
            # GREY time
            if ( inLSTwindow(self.LSTbins[i], lst15evening, lst15morning) and
                 inLSTwindow(self.LSTbins[i], LSTmoonrise, LSTmoonset) and
                 not inLSTwindow(self.LSTbins[i], LSTbrightstart, LSTbrightend) ):
                nfields = 0
                for tile in finalTileList:
                    tileLST = tile['RA'] + tile['HA']
                    if tileLST < 0.0:
                        tileLST += 360.0
                    if tileLST > 360.0:
                        tileLST -= 360.0
                    if ( tile['STATUS']<2 and
                         tileLST >= self.LSTbins[i] - 0.5*self.LSTres and
                         tileLST <= self.LSTbins[i] + 0.5*self.LSTres and
                         (tile['OBSCONDITIONS'] & obsbits.mask('GRAY')) != 0 ):
                        tile['PRIORITY'] = nfields + 3
                        #tile['LSTMIN'] = self.LSTbins[i] - 0.5*self.LSTres
                        #tile['LSTMAX'] = self.LSTbins[i] + 0.5*self.LSTres
                        planList0.append(tile)
                        nfields += 1
                    if nfields == 5:
                        break
                    else:
                        continue
                if nfields < 5: # If fewer than 5 grey tiles fall within this window, pad with bright tiles 
                    for tile in finalTileList:
                        tileLST = tile['RA'] + tile['HA']
                        if tileLST < 0.0:
                            tileLST += 360.0
                        if tileLST > 360.0:
                            tileLST -= 360.0
                        if ( tile['STATUS']<2 and
                            tileLST >= self.LSTbins[i] - 0.5*self.LSTres and
                            tileLST <= self.LSTbins[i] + 0.5*self.LSTres and
                            (tile['OBSCONDITIONS'] & obsbits.mask('BRIGHT')) != 0 ):
                            tile['PRIORITY'] = nfields + 3
                            #tile['LSTMIN'] = self.LSTbins[i] - 0.5*self.LSTres
                            #tile['LSTMAX'] = self.LSTbins[i] + 0.5*self.LSTres
                            planList0.append(tile)
                            nfields += 1
                        if nfields == 5:
                            break
                        else:
                            continue
            # BRIGHT time
            if ( (inLSTwindow(self.LSTbins[i], lst13evening, lst13morning) and
                  not inLSTwindow(self.LSTbins[i], lst15evening, lst15morning)) or
                  inLSTwindow(self.LSTbins[i], LSTbrightstart, LSTbrightend) ):
                nfields = 0
                for tile in finalTileList:
                    tileLST = tile['RA'] + tile['HA']
                    if tileLST < 0.0:
                        tileLST += 360.0
                    if tileLST > 360.0:
                        tileLST -= 360.0
                    if ( tile['STATUS']<2 and
                         tileLST >= self.LSTbins[i] - 0.5*self.LSTres and
                         tileLST <= self.LSTbins[i] + 0.5*self.LSTres and
                         (tile['OBSCONDITIONS'] & obsbits.mask('BRIGHT')) != 0 ):
                        tile['PRIORITY'] = nfields + 3
                        #tile['LSTMIN'] = self.LSTbins[i] - 0.5*self.LSTres
                        #tile['LSTMAX'] = self.LSTbins[i] + 0.5*self.LSTres
                        planList0.append(tile)
                        nfields += 1
                    if nfields == 5:
                        break
                    else:
                        continue

        cols = np.recarray((len(planList0),),
                           names = ('TILEID', 'RA', 'DEC', 'PASS', 'EBV_MED', 'PROGRAM', 'OBSCONDITIONS', 'GAL_CAP', 'SUBLIST', 'PRIORITY', 'STATUS', 'HA', 'LSTMIN', 'LSTMAX', 'EXPLEN'),
                           formats = ['i4', 'f8', 'f8', 'i4', 'f4', 'a6', 'i2', 'i4', 'i4', 'i4', 'i4', 'f8', 'f8', 'f8', 'f8'])
        cols[:] = planList0[:]
        tbhdu = pyfits.BinTableHDU.from_columns(cols)

        prihdr = pyfits.Header()
        prihdr['MOONFRAC'] = day_stats['MoonFrac']
        prihdu = pyfits.PrimaryHDU(header=prihdr)
        filename = 'obsplan' + day_stats['dirName'] + '.fits'
        thdulist = pyfits.HDUList([prihdu, tbhdu])
        thdulist.writeto(filename, clobber=True)

        tilesTODO = len(planList0)

        return filename


####################################################################
# Below is a translation of Kyle's IDL code to compute hour angles #
####################################################################
    def plan_ha(self, survey_begin, survey_end, BGS=False):
        """Main driver of hour angle computations
        
            Args:
                survey_begin: MJD of (re-)start of survey
                survey_end: MJD of the expected end

            Optional:
                BGS: bool, true if bright sample
        """

        if BGS:
            exptime = 600.0 / 3600.0
        else:
            exptime = 1000.0 / 3600.0
        # First define general survey characteristics
        r_threshold = 1.54 # this is an initial guess for required SN2/pixel over r-band 
        b_threshold = 0.7 # same for g-band, scaled relative to r-band throughout analysis, the ratio of r-b cannot change
        times = np.copy(self.LSTbins)*24.0/360.0 # LST bins in hours
        scheduled_times = np.zeros(self.nLST) # available hours at each LST bin over the full survey, after accounting for weather loss
        observed_times = np.zeros(self.nLST) # hours spent observing at each LST bin, iteratively filled until optimal HA distribution is achieved
        #remaining_times = np.zeros(self.nLST) # simply the difference between scheduled times and observed times
        sgcfraction_times = np.zeros(self.nLST) # the fraction of total time in each bin of LST, SGC
        ngcfraction_times = np.zeros(self.nLST) # the fraction of total time in each bin of LST, NGC
        weather=0.74*0.77    # SRD assumption: 74% open dome and 77% good seeing
        excess=1.01

        # There is some repeated code from the afternoon plan
        # which should be factored out.
        for night in self.surveycal:
            lst15evening = mjd2lst(night['MJDetwi'])
            lst15morning = mjd2lst(night['MJDmtwi'])
            lst13evening = mjd2lst(night['MJDe13twi'])
            lst13morning = mjd2lst(night['MJDm13twi'])
            LSTmoonrise = mjd2lst(night['MJDmoonrise'])
            #if LSTmoonrise < 180.0:
            #    LSTmoonrise += 360.0
            LSTmoonset = mjd2lst(night['MJDmoonset'])
            #if LSTmoonset < 180.0:
            #    LSTmoonset += 360.0
            if night['MJD_bright_start'] > 0.0:
                LSTbrightstart = mjd2lst(night['MJD_bright_start'])
                #if LSTbrightstart < 180.0:
                #    LSTbrightstart += 360.0
            else:
                LSTbrightstart = 1.0e99
            if night['MJD_bright_end'] > 0.0:
                LSTbrightend = mjd2lst(night['MJD_bright_end'])
                #if LSTbrightend < 180.0:
                #    LSTbrightend += 360.0
            else:
                LSTbrightend = -1.0e99
            for i in range(self.nLST):
                if BGS:
                    if ( (lst13evening < self.LSTbins[i] and self.LSTbins[i] <= lst15evening) or
                         (lst15morning <= self.LSTbins[i] and self.LSTbins[i] < lst13morning) or
                         ( (lst15evening < self.LSTbins[i] and self.LSTbins[i] < lst15morning) and
                         (LSTbrightstart < self.LSTbins[i] and self.LSTbins[i] < LSTbrightend) ) ):
                        scheduled_times += 1.0
                else:
                    if ( (lst15evening < self.LSTbins[i] and self.LSTbins[i] < lst15morning) and
                         (LSTmoonrise > self.LSTbins[i] or self.LSTbins[i] > LSTmoonset) ):
                        scheduled_times[i] += 1.0
                    if ( (lst15evening < self.LSTbins[i] and self.LSTbins[i] < lst15morning) and
                         (LSTmoonrise < self.LSTbins[i] and self.LSTbins[i] < LSTmoonset) and
                         (LSTbrightstart > self.LSTbins[i] or self.LSTbins[i] > LSTbrightend) ):
                        scheduled_times[i] += 1.0
        scheduled_times *= weather*self.LSTres/15.0 # in hours
        remaining_times = np.copy(scheduled_times)
        
        surveystruct = {'exptime' : exptime/3600.0,  # nominal exposure time
                        'overhead1' : 2.0/60.0,      # amount of time for cals and field acquisition
                        'overhead2' : 1.0/60.0,      # amount of time for readout
                        'survey_begin' : survey_begin,  
                        'survey_end' : survey_end,
                        'res' : self.LSTres*24.0/360.0,
                        'avg_rsn' : 0.75,            # SN2/pixel in r-band during nominal exposure time under average conditions, needs to be empirically determined
                        'avg_bsn' : 0.36,            # same, for g-band
                        'alpha_red' : 1.25,          # power law for airmass dependence, r-band
                        'alpha_blue' : 1.25,         # same, for g-band
                        'r_threshold' : r_threshold,
                        'b_threshold' : b_threshold,
                        'weather' : weather,         # estimate of time available, after weather loss
                        'times' : times,             # time is in hours
                        'scheduled_times' : scheduled_times,
                        'observed_times' : observed_times,
                        'remaining_times' : remaining_times,
                        'ngcfraction_times' : ngcfraction_times,
                        'sgcfraction_times' : sgcfraction_times,
                        'ngc_begin' : 5.5,           # estimate of bounds for NGC
                        'ngc_end' : 20.0,            # estimate of bounds for NGC
                        'platearea' : 1.4,           # area in sq degrees per unique tile
                        'surveyarea' : 14000.0,      # required survey area
                        'survey_duration' : 0.0,     # Total time for survey, after weather
                        'fiducial_exptime' : 0.0,    # open shutter time at zero extinction and 1 airmass
                        'dark' : 0.55}               # definition of dark time, in terms of moon phase

        obs = self.compute_extinction(BGS)

        # FIND MINIMUM AMOUNT OF TIME REQUIRED TO COMPLETE PLATES
        num_obs = len(obs)
        ha = np.zeros(num_obs, dtype='f8')
        for i in range(num_obs):
            self.filltimes(obs, surveystruct, ha[i], i)

        # NOW RUN ALL TILES THROUGH THE TIME ALLOCATED AT EACH LST BIN AND FIND CHEAPEST FOOTPRINT TO COVER IN TIME ALLOCATED
        optimize = 1

        # ADJUST THRESHOLDS ONCE TO MATCH AVAILABLE LST DISTRIBUTION
        a = np.ravel(np.where(obs['obs_bit'] > 1))
        rel_area = len(a)*surveystruct['platearea']/surveystruct['surveyarea']
        oh_avg = 0.0
        if rel_area < 1.0 and rel_area > 0.0:
            obs_avg = np.average(obs['obstime'][np.ravel(a)])
            oh_avg = np.average(obs['overhead'][np.ravel(a)])
            t_scheduled = obs_avg - oh_avg
            t_required = obs_avg*rel_area - oh_avg
            surveystruct['r_threshold'] *= t_required/t_scheduled
            surveystruct['b_threshold'] *= t_required/t_scheduled
        if np.sum(surveystruct['remaining_times']) > 0.0:
            t_scheduled = np.sum(surveystruct['observed_times'])/num_obs - oh_avg
            t_required = np.sum(surveystruct['scheduled_times'])/num_obs - oh_avg
            surveystruct['r_threshold'] *= t_required/t_scheduled*excess
            surveystruct['b_threshold'] *= t_required/t_scheduled*excess
        self.retile(obs, surveystruct, optimize)

        #obs.sort(axis=0, order=('tileid'))
        return obs

    def compute_extinction (self, BGS=False):

        if BGS:
            a = np.where(self.tiles['PASS'] > 4)
        else:
            a = np.where(self.tiles['PASS'] <= 4)
        subtiles = self.tiles[a]
        ntiles = len(subtiles)
        tileid = subtiles['TILEID']
        used = np.zeros(ntiles, dtype='i4')
        ra = subtiles['RA']
        dec = subtiles['DEC']
        ebv = subtiles['EBV_MED']
        num = int( np.floor(len(subtiles)/5) )
        indices = 5*np.arange(num, dtype='i4')
        qsoflag = np.zeros(ntiles, dtype='i4')
        qsoflag[indices] = 1

        layer = subtiles['PASS']
        program = subtiles['PROGRAM']
        obsconditions = subtiles['OBSCONDITIONS']

        locationid = np.copy(tileid)

        i_increase = np.zeros(ntiles, dtype='f8')
        g_increase = np.zeros(ntiles, dtype='f8')
        glong = np.zeros(ntiles, dtype='f8')
        glat = np.zeros(ntiles, dtype='f8')
        select = np.zeros(ntiles, dtype='f8')
        overhead = np.zeros(ntiles, dtype='f8')

        # From http://arxiv.org/pdf/1012.4804v2.pdf Table 6
        # R_u = 4.239
        R_g = 3.303
        # R_r = 2.285
        R_i = 1.698
        # R_z = 1.263

        for i in range(ntiles):
            glong[i], glat[i] = equ2gal_J2000(ra[i], dec[i])
        i_increase = np.power(10.0, 0.8*R_i*ebv)
        g_increase = np.power(10.0, 0.8*R_g*ebv)

        ra *= 24.0/360.0    # units of hours
        ha = np.zeros(ntiles, dtype='f8')
        airmass = np.ones(ntiles, dtype='f8')
        obs_bit = np.zeros(ntiles, dtype='i4')
        obstime = np.zeros(ntiles, dtype='f8')
        red_sn = np.zeros(ntiles, dtype='f8')
        blue_sn = np.zeros(ntiles, dtype='f8')
        beginobs = np.zeros(ntiles, dtype='f8')
        endobs = np.zeros(ntiles, dtype='f8')

        obs = {'tileid' : tileid,
                'locationid' : locationid,
                'qsoflag' : qsoflag,
                'ra' : ra,
                'dec' : dec,
                'glong' : glong,
                'glat' : glat,
                'ha' : ha,
                'airmass' : airmass,
                'ebv' : ebv,
                'i_increase' : i_increase,
                'g_increase' : g_increase,
                'obs_bit' : obs_bit,
                'obstime' : obstime,
                'overhead' : overhead,
                'beginobs' : beginobs,
                'endobs' : endobs,
                'red_sn' : red_sn,
                'blue_sn' : blue_sn,
                'pass' : layer,
                'PROGRAM' : program,
                'OBSCONDITIONS' : obsconditions}

        return obs

    def retile (self, obs, surveystruct, optimize):

        num_obs = len(obs['tileid'])
        times = surveystruct['times']
        num_times = len(times)
        rank_times = np.zeros(num_times, dtype='f8')
        rank_plates = np.zeros(num_obs, dtype='f8')
        dec = obs['dec']
        ra = obs['ra']

        ha_tmp = np.empty(num_obs, dtype='f8')
        airmass_tmp = np.empty(num_obs, dtype='f8')
        temp_rank = np.empty(num_obs, dtype='f8')
        for i in range(num_times):
            ha_tmp = surveystruct['times'][i] - ra
            ha_tmp[np.ravel(np.where(ha_tmp >= 12.0))] -= 24.0
            ha_tmp[np.ravel(np.where(ha_tmp <= -12.0))] += 24.0
            airmass_tmp = airMassCalculator(ra, dec, ha_tmp+ra)
            temp_rank = np.power(airmass_tmp, surveystruct['alpha_red']*obs['i_increase'])
            temp_rank.sort()
            rank_times[i] = np.average(temp_rank[0:50])

        # What is the purpose of this???
        r1 = np.max(rank_times[0:num_times//2-1])
        r2 = np.max(rank_times[num_times//2:num_times-1])
        a1 = np.where(rank_times == r1)
        a2 = np.where(rank_times == r2)
        a = np.concatenate([np.ravel(a1), np.ravel(a2)])
        a.sort()
        index1 = a[0] + 2
        index2 = a[1] - 7 # Adjust boundary between SGC + NGC: should be done iteratively.
        #print(index1, index2, len(surveystruct['scheduled_times']))
        print(surveystruct['scheduled_times'])
        ends =  0.5*surveystruct['scheduled_times'][index1] + 0.5*surveystruct['scheduled_times'][index2]
        ngctime = np.sum(surveystruct['scheduled_times'][index1:index2-1]) + ends
        sgctime = np.sum(surveystruct['scheduled_times'][0:index1-1]) + np.sum(surveystruct['scheduled_times'][index2:num_times-1]) + ends
        #print(ngctime, sgctime)
        surveystruct['sgcfraction_times'][0:index1] = surveystruct['scheduled_times'][0:index1]/sgctime
        surveystruct['sgcfraction_times'][index2:num_times-1] = surveystruct['scheduled_times'][index2:num_times-1]/sgctime
        surveystruct['ngcfraction_times'][index1:index2] = surveystruct['scheduled_times'][index1:index2]/ngctime
        surveystruct['sgcfraction_times'][index1] = 0.5*surveystruct['scheduled_times'][index1]/sgctime
        surveystruct['ngcfraction_times'][index1] = 0.5*surveystruct['scheduled_times'][index1]/ngctime
        surveystruct['sgcfraction_times'][index2] = 0.5*surveystruct['scheduled_times'][index2]/sgctime
        surveystruct['ngcfraction_times'][index2] = 0.5*surveystruct['scheduled_times'][index2]/ngctime
        
        obs['obstime'][:] = 0.0
        sgcplates = np.ravel(np.where( (obs['ra'] < surveystruct['ngc_begin']) |
                                       (obs['ra'] > surveystruct['ngc_end']) ))
        ngcplates = np.ravel(np.where( (obs['ra'] > surveystruct['ngc_begin']) &
                                       (obs['ra'] < surveystruct['ngc_end']) ))
        sgc_obstime = 0.0
        ngc_obstime = 0.0

        a = np.where(obs['obs_bit'] < 2) # Changed from Kyle's: it was <=2
        obs['obs_bit'][np.ravel(a)] = 0
        obs['ha'][np.ravel(a)] = 0.0

        # Start by filling the hardest regions with tiles, NGC then SGC
        dec = obs['dec'][ngcplates]
        ra = obs['ra'][ngcplates]
        orig_ha = obs['ha'][ngcplates]
        transit = np.zeros(len(ngcplates), dtype='i2')

        #print(index1, index2)
        nindices = index2-index1+1
        for i in range(nindices):
            ihalf = i//2
            if 2*ihalf == i:
                index = index1 + ihalf
                ha = times[index] - obs['ra'][ngcplates] - 0.5*surveystruct['res']
            else:
                index = index2 - ihalf
                ha = times[index] - obs['ra'][ngcplates] + 0.5*surveystruct['res']
            ha[np.ravel(np.where(ha >= 12.0))] -= 24.0
            ha[np.ravel(np.where(ha <= -12.0))] += 24.0
            num_reqplates = int(np.ceil( (surveystruct['ngcfraction_times'][index]*ngctime-surveystruct['observed_times'][index])/surveystruct['res'] ))
            #print(num_reqplates)
            tile = obs['tileid'][ngcplates]
            obs_bit = obs['obs_bit'][ngcplates]
            transit[np.ravel (np.where( (obs['ra'][ngcplates]+orig_ha > surveystruct['times'][i]-0.5*surveystruct['res']) &
                                        (obs['ra'][ngcplates]+orig_ha <= surveystruct['times'][i]+0.5*surveystruct['res']) ) )] = 1
            airmass = airMassCalculator(ra, dec, ra+ha)
            orig_airmass = airMassCalculator(ra, dec, ra+orig_ha)
            rank_plates_tmp = np.power(airmass, surveystruct['alpha_red']*obs['i_increase'][ngcplates])
            if optimize:
                rank_plates_tmp -= np.power(orig_airmass, surveystruct['alpha_red']*obs['i_increase'][ngcplates])
            angc = np.ravel( np.where( (obs_bit == 1) & (np.abs(ha) < 1.0) ) )
            asize = len(angc)
            if asize > 0 and optimize == 0:
                rank_plates_tmp[angc] = 1000.0
            if len(np.ravel(np.where(obs_bit < 2))) == 0:
                break
            if asize < num_reqplates:
                num_reqplates = asize
            rank_plates = rank_plates_tmp[angc]
            tile0 = sort2arr(tile[angc],rank_plates)
            ha0 = sort2arr(ha[angc], rank_plates)
            for j in range(num_reqplates):
                j2 = np.ravel(np.where(obs['tileid'] == tile[j]))[0]
                d = obs['dec'][j2]
                ra = obs['ra'][j2]
                h = ha0[j]
                airmass = airMassCalculator(ra, d, ra+h)
                red = surveystruct['avg_rsn']/np.power(airmass, surveystruct['alpha_red']/obs['i_increase'][j2])
                rtime = surveystruct['overhead1'] + surveystruct['exptime']*surveystruct['r_threshold']/red
                blue = surveystruct['avg_bsn']/np.power(airmass, surveystruct['alpha_blue'])/obs['g_increase'][j2]
                btime = surveystruct['overhead1'] + surveystruct['exptime']*surveystruct['b_threshold']/blue
                time = np.max([rtime, btime])
                ihalf = i//2
                if 2*ihalf == i:
                    h += 0.5*time
                else:
                    h -= 0.5*time
                obs['obs_bit'][j2] = 2
                self.filltimes(obs, surveystruct, h, j2)
                obs['ha'][j2] = h

        nindices = index1+num_times-index2+1

        dec = obs['dec'][sgcplates]
        ra = obs['ra'][sgcplates]
        orig_ha = obs['ha'][sgcplates]
        tile = obs['tileid'][sgcplates]
        obs_bit = obs['obs_bit'][sgcplates]

        for i in range(nindices):
            ihalf = i//2
            if 2*ihalf != i:
                index = index2 + ihalf
                if index < 0:
                    index += num_times
                if index >= num_times:
                    index -= num_times
                ha = times[index] - obs['ra'][sgcplates] - 0.5*surveystruct['res']
            else:
                index = index1 - ihalf
                ha = times[index] - obs['ra'][sgcplates] + 0.5*surveystruct['res']
            ha[np.ravel(np.where(ha >= 12.0))] -= 24.0
            ha[np.ravel(np.where(ha <= -12.0))] += 24.0
            num_reqplates = int(np.ceil((surveystruct['scheduled_times'][index] - surveystruct['observed_times'][index])/surveystruct['res']))
            airmass = airMassCalculator(ra, dec, ha+ra)
            orig_airmass = airMassCalculator(ra, dec, orig_ha+ra)
            rank_plates = np.power(airmass, surveystruct['alpha_red']*obs['i_increase'][sgcplates])
            if optimize:
                rank_plates -= np.power(orig_airmass, surveystruct['alpha_red']*obs['i_increase'][sgcplates])
            asgc = np.ravel(np.where( (obs_bit == 1) & (np.abs(ha) < 1.0) ))
            asize = len(asgc)
            if  asize > 0 and optimize == 0:
                rank_plates[asgc] = 1000.0
            if len(np.ravel(np.where(obs_bit < 2))) == 0:
                break
            num_reqplates = min([num_reqplates,asize])
            rank_plates = rank_plates[asgc]
            tile0 = sort2arr(tile[asgc],rank_plates)
            ha0 = sort2arr(ha[asgc], rank_plates)
            for j in range(num_reqplates):
                j2 = np.ravel(np.where(obs['tileid'] == tile[j]))[0]
                d = obs['dec'][j2]
                ra = obs['ra'][j2]
                h = ha0[j]
                airmass = airMassCalculator(ra, d, ra+h)
                red = surveystruct['avg_rsn']/np.power(airmass, surveystruct['alpha_red']/obs['i_increase'][j2])
                rtime = surveystruct['overhead1'] + surveystruct['exptime']*surveystruct['r_threshold']/red
                blue = surveystruct['avg_bsn']/np.power(airmass, surveystruct['alpha_blue']/obs['g_increase'][j2])
                btime = surveystruct['overhead1'] + surveystruct['exptime']*surveystruct['b_threshold']/blue
                time = np.max([rtime,btime])
                if 2*ihalf != i:
                     h += 0.5*time
                if 2*ihalf == i:
                    h -= 0.5*time
                obs['obs_bit'][j2] = 2
                self.filltimes(obs, surveystruct, h, j2)
                #obs['ha'][j2] = h

    def filltimes(self, obs, surveystruct, ha, index):

        res = surveystruct['res']
        times = surveystruct['times']

        overhead = surveystruct['overhead1']
        airmass = airMassCalculator(obs['ra'][index], obs['dec'][index], ha+obs['ra'][index])
        red = surveystruct['avg_rsn'] / np.power(airmass, surveystruct['alpha_red']) / obs['i_increase'][index]
        rtime = surveystruct['exptime']*surveystruct['r_threshold']/red
        blue = surveystruct['avg_bsn'] / np.power(airmass, surveystruct['alpha_blue'])/obs['g_increase'][index]
        btime = surveystruct['exptime']*surveystruct['b_threshold']/blue
        if btime > 1.0 or rtime > 1.0:
            overhead += surveystruct['overhead2']
        rtime += overhead
        btime += overhead
        obs['overhead'][index] = overhead
        time = np.max([rtime,btime])
        print(red, rtime, blue, btime, time)

        obs['red_sn'][index] = red*(time-overhead)/surveystruct['exptime']
        obs['blue_sn'][index] = blue*(time-overhead)/surveystruct['exptime']
        obs['obstime'][index] = time
        obs['beginobs'][index] = obs['ra'][index] + ha - 0.5*time
        obs['endobs'][index] = obs['ra'][index] + ha + 0.5*time
        obs['airmass'][index] = airmass
        obs['ha'][index] = ha

        if obs['beginobs'][index] < 0.0 and obs['endobs'][index] < 0.0:
            obs['beginobs'][index] += 24.0
            obs['endobs'][index] += 24.0

        if obs['beginobs'][index] > 24.0 and obs['endobs'][index] > 24.0:
            obs['beginobs'][index] -= 24.0
            obs['endobs'][index] -= 24.0

        #fill in times over LST range
        num = len(surveystruct['times'])
        for i in range(num):
            if obs['beginobs'][index] <= surveystruct['times'][i]-0.5*res and obs['endobs'][index] >= surveystruct['times'][i]+0.5*res:
                surveystruct['remaining_times'][i] -= res
                surveystruct['observed_times'][i] += res
            if obs['beginobs'][index] > surveystruct['times'][i]-0.5*res and obs['beginobs'][index] < surveystruct['times'][i]+0.5*res:
                surveystruct['remaining_times'][i] -= (surveystruct['times'][i]+0.5*res-obs['beginobs'][index])
                surveystruct['observed_times'][i] += (surveystruct['times'][i]+0.5*res-obs['beginobs'][index])
            if obs['endobs'][index] > surveystruct['times'][i]-0.5*res and obs['endobs'][index] < surveystruct['times'][i]+0.5*res:
                surveystruct['remaining_times'][i] -= (-(surveystruct['times'][i]-0.5*res)+obs['endobs'][index])
                surveystruct['observed_times'][i] += (-(surveystruct['times'][i]-0.5*res)+obs['endobs'][index])

        if obs['beginobs'][index] < 0.0:
            t = np.floor(-obs['beginobs'][index]/res)
            it = int(t)
            if t > 0.0:
                surveystruct['remaining_times'][num-it:num-1] -= res
                surveystruct['observed_times'][num-it:num-1] += res
            surveystruct['remaining_times'][num-it-1] -= (-obs['beginobs'][index]-t*res)
            surveystruct['observed_times'][num-it-1] += (-obs['beginobs'][index]-t*res)
            obs['beginobs'][index] += 24.0

        if obs['endobs'][index] > 24.0:
            obs['endobs'][index] -= 24.0
            t = np.floor(obs['endobs'][index]/res)
            it = int(t)
            if t > 0.0:
                surveystruct['remaining_times'][0:it-1] -= res
                surveystruct['observed_times'][0:it-1] += res
            surveystruct['remaining_times'][t] -= (obs['endobs'][index]-t*res)
            surveystruct['observed_times'][t] += (obs['endobs'][index]-t*res)



