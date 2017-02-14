import numpy as np
import astropy.io.fits as pyfits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from pkg_resources import resource_filename
from operator import itemgetter
from desisurvey.utils import radec2altaz, mjd2lst, equ2gal_J2000
from desitarget.targetmask import obsconditions as obsbits
import copy

class surveyPlan:
    """
    Main class for survey planning
    """
    
    def __init__(self, tilesubset=None):
        """Initialises survey by reading in the file desi_tiles.fits
        and populates the class members.

        Optional:
            tilesubset: array of integer tileids to use; ignore others
        """

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
            
        # Loop over elements in table
        self.tiles = []
        for i in range(len(tiledata)):
            b, l = equ2gal_J2000(tiledata[i].field('RA'), tiledata[i].field('DEC'))
            tile = {'tileID' : tiledata[i].field('TILEID'),
                    'RA' : tiledata[i].field('RA'),
                    'DEC' : tiledata[i].field('DEC'),
                    'PASS' : tiledata[i].field('PASS'),
                    'Ebmv' : tiledata[i].field('EBV_MED'),
                    'program' : tiledata[i].field('PROGRAM'),
                    'obsconds' : tiledata[i].field('OBSCONDITIONS'),
                    'cap' : b / np.abs(b),
                    'priority' : 0,
                    'status' : 0,
                    'HA' : 0.0,
                    'lst_min' : 0.0,
                    'lst_max' : 0.0,
                    'med_exptime': 0.0 }
            self.tiles.append(tile)
        self.numtiles = len(self.tiles)
        self.assignHA()

        self.LSTres = 2.5 # bin width in degrees, starting with 10 minutes for now
        self.nLST = int(np.floor(360.0/self.LSTres))
        self.LSTbins = np.zeros(self.nLST)
        #print(self.nLST, self.LSTres)
        for i in range(self.nLST):
            self.LSTbins[i] = (float(i) + 0.5) * self.LSTres
            if self.LSTbins[i] < 180.0:
                self.LSTbins[i] += 360.0
            #print (self.LSTbins[i])

    def assignHA(self):
        """Assigns optimal hour angles for the DESI tiles;
        can be re-run at any point during the survey to
        reoptimise the schedule.
        """
        # Right now, just reads in the pre-computed HA a LSTbegin and LSTend
        hdulist0 = pyfits.open(resource_filename('surveysim', 'data/tile-info.fits'))
        tiledata0 = hdulist0[1].data
        j = 0
        for tile in self.tiles:
            #print ("Assingning HA value for tile ", tile['tileID'])
            while j < len(tiledata0):
                if tile['tileID'] == tiledata0[j][0]:
                    tile['HA'] = tiledata0[j][8]
                    tile['lst_min'] = tiledata0[j][16]
                    tile['lst_max'] = tiledata0[j][17]
                    tile['med_exptime'] = tiledata0[j][14]
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
        uptiles = []
        nto = len(tiles_observed)
        for tile in self.tiles:
            for i in range(nto):
                if tile['tileID'] == tiles_observed['TILEID'][i]:
                    tile['status'] = tiles_observed['STATUS'][i]
                    break
            if tile['status']==0:
                uptiles.append(tile)

        # Dark layers: 0, 1, 2, 3; ELG layer: 4; BGS layers: 5, 6, 7.
        layer0 = []
        layer0_special = []
        layer1 = []
        layer1_special = []
        layer2 = []
        layer2_special = []
        layer3 = []
        layer3_special = []
        layer4 = []
        layer4_special = []
        layer5 = []
        layer5_special = []
        layer6 = []
        layer6_special = []
        layer7 = []
        layer7_special = []
        for tile in uptiles:
            # DARK layer 0
            if tile['PASS']==0:
                if self.inFirstYearFullDepthField(tile['DEC'], tile['cap'], True):
                    layer0_special.append(tile)
                else:
                    layer0.append(tile)
            # DARK layer 1
            elif tile['PASS']==1:
                if self.inFirstYearFullDepthField(tile['DEC'], tile['cap'], False):
                    layer1_special.append(tile)
                else:
                    layer1.append(tile)
            # DARK layer 2
            elif tile['PASS']==2:
                if self.inFirstYearFullDepthField(tile['DEC'], tile['cap'], False):
                    layer2_special.append(tile)
                else:
                    layer2.append(tile)
            # DARK layer 3
            elif tile['PASS']==3:
                if self.inFirstYearFullDepthField(tile['DEC'], tile['cap'], False):
                    layer3_special.append(tile)
                else:
                    layer3.append(tile)
            # ELG layer 4
            elif tile['PASS']==4:
                if self.inFirstYearFullDepthField(tile['DEC'], tile['cap'], True):
                    layer4_special.append(tile)
                else:
                    layer4.append(tile)
            # BGS layer 5
            elif tile['PASS']==5:
                if self.inFirstYearFullDepthField(tile['DEC'], tile['cap'], True):
                    layer5_special.append(tile)
                else:
                    layer5.append(tile)
            # BGS layer 6
            elif tile['PASS']==6:
                if self.inFirstYearFullDepthField(tile['DEC'], tile['cap'], False):
                    layer6_special.append(tile)
                else:
                    layer6.append(tile)
            # BGS layer 7
            elif tile['PASS']==7:
                if self.inFirstYearFullDepthField(tile['DEC'], tile['cap'], False):
                    layer7_special.append(tile)
                else:
                    layer7.append(tile)

        # Order each sublist by DEC and merge to form new ordered tile list
        layer0_special = sorted(layer0_special, key=itemgetter('DEC'), reverse=False)
        layer0 = sorted(layer0, key=itemgetter('DEC'), reverse=False)
        layer1_special = sorted(layer1_special, key=itemgetter('DEC'), reverse=False)
        layer1 = sorted(layer1, key=itemgetter('DEC'), reverse=False)
        layer2_special = sorted(layer2_special, key=itemgetter('DEC'), reverse=False)
        layer2 = sorted(layer2, key=itemgetter('DEC'), reverse=False)
        layer3_special = sorted(layer3_special, key=itemgetter('DEC'), reverse=False)
        layer3 = sorted(layer3, key=itemgetter('DEC'), reverse=False)
        layer4_special = sorted(layer4_special, key=itemgetter('DEC'), reverse=False)
        layer4 = sorted(layer4, key=itemgetter('DEC'), reverse=False)
        layer5_special = sorted(layer5_special, key=itemgetter('DEC'), reverse=False)
        layer5 = sorted(layer5, key=itemgetter('DEC'), reverse=False)
        layer6_special = sorted(layer6_special, key=itemgetter('DEC'), reverse=False)
        layer6 = sorted(layer6, key=itemgetter('DEC'), reverse=False)
        layer7_special = sorted(layer7_special, key=itemgetter('DEC'), reverse=False)
        layer7 = sorted(layer7, key=itemgetter('DEC'), reverse=False)
        
        finalTileList = layer0_special + layer1_special + layer2_special + layer3_special
        finalTileList += layer4_special + layer5_special + layer6_special + layer7_special
        finalTileList += layer0 + layer1 + layer2 + layer3 + layer4 + layer5 + layer6 + layer7
        """
        print( len(layer0_special), len(layer1_special), len(layer2_special), len(layer3_special),
               len(layer4_special), len(layer5_special), len(layer6_special), len(layer7_special),
               len(layer0), len(layer1), len(layer2), len(layer3), len(layer4), len(layer5), len(layer6), len(layer7) )
        """
        # Assign tiles to LST bins
        planList0 = []
        lst15evening = mjd2lst(day_stats['MJDetwi'])
        lst15morning = mjd2lst(day_stats['MJDmtwi']) + 360.0
        lst13evening = mjd2lst(day_stats['MJDe13twi'])
        lst13morning = mjd2lst(day_stats['MJDe13twi']) + 360.0
        LSTmoonrise = mjd2lst(day_stats['MJDmoonrise'])
        if LSTmoonrise < 180.0:
            LSTmoonrise += 360.0
        LSTmoonset = mjd2lst(day_stats['MJDmoonset'])
        if LSTmoonset < 180.0:
            LSTmoonset += 360.0
        if day_stats['MJD_bright_start'] != None:
            LSTbrightstart = mjd2lst(day_stats['MJD_bright_start'])
            if LSTbrightstart < 180.0:
                LSTbrightstart += 360.0
        else:
            LSTbrightstart = -1.0e99
        if day_stats['MJD_bright_end'] != None:
            LSTbrightend = mjd2lst(day_stats['MJD_bright_end'])
            if LSTbrightend < 180.0:
                LSTbrightend += 360.0
        else:
            LSTbrightend = 1.0e99
        #print(lst15evening,lst15morning,lst13evening,lst13morning,LSTmoonrise,LSTmoonset,LSTbrightstart,LSTbrightend)
        # Loop over LST bins
        for i in range(self.nLST):
            # DARK time
            if( (lst15evening < self.LSTbins[i] and self.LSTbins[i] < lst15morning) and
                (LSTmoonrise > self.LSTbins[i] or self.LSTbins[i] > LSTmoonset) ):
                nfields = 0
                for tile in finalTileList:
                    tileLST = tile['RA'] + tile['HA']
                    if tileLST<180.0:
                        tileLST+=360.0
                    if ( tile['status']<2 and
                         tileLST >= self.LSTbins[i] - 0.5*self.LSTres and
                         tileLST <= self.LSTbins[i] + 0.5*self.LSTres and
                         (tile['obsconds'] & obsbits.mask('DARK')) != 0 ):
                        tile['priority'] = nfields + 3
                        tile['lst_min'] = self.LSTbins[i] - 0.5*self.LSTres
                        tile['lst_max'] = self.LSTbins[i] + 0.5*self.LSTres
                        planList0.append(self.dict2tuple(tile))
                        nfields += 1
                    if nfields == 5:
                        break
                    else:
                        continue
                if nfields < 5: # If fewer than 5 dark tiles fall within this window, pad with grey tiles 
                    for tile in finalTileList:
                        tileLST = tile['RA'] + tile['HA']
                        if tileLST<180.0:
                            tileLST+=360.0
                        if ( tile['status']<2 and
                            tileLST >= self.LSTbins[i] - 0.5*self.LSTres and
                            tileLST <= self.LSTbins[i] + 0.5*self.LSTres and
                            (tile['obsconds'] & obsbits.mask('GRAY')) != 0 ):
                            tile['priority'] = nfields + 3
                            tile['lst_min'] = self.LSTbins[i] - 0.5*self.LSTres
                            tile['lst_max'] = self.LSTbins[i] + 0.5*self.LSTres
                            planList0.append(self.dict2tuple(tile))
                            nfields += 1
                        if nfields == 5:
                            break
                        else:
                            continue
                if nfields < 5: # If fewer than 5 dark or grey tiles fall within this window, pad with bright tiles 
                    for tile in finalTileList:
                        tileLST = tile['RA'] + tile['HA']
                        if tileLST<180.0:
                            tileLST+=360.0
                        if ( tile['status']<2 and
                            tileLST >= self.LSTbins[i] - 0.5*self.LSTres and
                            tileLST <= self.LSTbins[i] + 0.5*self.LSTres and
                            (tile['obsconds'] & obsbits.mask('BRIGHT')) != 0 ):
                            tile['priority'] = nfields + 3
                            tile['lst_min'] = self.LSTbins[i] - 0.5*self.LSTres
                            tile['lst_max'] = self.LSTbins[i] + 0.5*self.LSTres
                            planList0.append(self.dict2tuple(tile))
                            nfields += 1
                        if nfields == 5:
                            break
                        else:
                            continue
            # GREY time
            if ( (lst15evening < self.LSTbins[i] and self.LSTbins[i] < lst15morning) and
                 (LSTmoonrise < self.LSTbins[i] and self.LSTbins[i] < LSTmoonset) and
                 (LSTbrightstart > self.LSTbins[i] or self.LSTbins[i] > LSTbrightend) ):
                nfields = 0
                for tile in finalTileList:
                    tileLST = tile['RA'] + tile['HA']
                    if tileLST<180.0:
                        tileLST+=360.0
                    if ( tile['status']<2 and
                         tileLST >= self.LSTbins[i] - 0.5*self.LSTres and
                         tileLST <= self.LSTbins[i] + 0.5*self.LSTres and
                         (tile['obsconds'] & obsbits.mask('GRAY')) != 0 ):
                        tile['priority'] = nfields + 3
                        tile['lst_min'] = self.LSTbins[i] - 0.5*self.LSTres
                        tile['lst_max'] = self.LSTbins[i] + 0.5*self.LSTres
                        planList0.append(self.dict2tuple(tile))
                        nfields += 1
                    if nfields == 5:
                        break
                    else:
                        continue
                if nfields < 5: # If fewer than 5 grey tiles fall within this window, pad with bright tiles 
                    for tile in finalTileList:
                        tileLST = tile['RA'] + tile['HA']
                        if tileLST<180.0:
                            tileLST+=360.0
                        if ( tile['status']<2 and
                            tileLST >= self.LSTbins[i] - 0.5*self.LSTres and
                            tileLST <= self.LSTbins[i] + 0.5*self.LSTres and
                            (tile['obsconds'] & obsbits.mask('BRIGHT')) != 0 ):
                            tile['priority'] = nfields + 3
                            tile['lst_min'] = self.LSTbins[i] - 0.5*self.LSTres
                            tile['lst_max'] = self.LSTbins[i] + 0.5*self.LSTres
                            planList0.append(self.dict2tuple(tile))
                            nfields += 1
                        if nfields == 5:
                            break
                        else:
                            continue
            # BRIGHT time
            if ( (lst13evening < self.LSTbins[i] and self.LSTbins[i] <= lst15evening) or
                 (lst15morning <= self.LSTbins[i] and self.LSTbins[i] < lst13morning) or
                 ( (lst15evening < self.LSTbins[i] and self.LSTbins[i] < lst15morning) and
                   (LSTmoonrise < self.LSTbins[i] and self.LSTbins[i] < LSTmoonset) and
                   (LSTbrightstart < self.LSTbins[i] and self.LSTbins[i] < LSTbrightend) ) ):
                nfields = 0
                for tile in finalTileList:
                    tileLST = tile['RA'] + tile['HA']
                    if tileLST<180.0:
                        tileLST+=360.0
                    if ( tile['status']<2 and
                         tileLST >= self.LSTbins[i] - 0.5*self.LSTres and
                         tileLST <= self.LSTbins[i] + 0.5*self.LSTres and
                         (tile['obsconds'] & obsbits.mask('BRIGHT')) != 0 ):
                        tile['priority'] = nfields + 3
                        tile['lst_min'] = self.LSTbins[i] - 0.5*self.LSTres
                        tile['lst_max'] = self.LSTbins[i] + 0.5*self.LSTres
                        planList0.append(self.dict2tuple(tile))
                        nfields += 1
                    if nfields == 5:
                        break
                    else:
                        continue

        cols = np.rec.array(planList0,
                            names = ('TILEID', 'RA', 'DEC', 'PASS', 'EBV_MED', 'PROGRAM', 'OBSCONDITIONS', 'GAL_CAP', 'PRIORITY', 'STATUS', 'HA', 'LSTMIN', 'LSTMAX', 'EXPLEN'),
                            formats = ['i4', 'f8', 'f8', 'i4', 'f4', 'a6', 'i2', 'i4', 'i4', 'i4', 'f8', 'f8', 'f8', 'f8'])

        tbhdu = pyfits.BinTableHDU.from_columns(cols)

        prihdr = pyfits.Header()
        prihdr['MOONFRAC'] = day_stats['MoonFrac']
        prihdu = pyfits.PrimaryHDU(header=prihdr)
        filename = 'obsplan' + day_stats['dirName'] + '.fits'
        thdulist = pyfits.HDUList([prihdu, tbhdu])
        thdulist.writeto(filename, clobber=True)

        tilesTODO = len(planList0)

        return filename

    def dict2tuple(self, tile):
        """Utility function to convert a dictionary's
        values into a tuple with always the same order"""
        tup0 = (tile['tileID'],tile['RA'],tile['DEC'],tile['PASS'],tile['Ebmv'],tile['program'],tile['obsconds'],tile['cap'],tile['priority'],tile['status'],tile['HA'],tile['lst_min'],tile['lst_max'],tile['med_exptime'])
        return tup0

###############################################################
"""
    def plan_ha(survey_begin, survey_end, exptime):
        #Translation into Python of Kyle's IDL code
        #
        #    Args:
        #        survey_begin: MJD of (re-)start of survey
        #        survey_end: MJD of the expected end
        #        exptime: nominal exposure time (seconds)
        
        # First define general survey characteristics
        r_threshold = 1.54 # this is an initial guess for required SN2/pixel over r-band 
        b_threshold = 0.7 # same for g-band, scaled relative to r-band throughout analysis, the ratio of r-b cannot change
        times = self.LSTbins*24.0/360.0 # LST bins in hours
        scheduled_times = np.zeros(self.nLST) # available hours at each LST bin over the full survey, after accounting for weather loss
        observed_times = np.zeros(self.nLST) # hours spent observing at each LST bin, iteratively filled until optimal HA distribution is achieved
        remaining_times = np.zeros(self.nLST) # simply the difference between scheduled times and observed times
        sgcfraction_times = np.zeros(self.nLST) # the fraction of total time in each bin of LST, SGC
        ngcfraction_times = np.zeros(self.nLST) # the fraction of total time in each bin of LST, NGC
        weather=0.74*0.77    # SRD assumption: 74% open dome and 77% good seeing

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



        compute_extinction (surveystruct,obs,file_tiles,file_plates)

        # FIND MINIMUM AMOUNT OF TIME REQUIRED TO COMPLETE PLATES, ACCOUNTING ONLY FOR ZENITH AVOIDANCE
        run_plates(obs, surveystruct)
        num=n_elements(obs)
        for i=0,num-1 do begin
            ha=0.
            filltimes,obs,surveystruct,ha,i
        # NOW RUN ALL TILES THROUGH THE TIME ALLOCATED AT EACH LST BIN AND FIND CHEAPEST FOOTPRINT TO COVER IN TIME ALLOCATED
        optimize=1
        retile(obs, surveystruct, optimize)

        # ADJUST THRESHOLDS ONCE TO MATCH AVAILABLE LST DISTRIBUTION
        if n_elements(where(obs.obs_bit gt 1))*surveystruct.platearea lt surveystruct.surveyarea:
            a=where(obs.obs_bit gt 1)
            t_scheduled=mean(obs[a].obstime)-mean(obs[a].overhead)
            t_required=mean(obs[a].obstime)*n_elements(where(obs.obs_bit gt 1))*surveystruct.platearea/surveystruct.surveyarea-mean(obs[a].overhead)
            surveystruct.r_threshold=surveystruct.r_threshold*t_required/t_scheduled
            surveystruct.b_threshold=surveystruct.b_threshold*t_required/t_scheduled
        if total(surveystruct.remaining_times) > 0:
            t_scheduled=total(surveystruct.observed_times)/n_elements(obs)-mean(obs[a].overhead)
            t_required=total(surveystruct.scheduled_times)/n_elements(obs)-mean(obs[a].overhead)
            surveystruct.r_threshold=surveystruct.r_threshold*t_required/t_scheduled*excess
            surveystruct.b_threshold=surveystruct.b_threshold*t_required/t_scheduled*excess
        retile(obs, surveystruct, optimize)

        # now run all tiles through survey and round hour angles to nearest degree
        obs.ha=float(round(obs.ha*15.))/15.
        surveystruct.observed_times[*]=0.
        surveystruct.remaining_times[*]=surveystruct.scheduled_times
        run_plates,obs,surveystruct

    def compute_extinction (surveystruct,obs,file_tiles,file_plates):

        data=mrdfits(file_tiles,1)
        a=where(data.pass le 4 AND data.in_desi eq 1)
        data=data[a]
        tileid=data.tileid
        used=tileid*0
        ra=data.ra
        dec=data.dec
        ebv=data.EBV_MED
        num=floor(n_elements(ra)/5.)
        indices=indgen(num)*5
        qsoflag=used*0
        qsoflag[indices]=1
        tileid=fix(tileid)

        a=where(used eq 1 or ebv lt 0.5)
        a=where(ra gt 45 AND ra lt 100 AND dec lt 10)
        a=where(ebv lt 0.45 AND dec gt -14.75 AND dec lt 82.75)
        a=where(ebv lt 45 AND dec gt -100.75 AND dec lt 92.75)
        pass=data[a].pass
        program=data[a].program
        obsconditions=data[a].obsconditions

        qsoflag=qsoflag[a]
        ra=ra[a]
        dec=dec[a]
        ebv=ebv[a]
        num=n_elements(ra)
        tileid=tileid[a]
        locationid=tileid

        i_increase=fltarr(num)
        g_increase=fltarr(num)
        glong=fltarr(num)
        glat=fltarr(num)
        select=fltarr(num)
        overhead=fltarr(num)

        # From http://arxiv.org/pdf/1012.4804v2.pdf Table 6
        # R_u = 4.239
        # R_g = 3.303
        # R_r = 2.285
        # R_i = 1.698
        # R_z = 1.263

        for i=0,num-1 do begin
            ra1=ra[i]
            dec1=dec[i]
            euler,ra1,dec1,aout,bout,1
            glong[i]=aout
            glat[i]=bout
            Ag=3.303*ebv[i]
            Ai=1.698*ebv[i]
            i_increase[i]=(10^(Ai/2.5))^2
            g_increase[i]=(10^(Ag/2.5))^2
            select[i]=randomu(seed)

        ra=ra*24./360.    ; units of hours
        ha=ra*0.
        airmass=ha+1.
        obs_bit=intarr(num)
        obs_bit[*]=0
        obstime=ha
        red_sn=ha
        blue_sn=ha
        beginobs=ha
        endobs=ha
        plateid=round(ha)

        obs = {'tileid' : tileid[0],
                'locationid' : locationid[0],
                'plateid' : plateid[0],
                'qsoflag' : 0,
                'ra' : ra[0],
                'dec' : dec[0],
                'glong' : glong[0],
                'glat' : glat[0],
                'ha' : ha[0],
                'airmass' : airmass[0],
                'ebv' : ebv[0],
                'i_increase' : i_increase[0],
                'g_increase' : g_increase[0],
                'obs_bit' : obs_bit[0],
                'obstime' : obstime[0],
                'overhead' : overhead[0],
                'beginobs' : beginobs[0],
                'endobs' : endobs[0],
                'red_sn' : red_sn[0],
                'blue_sn' : blue_sn[0],
                'pass' : layer[0],
                'PROGRAM' : program[0],
                'OBSCONDITIONS' : OBSCONDITIONS[0]}

        a=where(dec gt -20 AND ((ra gt 5.5 AND ra lt 20) OR (ra lt 5.5 AND dec lt 30) OR (ra gt 20 AND dec lt 30)))
        # NGC at dec>-10 and SGC with DECam coverage
        a=where(dec gt -20 AND ((dec gt -8.75 AND ra gt 5.5 AND ra lt 20) OR (ra lt 5.5 AND dec lt 32.75) OR (ra gt 20 AND dec lt 32.75)))
        # Only DECam coverage in NGC and SGC
        # a=where(dec gt -20 AND dec lt 90)

        obs = replicate(obs, n_elements(ha))

        obs.tileid = tileid[a]
        obs.locationid = locationid[a]
        obs.plateid = plateid[a]
        obs.layer = layer[a]
        obs.ra = ra[a]
        obs.dec = dec[a]
        obs.glong = glong[a]
        obs.glat = glat[a]
        obs.ha = ha[a]
        obs.airmass = airmass[a]
        obs.ebv = ebv[a]
        obs.i_increase = i_increase[a]
        obs.g_increase = g_increase[a]
        obs.obs_bit = obs_bit[a]
        obs.obstime = obstime[a]
        obs.overhead = overhead[a]
        obs.beginobs = beginobs[a]
        obs.endobs = endobs[a]
        obs.red_sn = red_sn[a]
        obs.blue_sn = blue_sn[a]
        obs.qsoflag = qsoflag[a]
        obs.program = program[a]
        obs.obsconditions = obsconditions[a]


    def retile (obs, surveystruct, optimize):

        times=surveystruct.times
        num_times=n_elements(surveystruct.times)
        rank_times=fltarr(num_times)
        rank_plates=fltarr(n_elements(obs.tileid))
        kpno_lat=surveystruct.kpno_lat
        dec=obs.dec

    for i=0,num_times-1:
        ha_tmp=surveystruct.times[i]-obs.ra
        a=where(ha_tmp ge 12.,count)
        if count gt 0.:
            ha_tmp[a]=ha_tmp[a]-24.
        a=where(ha_tmp le -12.,count)
        if count gt 0.:
            ha_tmp[a]=ha_tmp[a]+24.
        compute_airmass (kpno_lat,dec,ha_tmp,airmass_tmp)
        temp_rank=airmass_tmp^surveystruct.alpha_red*obs.i_increase
        a=where(obs.dec gt 10)
        temp_rank=temp_rank[sort(temp_rank)]
        rank_times[i]=mean(temp_rank[0:50])

    r1=max(rank_times[0:num_times/2-1])
    r2=max(rank_times[num_times/2:num_times-1])
    a=where(rank_times eq r1 OR rank_times eq r2)
    a[0]=a[0]+2
    a[1]=a[1]-7
    t=[times[a[0]],times[a[1]]]
    index1=a[0]
    index2=a[1]

    ngctime=total(surveystruct.scheduled_times[index1+1:index2-1]) + 0.5*surveystruct.scheduled_times[index1] + 0.5*surveystruct.scheduled_times[index2]
    sgctime=total(surveystruct.scheduled_times[0:index1-1]) + total(surveystruct.scheduled_times[index2+1:num_times-1]) + 0.5*surveystruct.scheduled_times[index1] + 0.5*surveystruct.scheduled_times[index2]
    surveystruct.sgcfraction_times[0:index1]=surveystruct.scheduled_times[0:index1]/sgctime
    surveystruct.sgcfraction_times[index2:num_times-1]=surveystruct.scheduled_times[index2:num_times-1]/sgctime
    surveystruct.ngcfraction_times[index1:index2]=surveystruct.scheduled_times[index1:index2]/ngctime
    surveystruct.sgcfraction_times[index1]=0.5*surveystruct.scheduled_times[index1]/sgctime
    surveystruct.ngcfraction_times[index1]=0.5*surveystruct.scheduled_times[index1]/ngctime
    surveystruct.sgcfraction_times[index2]=0.5*surveystruct.scheduled_times[index2]/sgctime
    surveystruct.ngcfraction_times[index2]=0.5*surveystruct.scheduled_times[index2]/ngctime

    obs[*].obstime=0.
    sgcplates=where(obs.ra lt surveystruct.ngc_begin OR obs.ra gt surveystruct.ngc_end)
    ngcplates=where(obs.ra gt surveystruct.ngc_begin AND obs.ra lt surveystruct.ngc_end)
    sgc_obstime=0.
    ngc_obstime=0.

    surveystruct.remaining_times=surveystruct.scheduled_times
    surveystruct.observed_times[*]=0.
    a=where(obs.obs_bit le 2)
    obs[a].obs_bit=0
    obs[a].ha=0.
    run_plates,obs,surveystruct
    surveystruct.observed_times[*]=0.
    surveystruct.remaining_times=surveystruct.scheduled_times
    exptime=surveystruct.exptime

    # fill regions that are constrained by zenith avoidance and which otherwise would prefer lower HA
    # also applies to other plates constrained for a specific HA
    a=where(obs.obs_bit eq 3,count)
    if count gt 0:
        for i=0,count-1:
            index=a[i]
            h=obs[index].ha
            filltimes,obs,surveystruct,h,index

    # Start by filling the hardest regions with tiles, NGC then SGC

    indices=intarr(index2-index1+1)
    for i=0,index2-index1:
        if i/2 eq float(i)/2:
            indices[i]=index1+i/2 $
        else:
            indices[i]=index2-i/2
            
    dec=obs[ngcplates].dec
    orig_ha=obs[ngcplates].ha
    transit=obs[ngcplates].obs_bit*0

    for i=0,n_elements(indices)-1:
        index=indices[i]
        num_reqplates=ceil((surveystruct.ngcfraction_times[index]*ngctime-surveystruct.observed_times[index])/surveystruct.res)
        tile=obs[ngcplates].tileid
        obs_bit=obs[ngcplates].obs_bit
        if i/2 eq float(i)/2:
            ha=times[index]-obs[ngcplates].ra-surveystruct.res/2.
        if i/2 ne float(i)/2:
            ha=times[index]-obs[ngcplates].ra+surveystruct.res/2.
        a=where(ha ge 12.,count)
        if count gt 0.:
            ha[a]=ha[a]-24.
        a=where(ha le -12.,count)
        if count gt 0.:
            ha[a]=ha[a]+24.
        a=where(obs[ngcplates].ra+orig_ha gt surveystruct.times[i]-surveystruct.res/2. AND obs[ngcplates].ra+orig_ha le surveystruct.times[i]+surveystruct.res/2.,count)
        if count gt 0:
            transit[a]=1
        compute_airmass(kpno_lat,dec,ha,airmass_tmp)
        compute_airmass(kpno_lat,dec,orig_ha,orig_airmass)
        rank_plates=airmass_tmp^surveystruct.alpha_red*obs[ngcplates].i_increase
        if optimize:
            rank_plates=rank_plates-orig_airmass^surveystruct.alpha_red*obs[ngcplates].i_increase
        a=where(obs.obs_bit eq 1 AND abs(ha) lt 1,count)
        if count gt 0 AND optimize eq 0:
            rank_plates[a]=1000.
        a=where(obs_bit lt 2, count)
        if count eq 0:
            break
        num_reqplates=min([num_reqplates,count])
        rank_plates=rank_plates[a]
        tile=tile[a]
        tile=(tile[sort(rank_plates)])
        ha=ha[a]
        ha=(ha[sort(rank_plates)])
        for j=0,num_reqplates-1:
            a=where(obs.tileid eq tile[j])
            index=a[0]
            d=obs[index].dec
            h=ha[j]
            compute_airmass,surveystruct.kpno_lat,d,h,airmass
            red=surveystruct.avg_rsn/airmass^surveystruct.alpha_red/obs[index].i_increase
            rtime=surveystruct.overhead1+surveystruct.exptime*surveystruct.r_threshold/red
            blue=surveystruct.avg_bsn/airmass^surveystruct.alpha_blue/obs[index].g_increase
            btime=surveystruct.overhead1+surveystruct.exptime*surveystruct.b_threshold/blue
            time=max([rtime,btime])
            if i/2 eq float(i)/2:
                h=h+time/2.
            if i/2 ne float(i)/2:
                h=h-time/2.
            obs[index].obs_bit=2
            filltimes,obs,surveystruct,h,index
            obs[index].ha=h

    indices=intarr(index1+num_times-index2+1)
    for i=0,index1+num_times-index2:
        if i/2 eq float(i)/2:
            indices[i]=index1-i/2 $
        else:
            indices[i]=index2+i/2
    a=where(indices lt 0,count)
    if count gt 0:
        indices[a]=indices[a]+num_times
    a=where(indices ge num_times,count)
    if count gt 0 then indices[a]=indices[a]-num_times

    dec=obs[sgcplates].dec

    for i=0,n_elements(indices)-1:
        index=indices[i]
        num_reqplates=ceil((surveystruct.scheduled_times[index]-surveystruct.observed_times[index])/surveystruct.res)
        tile=obs[sgcplates].tileid
        obs_bit=obs[sgcplates].obs_bit
        if i/2 ne float(i)/2:
            ha=times[index]-obs[sgcplates].ra-surveystruct.res/2.
        if i/2 eq float(i)/2:
            ha=times[index]-obs[sgcplates].ra+surveystruct.res/2.
        a=where(ha ge 12.,count)
        if count gt 0.:
            ha[a]=ha[a]-24.
        a=where(ha le -12.,count)
        if count gt 0.:
            ha[a]=ha[a]+24.
        compute_airmass,kpno_lat,dec,ha,airmass_tmp
        compute_airmass,kpno_lat,dec,orig_ha,orig_airmass
        rank_plates=airmass_tmp^surveystruct.alpha_red*obs[sgcplates].i_increase
        if optimize:
            rank_plates=rank_plates-orig_airmass^surveystruct.alpha_red*obs[sgcplates].i_increase
        a=where(obs.obs_bit eq 1 AND abs(ha) lt 1,count)
        if count gt 0 AND optimize eq 0:
            rank_plates[a]=1000.
        a=where(obs_bit lt 2, count)
        if count eq 0:
            break
        num_reqplates=min([num_reqplates,count])
        rank_plates=rank_plates[a]
        tile=tile[a]
        tile=(tile[sort(rank_plates)])
        ha=ha[a]
        ha=(ha[sort(rank_plates)])
        for j=0,num_reqplates-1:
            a=where(obs.tileid eq tile[j])
            index=a[0]
            d=obs[index].dec
            h=ha[j]
            compute_airmass,surveystruct.kpno_lat,d,h,airmass
            red=surveystruct.avg_rsn/airmass^surveystruct.alpha_red/obs[index].i_increase
            rtime=surveystruct.overhead1+surveystruct.exptime*surveystruct.r_threshold/red
            blue=surveystruct.avg_bsn/airmass^surveystruct.alpha_blue/obs[index].g_increase
            btime=surveystruct.overhead1+surveystruct.exptime*surveystruct.b_threshold/blue
            time=max([rtime,btime])
            if i/2 ne float(i)/2 then h=h+time/2.
            if i/2 eq float(i)/2 then h=h-time/2.
            obs[index].obs_bit=2
            filltimes,obs,surveystruct,h,index
            obs[index].ha=h

"""
