import numpy as np
import astropy.io.fits as pyfits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from pkg_resources import resource_filename
from operator import itemgetter
from desisurvey.utils import radec2altaz, mjd2lst, equ2gal_J2000
from desitarget.targetmask import obsconditions as obsbits

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
        hdulist0 = pyfits.open(resource_filename('surveysim', 'data/tile-info.fits'))
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
                    'Pass' : tiledata[i].field('PASS'),
                    'Ebmv' : tiledata[i].field('EBV_MED'),
                    'program' : tiledata[i].field('PROGRAM'),
                    'obsconds' : tiledata[i].field('OBSCONDITIONS'),
                    'cap' : b / np.abs(b),
                    'priority' : 0,
                    'status' : 0,
                    'ha' : 0.0,
                    'lst_min' : 0.0,
                    'lst_max' : 0.0 }
            self.tiles.append(tile)
        self.assignHA()

        self.LSTres = 2.5 # bin width in degrees, starting with 10 minutes for now
        self.nLST = int(np.floor(360.0/LSTres))
        self.LSTbins = np.zeros(self.nLST)
        for i in range(self.nLST):
            self.LSTbins = (float(i) + 0.5) * self.LSTres

    def assignHA(None):
        """Assigns optimal hour angles for the DESI tiles;
        can be re-run at any point during the survey to
        reoptimise the schedule.
        """
        return 0 # This function is now just a place holder.

    def inFirstYearFullDepthField(dec, bgal, first_pass):
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
        nto = len(tiles_observed)
        for i in range(nto):
            j = np.where(self.tiles['tileID'] == tiles_observed['TILEID'][i])
            self.tiles['status'] = tiles_observed['STATUS'][i]
        i_todo = np.where(self.tiles['STATUS'] == 0) # To be replaced to account for partially observed tile
        uptiles = self.tiles[i_todo]

        # Dark layers: 0, 1, 2, 3; ELG layer: 4; BGS layers: 5, 6, 7.
        # DARK layer 0
        il0 = np.where(uptiles['PASS']==0)
        layer0_all = uptiles[il0]
        layer0_all = sorted(layer0_all, key=itemgetter('DEC'), reverse=False)
        layer0 = []
        layer0_special = []
        for tile in layer0_all:
            if inFirstYearFullDepthField(tile['DEC'], tile['cap'], True):
                layer0_special.append(tile)
            else:
                layer0.append(tile)
        layer0_all.clear()
        # DARK layer 1
        il1 = np.where(uptiles['PASS']==1)
        layer1_all = uptiles[il1]
        layer1_all = sorted(layer1_all, key=itemgetter('DEC'), reverse=False)
        layer1 = []
        layer1_special = []
        for tile in layer1_all:
            if inFirstYearFullDepthField(tile['DEC'], tile['cap'], False):
                layer1_special.append(tile)
            else:
                layer1.append(tile)
        layer1_all.clear()
        # DARK layer 2
        il1 = np.where(uptiles['PASS']==2)
        layer2_all = uptiles[il2]
        layer2_all = sorted(layer2_all, key=itemgetter('DEC'), reverse=False)
        layer2 = []
        layer2_special = []
        for tile in layer2_all:
            if inFirstYearFullDepthField(tile['DEC'], tile['cap'], False):
                layer2_special.append(tile)
            else:
                layer2.append(tile)
        layer2_all.clear()
        # DARK layer 3
        il3 = np.where(uptiles['PASS']==3)
        layer3_all = uptiles[il3]
        layer3_all = sorted(layer3_all, key=itemgetter('DEC'), reverse=False)
        layer3 = []
        layer3_special = []
        for tile in layer3_all:
            if inFirstYearFullDepthField(tile['DEC'], tile['cap'], False):
                layer3_special.append(tile)
            else:
                layer3.append(tile)
        layer3_all.clear()
        # ELG layer 4
        il4 = np.where(uptiles['PASS']==4)
        layer4_all = uptiles[il4]
        layer4_all = sorted(layer0_all, key=itemgetter('DEC'), reverse=False)
        layer4 = []
        layer4_special = []
        for tile in layer4_all:
            if inFirstYearFullDepthField(tile['DEC'], tile['cap'], True):
                layer4_special.append(tile)
            else:
                layer4.append(tile)
        layer4_all.clear()
        # BGS layer 5
        il5 = np.where(uptiles['PASS']==5)
        layer5_all = uptiles[il5]
        layer5_all = sorted(layer5_all, key=itemgetter('DEC'), reverse=False)
        layer5 = []
        layer5_special = []
        for tile in layer5_all:
            if inFirstYearFullDepthField(tile['DEC'], tile['cap'], True):
                layer5_special.append(tile)
            else:
                layer5.append(tile)
        layer5_all.clear()
        # BGS layer 6
        il6 = np.where(uptiles['PASS']==6)
        layer6_all = uptiles[il6]
        layer6_all = sorted(layer6_all, key=itemgetter('DEC'), reverse=False)
        layer6 = []
        layer6_special = []
        for tile in layer6_all:
            if inFirstYearFullDepthField(tile['DEC'], tile['cap'], False):
                layer6_special.append(tile)
            else:
                layer6.append(tile)
        layer6_all.clear()
        # BGS layer 7
        il7 = np.where(uptiles['PASS']==7)
        layer7_all = uptiles[il7]
        layer7_all = sorted(layer7_all, key=itemgetter('DEC'), reverse=False)
        layer7 = []
        layer7_special = []
        for tile in layer7_all:
            if inFirstYearFullDepthField(tile['DEC'], tile['cap'], False):
                layer7_special.append(tile)
            else:
                layer7.append(tile)
        layer7_all.clear()
        # Merge to form new ordered tile list
        finalTileList = layer0_special + layer1_special + layer2_special + layer3_special
        finalTileList += layer4_special + layer5_special + layer6_special + layer7_special
        finalTileList += layer0 + layer1 + layer2 + layer3 + layer4 + layer5 + layer6 + layer7

        # Assign tiles to LST bins
        planList0 = []
        lst15evening = mjd2lst(day_stats['MJDetwi'])
        lst15morning = mjd2lst(day_stats['MJDmtwi'])
        lst13evening = mjd2lst(day_stats['MJDe13twi'])
        lst13morning = mjd2lst(day_stats['MJDe13twi'])
        LSTmoonrise = mjd2lst(day_stats['MJDmoonrise'])
        LSTmoonset = mjd2lst(day_stats['MJDmoonset'])
        if day_stats['MJD_bright_start'] != None:
            LSTbrightstart = mjd2lst(day_stats['MJD_bright_start'])
        else:
            LSTbrightstart = -1.0e99
        if day_stats['MJD_bright_end'] != None:
            LSTbrightend = mjd2lst(day_stats['MJD_bright_end'])
        else:
            LSTbrightend = 1.0e99
        # Loop over LST bins
        for i in range(self.nLST):
            # DARK time
            if( (lst15evening < self.LSTbins[i] and self.LSTbins[i] < lst15morning) and
                (LSTmoonrise > self.LSTbins[i] or self.LSTbins[i] > LSTmoonset) ):
                nfields = 0
                for tile in finalTileList:
                    if ( tile['status']<2 and
                         tile['RA'] + tile['HA'] >= self.LSTbins[i] - 0.5*self.LSTres and
                         tile['RA'] + tile['HA'] <= self.LSTbins[i] + 0.5*self.LSTres and
                         (obsconds[i] & obsbits.mask('DARK')) != 0 ):
                        tile['priority'] = nfields + 3
                        tile['lst_min'] = self.LSTbins[i] - 0.5*self.LSTres
                        tile['lst_max'] = self.LSTbins[i] + 0.5*self.LSTres
                        planList0.append(tile)
                        nfields += 1
                    if nfields == 5:
                        break
                    else:
                        continue
                if nfields < 5: # If fewer than 5 dark tiles fall within this window, pad with grey tiles 
                    for tile in finalTileList:
                    if ( tile['status']<2 and
                         tile['RA'] + tile['HA'] >= self.LSTbins[i] - 0.5*self.LSTres and
                         tile['RA'] + tile['HA'] <= self.LSTbins[i] + 0.5*self.LSTres and
                         (obsconds[i] & obsbits.mask('GRAY')) != 0 ):
                        tile['priority'] = nfields + 3
                        tile['lst_min'] = self.LSTbins[i] - 0.5*self.LSTres
                        tile['lst_max'] = self.LSTbins[i] + 0.5*self.LSTres
                        planList0.append(tile)
                        nfields += 1
                    if nfields == 5:
                        break
                    else:
                        continue
                if nfields < 5: # If fewer than 5 dark or grey tiles fall within this window, pad with bright tiles 
                    for tile in finalTileList:
                    if ( tile['status']<2 and
                         tile['RA'] + tile['HA'] >= self.LSTbins[i] - 0.5*self.LSTres and
                         tile['RA'] + tile['HA'] <= self.LSTbins[i] + 0.5*self.LSTres and
                         (obsconds[i] & obsbits.mask('BRIGHT')) != 0 ):
                        tile['priority'] = nfields + 3
                        tile['lst_min'] = self.LSTbins[i] - 0.5*self.LSTres
                        tile['lst_max'] = self.LSTbins[i] + 0.5*self.LSTres
                        planList0.append(tile)
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
                    if ( tile['status']<2 and
                         tile['RA'] + tile['HA'] >= self.LSTbins[i] - 0.5*self.LSTres and
                         tile['RA'] + tile['HA'] <= self.LSTbins[i] + 0.5*self.LSTres and
                         (obsconds[i] & obsbits.mask('GRAY')) != 0 ):
                        tile['priority'] = nfields + 3
                        tile['lst_min'] = self.LSTbins[i] - 0.5*self.LSTres
                        tile['lst_max'] = self.LSTbins[i] + 0.5*self.LSTres
                        planList0.append(tile)
                        nfields += 1
                    if nfields == 5:
                        break
                    else:
                        continue
                if nfields < 5: # If fewer than 5 grey tiles fall within this window, pad with bright tiles 
                    for tile in finalTileList:
                    if ( tile['status']<2 and
                         tile['RA'] + tile['HA'] >= self.LSTbins[i] - 0.5*self.LSTres and
                         tile['RA'] + tile['HA'] <= self.LSTbins[i] + 0.5*self.LSTres and
                         (obsconds[i] & obsbits.mask('BRIGHT')) != 0 ):
                        tile['priority'] = nfields + 3
                        tile['lst_min'] = self.LSTbins[i] - 0.5*self.LSTres
                        tile['lst_max'] = self.LSTbins[i] + 0.5*self.LSTres
                        planList0.append(tile)
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
                    if ( tile['status']<2 and
                         tile['RA'] + tile['HA'] >= self.LSTbins[i] - 0.5*self.LSTres and
                         tile['RA'] + tile['HA'] <= self.LSTbins[i] + 0.5*self.LSTres and
                         (obsconds[i] & obsbits.mask('BRIGHT')) != 0 ):
                        tile['priority'] = nfields + 3
                        tile['lst_min'] = self.LSTbins[i] - 0.5*self.LSTres
                        tile['lst_max'] = self.LSTbins[i] + 0.5*self.LSTres
                        planList0.append(tile)
                        nfields += 1
                    if nfields == 5:
                        break
                    else:
                        continue

        cols = np.rec.array(planList0,
                            names = ('TILEID', 'RA', 'DEC', 'PASS', 'EBV_MED', 'PROGRAM', 'OBSCONDITIONS', 'GAL_CAP', 'PRIORITY', 'STATUS', 'HA', 'LST_OBS'),
                            formats = ['i4', 'f8', 'f8', 'i4', 'f4', 'a6', 'i2', 'i4', 'i4', 'i4', 'f8', 'f8'])

        tbhdu = pyfits.BinTableHDU.from_columns(cols)

        prihdr = pyfits.Header()
        prihdr['MOONFRAC'] = day_stats['MoonFrac']
        prihdu = pyfits.PrimaryHDU(header=prihdr)
        filename = 'obsplan' + day_stats['dirName'] + '.fits'
        thdulist = pyfits.HDUList([prihdu, tbhdu])
        thdulist.writeto(filename, clobber=True)

        tilesTODO = len(planList)

        return tilesTODO, filename

