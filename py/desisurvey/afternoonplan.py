import numpy as np
import astropy.io.fits as pyfits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from pkg_resources import resource_filename
from operator import itemgetter
from desisurvey.utils import radec2altaz, mjd2lst
from desitarget.targetmask import obsconditions as obsbits

class surveyPlan:
    """
    Main class for survey planning
    """
    
    def __init__(self, tilesubset=None):
        """
        Initialises survey by reading in the file desi_tiles.fits
        and populates the class members.

        Optional:
            tilesubset: array of integer tileids to use; ignore others

        Note:
           Temporarily, the file contains an extra column compared to the one on desimodel,
           i.e. the assigned LST. The computation of this quantity will be integrated into
           this code.
        """

        hdulist0 = pyfits.open(resource_filename('surveysim', 'data/tile-info.fits'))
        tiledata = hdulist0[1].data
        tileID = tiledata.field('TILEID')
        RA = tiledata.field('RA')
        DEC = tiledata.field('DEC')
        Pass = tiledata.field('PASS')
        #InDESI = tiledata.field('IN_DESI') Has been removed from list.
        InDESI = np.ones(len(tileID), dtype=np.int8)
        Ebmv = tiledata.field('EBV')
        AM = tiledata.field('AIRMASS')
        #expFac = tiledata.field('EXPOSEFAC')
        #starDensity = tiledata.field('STAR_DENSITY')
        program = tiledata.field('PROGRAM')
        obsconds = tiledata.field('OBSCONDITIONS')
        obstime = tiledata.field('OBSTIME')
        lstbegin = tiledata.field('BEGINOBS')
        lstend = tiledata.field('ENDOBS')
        HA = tiledata.field('HA')
        bgal = tiledata.field('GLAT')
        hdulist0.close()

        #- Trim to requested subset of tiles if specified
        if tilesubset is not None:
            InDESI &= np.in1d(tileID, tilesubset)

        self.tileID = tileID.compress((InDESI==1).flat) #Assuming 0=out, 1=in
        self.RA = RA.compress((InDESI==1).flat)
        self.DEC = DEC.compress((InDESI==1).flat)
        self.Pass = Pass.compress((InDESI==1).flat)
        self.Ebmv = Ebmv.compress((InDESI==1).flat)
        #self.maxExpLen = 2.0 * obstime.compress((InDESI==1).flat)
        self.maxExpLen = obstime.compress((InDESI==1).flat)
        #self.starDensity = starDensity.compress((InDESI==1).flat)
        self.program = program.compress((InDESI==1).flat)
        self.obsconds = obsconds.compress((InDESI==1).flat)
        self.LSTmin = lstbegin.compress((InDESI==1).flat) * 15.0 - 2.5
        self.LSTmax = lstend.compress((InDESI==1).flat) * 15.0 + 2.5
        
        #LST = self.RA + HA.compress((InDESI==1).flat)
        #self.LSTmin = LST - 5.0
        #for i in range(len(self.LSTmin)):
        #    if self.LSTmin[i] < 0.0:
        #        self.LSTmin[i] += 360.0
        #    elif self.LSTmin[i] >360.0:
        #        self.LSTmin[i] -= 360.0
        #self.LSTmax = LST + 5.0
        #for i in range(len(self.LSTmax)):
        #    if self.LSTmax[i] < 0.0:
        #        self.LSTmax[i] += 360.0
        #    elif self.LSTmax[i] > 360.0:
        #        self.LSTmax[i] -= 360.0
        
        self.cap = np.chararray(len(self.tileID))
        btemp = bgal.compress((InDESI==1).flat)
        # Formulae for ecliptic to galactic (angles in degrees):
        # tan(l-303) = sin(192.25-alpha) / (cos(192.25-alpha)sin(27.4)-tan(delta)cos(27.4))
        # sin(b) = sin(delta)sin(27.4) + cos(delta)cos(27.4)cos(192.5-alpha)
        # These are good for B1950, but it should be good enough for our purposes.
        for i in range(len(self.cap)):
            #a = np.radians(192.25 - self.RA[i])
            #b = np.radians(self.DEC[i])
            #c = np.radians(27.4)
            #if  (np.sin(b)*np.sin(c) + np.cos(b)*np.cos(c)*np.cos(a)) >= 0.0:
            if btemp[i] >= 0.0:
                self.cap[i] = 'N'
            else:
                self.cap[i] = 'S'
        self.status = np.zeros(len(self.tileID)) # This should be obsbit, but it is set to 2 everywhere.
        self.priority = np.zeros(len(self.tileID))
        # Assign priority as a function of DEC; this will
        # be adjested in the afternoon planning stage.
        for i in range(len(self.priority)):
            dec = self.DEC[i]
            if  dec <= 15.0:
                priority = 3
            elif dec > 15.0 and dec <= 30.0:
                priority = 4
            elif dec > 30.0 and dec <= 45.0:
                priority = 5
            elif dec > 45.0 and dec <= 60.0:
                priority = 6
            else:
                priority = 7
            self.priority[i] = priority


    def afternoonPlan(self, day_stats, tiles_observed):
        """
        All the file names are hard coded, so there is no need to
        have them as arguments to this function.

        Args:
            day_stats: dictionnary containing the following keys:
                       'MJDsunset', 'MJDsunrise', 'MJDetwi', 'MJDmtwi', 'MJDe13twi',
                       'MJDm13twi', 'MJDmoonrise', 'MJDmoonset', 'MoonFrac', 'dirName'
            tiles_observed: table with follwing columns: tileID, status

        Returns:
            string containg the filename for today's plan; it has the format
            obsplanYYYYMMDD.fits
        """

        year = int(np.floor( (day_stats['MJDsunset'] - tiles_observed.meta['MJDBEGIN'])) / 365.25 ) + 1
        # Adjust DARK time program tile priorities
        # From the DESI document 1767 (v3) "Baseline survey strategy":
        # In the northern galactic cap:
        # Year - Layer 1 tiles - Layer 2 tiles - Layer 3 tiles - Layer 4 tiles
        # 1         900              200              0                0
        # 2         485              415            200              200
        # 3           0              770            100              100
        # 4           0                0            450              450
        # 5           0                0            685              685
        # In the southern galactic cap:
        # Year - Layer 1 tiles - Layer 2 tiles - Layer 3 tiles - Layer 4 tiles
        # 1         450                0              0                0
        # 2         165              300              0                0
        # 3           0              315             90               90
        # 4           0                0            265              260
        # 5           0                0            260              265
        # Priorities shall be adjusted accordingly.
        # Priorities can only be set to 0, 1, 2 or 8, 9, 10 by
        # *human intervention*!

        # Update status
        nto = len(tiles_observed)
        for i in range(nto):
            j = np.where(self.tileID == tiles_observed['TILEID'][i])
            self.status[j] = tiles_observed['STATUS'][i]

        planList0 = []

        lst15evening = mjd2lst(day_stats['MJDetwi'])
        lst15morning = mjd2lst(day_stats['MJDmtwi'])
        lst13evening = mjd2lst(day_stats['MJDe13twi'])
        lst13morning = mjd2lst(day_stats['MJDe13twi'])
        # Dark and grey Pass 1, 2, 3, & 4 are numbered 0, 1, 2, 3.
        # BGS, pass 1, 2 & 3 are numbered 4, 5, 6.
        for i in range(len(self.tileID)):
            if ( self.status[i] < 2 ):
                # Add this tile to the plan, first adjust its priority.
                if ( ((self.obsconds[i] & obsbits.mask('DARK|GRAY')) != 0) and
                     (lst15evening < self.LSTmin[i] and self.LSTmax[i] < lst15morning) ):
                    if year == 1:
                        if ( (self.cap[i] == 'N' and (self.Pass[i] == 2 or self.Pass[i] == 3)) or
                            (self.cap[i] == 'S' and (self.Pass[i] == 1 or self.Pass[i] == 2 or self.Pass[i] == 3)) ):
                            self.priority[i] = 7
                        if ( self.cap[i] == 'N' and self.Pass[i] == 0 and self.priority[i] > 3):
                            self.priority[i] -= 1
                    if year == 2:
                        if ( self.cap[i] == 'S' and (self.Pass[i] == 2 or self.Pass[i] == 3) ):
                            self.priority[i] = 7
                    if year == 3:
                        if self.Pass[i] == 0:
                            self.priority[i] = 3
                        if self.Pass[i] == 1 and self.priority[i] > 3:
                            self.priority[i] -= 1
                    if year >= 4:
                        if self.Pass[i] <= 1:
                            self.priority[i] = 3
                elif ( ((self.obsconds[i] & obsbits.mask('BRIGHT')) != 0) and
                       (lst13evening < self.LSTmin[i] and self.LSTmax[i] < lst13morning) ):
                    if year == 1:
                        if ( self.Pass[i] == 4 or self.Pass[i] == 5 ):
                            self.priority[i] -= 1
                    if year == 2 or year == 3:
                        if self.Pass[i] == 4:
                            self.priority[i] = 3
                        if self.Pass[i] == 5:
                            self.priority[i] -= 1
                    if year >= 4:
                        if self.Pass[i] <= 5:
                            self.priority[i] = 3
                    
                planList0.append((self.tileID[i], self.RA[i], self.DEC[i], self.Ebmv[i], self.LSTmin[i], self.LSTmax[i],
                                 self.maxExpLen[i], self.priority[i], self.status[i], self.program[i], self.obsconds[i]))

        planList = sorted(planList0, key=itemgetter(7), reverse=False)
        cols = np.rec.array(planList,
                            names = ('TILEID', 'RA', 'DEC', 'EBV_MED', 'LSTMIN', 'LSTMAX', 'MAXEXPLEN', 'PRIORITY', 'STATUS', 'PROGRAM', 'OBSCONDITIONS'),
                            formats = ['i4', 'f8', 'f8', 'f8', 'f4', 'f4', 'f4', 'i4', 'i4', 'a6', 'i2'])

        tbhdu = pyfits.BinTableHDU.from_columns(cols)

        prihdr = pyfits.Header()
        prihdr['MOONFRAC'] = day_stats['MoonFrac']
        prihdu = pyfits.PrimaryHDU(header=prihdr)
        filename = 'obsplan' + day_stats['dirName'] + '.fits'
        thdulist = pyfits.HDUList([prihdu, tbhdu])
        thdulist.writeto(filename, clobber=True)

        tilesTODO = len(planList)

        return tilesTODO, filename

