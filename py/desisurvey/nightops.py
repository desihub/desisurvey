#! /usr/bin/python
from __future__ import print_function, division

import numpy as np
from datetime import datetime, timedelta
import os
from shutil import copyfile
from astropy.time import Time
import astropy.io.fits as pyfits
from astropy.table import Table, vstack
from desisurvey.exposurecalc import expTimeEstimator, airMassCalculator
from desisurvey.utils import mjd2lst
from desisurvey.nextobservation import nextFieldSelector
from surveysim.observefield import observeField
import desisurvey.nightcal


class obsCount:
    """
    Counter for observation number.  In real operations, each observation
    will have its own file with its number as part of the filename.
    """

    def __init__(self, start_val=0):
        """
        Initialise the counter to zero
        """
        self.obsNumber = start_val

    def update(self):
        """
        Adds 1 to the counter

        Returns:
            string containing the part of the filename with the observation number
        """
        self.obsNumber += 1
        return '{:08d}'.format(self.obsNumber)

def nightOps(day_stats, obsplan, w, ocnt, tilesObserved, tableOutput=True, use_jpl=False):
    """
    Carries out observations during one night and writes the output to disk

    Args:
        day_stats: dictionnary containing the follwing keys:
                   'MJDsunset', 'MJDsunrise', 'MJDetwi', 'MJDmtwi', 'MJDe13twi',
                   'MJDm13twi', 'MJDmoonrise', 'MJDmoonset', 'MoonFrac', 'dirName'
        obsplan: string, filename of today's afternoon plan
        w: dictionnary containing the following keys
           'Seeing', 'Transparency', 'OpenDome', 'Clouds'
        ocnt: obsCount object
        tilesObserved: table with follwing columns: tileID, status
        tableOutput: bool, if True writes a table of all the night's observations
                     instead of one file per observation.

    Returns:
        Updated tilesObserved table
    """

    nightOver = False
    mjd = day_stats['MJDsunset']

    if tableOutput:
        obsList = []
    else:
        os.mkdir(day_stats['dirName'])

    conditions = w.getValues(mjd)
    f = open("nightstats.dat", "a+")
    if conditions['OpenDome']:
        wcondsstr = "1 " + str(conditions['Seeing']) + " " + str(conditions['Transparency']) + " " + str(conditions['Clouds']) + "\n"
        f.write(wcondsstr)
    else:
        wcondsstr = "0 " + str(conditions['Seeing']) + " " + str(conditions['Transparency']) + " " + str(conditions['Clouds']) + "\n"
        f.write(wcondsstr)
    f.close()
    if conditions['OpenDome'] == False:
        print("\nBad weather forced the dome to remain shut for the night.")
    else:
        print("\nConditions at the beginning of the night: ")
        print("\tSeeing: ", conditions['Seeing'], "arcseconds")
        print("\tTransparency: ", conditions['Transparency'])
        print("\tCloud cover: ", 100.0*conditions['Clouds'], "%")

        # Initialize a moon (alt, az) interpolator using the pre-tabulated
        # ephemerides for this night.
        moon_pos = desisurvey.nightcal.get_moon_interpolator(day_stats)

        slew = False
        ra_prev = 1.0e99
        dec_prev = 1.0e99
        while nightOver == False:
            conditions = w.updateValues(conditions, mjd)

            lst = mjd2lst(mjd)
            moon_alt, moon_az = moon_pos(mjd)
            target, setup_time = nextFieldSelector(
                obsplan, mjd, conditions, tilesObserved, slew,
                ra_prev, dec_prev, moon_alt, moon_az, use_jpl)
            if target != None:
                # Compute mean to apparent to observed ra and dec???
                airmass, tile_alt, tile_az = airMassCalculator(
                    target['RA'], target['DEC'], lst, return_altaz=True)
                exposure = expTimeEstimator(conditions, airmass, target['Program'], target['Ebmv'], target['DESsn2'], day_stats['MoonFrac'], target['MoonDist'], target['MoonAlt'])
                #exposure = target['maxLen']
                #print ('Estimated exposure = ', exposure, 'Maximum allowed exposure for tileID', target['tileID'], ' = ', target['maxLen'])
                if exposure <= 3.0 * target['maxLen']:
                    status, real_exposure, real_sn2 = observeField(target, exposure)
                    target['Status'] = status
                    target['Exposure'] = real_exposure
                    target['obsSN2'] = real_sn2
                    mjd += (setup_time + real_exposure)/86400.0
                    tilesObserved.add_row([target['tileID'], status])
                    slew = True
                    ra_prev = target['RA']
                    dec_prev = target['DEC']
                    if tableOutput:
                        t = Time(mjd, format = 'mjd')
                        tbase = str(t.isot)
                        obsList.append((target['tileID'],  target['RA'], target['DEC'], target['PASS'], target['Program'], target['Ebmv'],
                                       target['maxLen'], target['MoonFrac'], target['MoonDist'], target['MoonAlt'], conditions['Seeing'], conditions['Transparency'],
                                       airmass, target['DESsn2'], target['Status'],
                                       target['Exposure'], target['obsSN2'], tbase, mjd))
                    else:
                        # Output headers, but no data.
                        # In the future: GFAs (i, x, y + metadata for i=id, time, postagestampid) and fiber positions.
                        prihdr = pyfits.Header()
                        prihdr['TILEID  '] = target['tileID']
                        prihdr['RA      '] = target['RA']
                        prihdr['DEC     '] = target['DEC']
                        prihdr['PROGRAM '] = target['Program']
                        prihdr['EBMV    '] = target['Ebmv']
                        prihdr['MAXLEN  '] = target['maxLen']
                        prihdr['MOONFRAC'] = target['MoonFrac']
                        prihdr['MOONDIST'] = target['MoonDist']
                        prihdr['MOONALT '] = target['MoonAlt']
                        prihdr['SEEING  '] = conditions['Seeing']
                        prihdr['LINTRANS'] = conditions['Transparency']
                        prihdr['AIRMASS '] = airmass
                        prihdr['DESSN2  '] = target['DESsn2']
                        prihdr['STATUS  '] = target['Status']
                        prihdr['EXPTIME '] = target['Exposure']
                        prihdr['OBSSN2  '] = target['obsSN2']
                        t = Time(mjd, format = 'mjd')
                        tbase = str(t.isot)
                        nt = len(tbase)
                        prihdr['DATE-OBS'] = tbase
                        prihdr['MJD     '] = mjd
                        filename = day_stats['dirName'] + '/desi-exp-' + ocnt.update() + '.fits'
                        prihdu = pyfits.PrimaryHDU(header=prihdr)
                        prihdu.writeto(filename, clobber=True)
                else:
                    # Try another target?
                    # Observe longer split into modulo(max_len)
                    mjd += 0.25/24.0
                    slew = False # Can slew to new target while waiting.
            else:
                mjd += 0.25/24.0
                slew = False
            # Check time
            if mjd > day_stats['MJDsunrise']:
                nightOver = True

    if tableOutput and len(obsList) > 0:
        filename = 'obslist' + day_stats['dirName'].decode('ascii') + '.fits'
        cols = np.rec.array(obsList,
                           names = ('TILEID  ',
                                    'RA      ',
                                    'DEC     ',
                                    'PASS    ',
                                    'PROGRAM ',
                                    'EBMV    ',
                                    'MAXLEN  ',
                                    'MOONFRAC',
                                    'MOONDIST',
                                    'MOONALT ',
                                    'SEEING  ',
                                    'LINTRANS',
                                    'AIRMASS ',
                                    'DESSN2  ',
                                    'STATUS  ',
                                    'EXPTIME ',
                                    'OBSSN2  ',
                                    'DATE-OBS',
                                    'MJD     '),
                            formats = ['i4', 'f8', 'f8', 'i4', 'a8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'i4', 'f8', 'f8', 'a24', 'f8'])
        tbhdu = pyfits.BinTableHDU.from_columns(cols)
        tbhdu.writeto(filename, clobber=True)
        # This file is to facilitate plotting
        if os.path.exists('obslist_all.fits'):
            obsListOld = Table.read('obslist_all.fits', format='fits')
            obsListNew = Table.read(filename, format='fits')
            obsListAll = vstack([obsListOld, obsListNew])
            obsListAll.write('obslist_all.fits', format='fits', overwrite=True)
        else:
            copyfile(filename, 'obslist_all.fits')

    return tilesObserved
