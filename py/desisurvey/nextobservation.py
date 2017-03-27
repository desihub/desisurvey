from __future__ import print_function, division
import numpy as np
import astropy.io.fits as pyfits
from astropy.time import Time
from desisurvey.utils import mjd2lst
from desitarget.targetmask import obsconditions as obsbits

MAX_AIRMASS = 2.0
MIN_MOON_SEP = 50.0
MIN_MOON_SEP_BGS = 50.0

def nextFieldSelector(obsplan, mjd, conditions, tilesObserved, slew, previous_ra, previous_dec, use_jpl=False):
    """
    Returns the first tile for which the current time falls inside
    its assigned LST window and is far enough from the Moon and
    planets.

    Args:
        obsplan: string, FITS file containing the afternoon plan
        mjd: float, current time
        conditions: dictionnary containing the weather info
        tilesObserved: list containing the tileID of all completed tiles
        slew: bool, True if a slew time needs to be taken into account
        previous_ra: float, ra of the previous observed tile (degrees)
        previous_dec: float, dec of the previous observed tile (degrees)
        use_jpl: bool, True if using jplephem and astropy instead of pyephem

    Returns:
        target: dictionnary containing the following keys:
                'tileID', 'RA', 'DEC', 'Program', 'Ebmv', 'maxLen',
                'MoonFrac', 'MoonDist', 'MoonAlt', 'DESsn2', 'Status',
                'Exposure', 'obsSN2', 'obsConds'
        overhead: float (seconds)
    """
    if (use_jpl):
        from desisurvey.avoidobjectJPL import avoidObject, moonLoc
    else:
        from desisurvey.avoidobject import avoidObject, moonLoc

    hdulist = pyfits.open(obsplan)
    tiledata = hdulist[1].data
    moonfrac = hdulist[1].header['MOONFRAC']
    tileID = tiledata['TILEID']
    tmin = tiledata['LSTMIN']
    tmax = tiledata['LSTMAX']
    explen = tiledata['EXPLEN']/240.0 # Need to call exposure time estimator instead
    ra = tiledata['RA']
    dec = tiledata['DEC']
    passnum = tiledata['PASS']
    program = tiledata['PROGRAM']
    obsconds = tiledata['OBSCONDITIONS']

    lst = mjd2lst(mjd)
    dt = Time(mjd, format='mjd')
    found = False
    for i in range(len(tileID)):
        dra = np.abs(ra[i]-previous_ra)
        if dra > 180.0:
            dra = 360.0 - dra
        ddec = np.abs(dec[i]-previous_dec)
        overhead = setup_time(slew, dra, ddec)
        t1 = tmin[i] + overhead/240.0
        t2 = tmax[i] - explen[i]
        if ( ((t1 <= t2) and (lst > t1 and lst < t2)) or ( (t2 < t1) and ((lst > t1 and t1 <=360.0) or (lst >= 0.0 and lst < t2))) ):
            moondist, moonalt, moonaz = moonLoc(dt, ra[i], dec[i])
            if (obsconds[i] & obsbits.mask('BRIGHT')) == 0:
                min_moon_sep = MIN_MOON_SEP
            else:
                min_moon_sep = MIN_MOON_SEP_BGS
            if (avoidObject(dt, ra[i], dec[i]) and moondist > min_moon_sep):
                if ( (len(tilesObserved) > 0 and tileID[i] not in tilesObserved['TILEID']) or len(tilesObserved) == 0 ):
                    found = True
                    break

    if found == True:
        tileID = tiledata['TILEID'][i]
        RA = ra[i]
        DEC = dec[i]
        PASSNUM = passnum[i]
        Ebmv = tiledata['EBV_MED'][i]
        maxLen = 2.0*tiledata['EXPLEN'][i]
        DESsn2 = 100.0 # Some made-up number -> has to be the same as the reference in exposurecalc.py
        status = tiledata['STATUS'][i]
        exposure = -1.0 # Updated after observation
        obsSN2 = -1.0   # Idem
        target = {'tileID' : tileID, 'RA' : RA, 'DEC' : DEC, 'PASS': PASSNUM,
                  'Program': program[i], 'Ebmv' : Ebmv, 'maxLen': maxLen,
                  'MoonFrac': moonfrac, 'MoonDist': moondist, 'MoonAlt': moonalt, 'DESsn2': DESsn2, 'Status': status,
                  'Exposure': exposure, 'obsSN2': obsSN2, 'obsConds': obsconds[i]}
    else:
        target = None
    return target, overhead

def setup_time(slew, dra, ddec):
    """
    Computes setup time: slew and focus (assumes readout can proceed during
    slew.

    Args:
        slew: bool, True if slew time needs to be taken into account
        dra: float, difference in RA between previous and current tile (degrees)
        ddec: float, difference in DEC between previous and current tile (degrees)

    Returns:
        float, total setup time (seconds)
    """

    focus_time = 30.0
    slew_time = 0.0
    if slew:
        d = np.maximum(dra, ddec)
        slew_time = 11.5 + d/0.45
    overhead = focus_time + slew_time
    if overhead < 120.0:
        overhead = 120.0
    return overhead
