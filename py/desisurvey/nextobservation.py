from __future__ import print_function, division

import numpy as np

import astropy.io.fits as pyfits
from astropy.time import Time
import astropy.coordinates
import astropy.units as u

from desitarget.targetmask import obsconditions as obsbits

import desiutil.log

from desisurvey.avoidobject import avoidObject, moonLoc
from desisurvey.utils import mjd2lst
import desisurvey.exposurecalc


MAX_AIRMASS = 2.0
MIN_MOON_SEP = 50.0
MIN_MOON_SEP_BGS = 50.0

LSTresSec = 600.0 # Also in afternoon planner and night obs.


def nextFieldSelector(obsplan, mjd, conditions, tilesObserved, slew,
                      previous_ra, previous_dec):
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

    Returns:
        target: dictionnary containing the following keys:
                'tileID', 'RA', 'DEC', 'Program', 'Ebmv', 'maxLen',
                'moon_illum_frac', 'MoonDist', 'MoonAlt', 'DESsn2', 'Status',
                'Exposure', 'obsSN2', 'obsConds'
        overhead: float (seconds)
    """
    log = desiutil.log.get_logger()

    hdulist = pyfits.open(obsplan)
    tiledata = hdulist[1].data
    moonfrac = hdulist[1].header['MOONFRAC']
    tileID = tiledata['TILEID']
    # Convert LST values from hours to degrees.
    tmin = tiledata['LSTMIN'] * 15
    tmax = tiledata['LSTMAX'] * 15
    explen = tiledata['EXPLEN']/240.0 # Need to call exposure time estimator instead
    ra = tiledata['RA']
    dec = tiledata['DEC']
    passnum = tiledata['PASS']
    program = tiledata['PROGRAM']
    obsconds = tiledata['OBSCONDITIONS']

    #- support tilesObserved as list or array or Table
    try:
        x = tilesObserved['TILEID']
        tilesObserved = x
    except (TypeError, KeyError, IndexError):
        pass

    lst = mjd2lst(mjd)
    dt = Time(mjd, format='mjd')
    found = False

    # Calculate the overhead times in seconds for each possible tile.
    proposed = astropy.coordinates.SkyCoord(
        ra=ra * u.deg, dec=dec * u.deg)
    if slew:
        previous = astropy.coordinates.SkyCoord(
            ra=previous_ra * u.deg, dec=previous_dec * u.deg)
    else:
        previous = None
    overheads = desisurvey.utils.get_overhead_time(
        previous, proposed).to(u.s).value

    for i in range(len(tileID)):

        overhead = overheads[i]

        # Estimate the exposure midpoint LST for this tile.
        lst_midpoint = lst + overhead / 240. + 0.5 * explen[i]
        if lst_midpoint >= 360:
            lst_midpoint -= 360
        # Select the first tile whose exposure midpoint falls within the
        # tile's LST window.
        if tmin[i] <= lst_midpoint and lst_midpoint <= tmax[i]:
            moondist, moonalt, moonaz = moonLoc(dt, ra[i], dec[i])
            if (obsconds[i] & obsbits.mask('BRIGHT')) == 0:
                min_moon_sep = MIN_MOON_SEP
            else:
                min_moon_sep = MIN_MOON_SEP_BGS
            if (avoidObject(dt, ra[i], dec[i]) and moondist > min_moon_sep):
                if ( (len(tilesObserved) > 0 and tileID[i] not in tilesObserved) or len(tilesObserved) == 0 ):
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
                  'moon_illum_frac': moonfrac, 'MoonDist': moondist, 'MoonAlt': moonalt, 'DESsn2': DESsn2, 'Status': status,
                  'Exposure': exposure, 'obsSN2': obsSN2, 'obsConds': obsconds[i]}
    else:
        target = None
    return target, overhead


def obsprio(priority, lst_assigned, lst):
    """Merit function for a tile given its priority and
    assigned LST.

    Args:
        priority (integer): priority (0-10) assigned by afternoon planner.
        lst_assigned (float): LST assigned by afternoon planner.
        lst (float): current LST
    Returns:
        float: merit function value
    """
    return (float(priority) -
            (lst_assigned-lst)*(lst_assigned-lst)/(LSTresSec*LSTresSec) )
