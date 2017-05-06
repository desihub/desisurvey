"""Select the next tile to observe during a night.
"""
from __future__ import print_function, division

import numpy as np

import astropy.io.fits
import astropy.time
import astropy.coordinates
import astropy.units as u

import desitarget.targetmask

import desiutil.log

import desisurvey.avoidobject
import desisurvey.utils
import desisurvey.config


def nextFieldSelector(obsplan, mjd, conditions, progress, slew,
                      previous_ra, previous_dec):
    """
    Returns the first tile for which the current time falls inside
    its assigned LST window and is far enough from the Moon and
    planets.

    Args:
        obsplan: string, FITS file containing the afternoon plan
        mjd: float, current time
        conditions: current weather conditions (not being used)
        progress: table of observations made so far
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
    config = desisurvey.config.Configuration()

    hdulist = astropy.io.fits.open(obsplan)
    tiledata = hdulist[1].data
    moonfrac = hdulist[1].header['MOONFRAC']
    tileID = tiledata['TILEID']
    # Convert LST values from hours to degrees.
    tmin = tiledata['LSTMIN'] * 15
    tmax = tiledata['LSTMAX'] * 15
    # Need to call exposure time estimator instead
    explen = tiledata['EXPLEN']/240.0
    ra = tiledata['RA']
    dec = tiledata['DEC']
    passnum = tiledata['PASS']
    program = tiledata['PROGRAM']
    obsconds = tiledata['OBSCONDITIONS']
    bright_mask = desitarget.targetmask.obsconditions.mask('BRIGHT')

    lst = desisurvey.utils.mjd2lst(mjd)
    dt = astropy.time.Time(mjd, format='mjd')
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
            # Calculate the moon separation and altitude angles.
            moondist, moonalt, _ = desisurvey.avoidobject.moonLoc(
                dt, ra[i], dec[i])
            # Check that this observation is not too close to the moon
            # or any planets. Should we skip the moon separation
            # test when the moon is below the horizon?
            if (desisurvey.avoidobject.avoidObject(dt, ra[i], dec[i]) and
                moondist > config.avoid_bodies.moon().to(u.deg).value):
                # Check that this tile still needs more exposure.
                if progress.get_tile(tileID[i])['status'] < 2:
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
                  'moon_illum_frac': moonfrac, 'MoonDist': moondist,
                  'MoonAlt': moonalt, 'DESsn2': DESsn2, 'Status': status,
                  'Exposure': exposure, 'obsSN2': obsSN2,
                  'obsConds': obsconds[i]}
    else:
        target = None
    return target, overhead
