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
    max_moondist = config.avoid_bodies.moon().to(u.deg).value
    max_zenith_angle = 90 * u.deg - config.min_altitude()

    hdulist = astropy.io.fits.open(obsplan)
    tiledata = hdulist[1].data
    moonfrac = hdulist[1].header['MOONFRAC']
    tileID = tiledata['TILEID']
    # Convert LST values from hours to degrees.
    tmin = tiledata['LSTMIN'] * 15
    tmax = tiledata['LSTMAX'] * 15
    # Need to call exposure time estimator instead
    explen = tiledata['EXPLEN']/240.0
    passnum = tiledata['PASS']
    program = tiledata['PROGRAM']
    obsconds = tiledata['OBSCONDITIONS']
    bright_mask = desitarget.targetmask.obsconditions.mask('BRIGHT')

    # Convert the current MJD to LST and an astropy time.
    lst = desisurvey.utils.mjd2lst(mjd)
    dt = astropy.time.Time(mjd, format='mjd')

    # Initialize pointings for each possible tile.
    ra = tiledata['RA']
    dec = tiledata['DEC']
    proposed = astropy.coordinates.SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    # Calculate current zenith angle for each possible tile.
    zenith = desisurvey.utils.get_observer(
        dt, alt=90 * u.deg, az=0 * u.deg).transform_to(astropy.coordinates.ICRS)
    zenith_angles = proposed.separation(zenith)

    # Calculate the overhead times in seconds for each possible tile.
    if slew:
        previous = astropy.coordinates.SkyCoord(
            ra=previous_ra * u.deg, dec=previous_dec * u.deg)
    else:
        previous = None
    overheads = desisurvey.utils.get_overhead_time(
        previous, proposed).to(u.s).value

    found = False
    for i in range(len(tileID)):
        # Estimate this tile's exposure midpoint LST in the range [0,360] deg.
        overhead = overheads[i]
        lst_midpoint = lst + overhead / 240. + 0.5 * explen[i]
        if lst_midpoint >= 360:
            lst_midpoint -= 360
        # Skip tiles whose exposure midpoint falls outside their LST window.
        if lst_midpoint < tmin[i] or lst_midpoint > tmax[i]:
            continue
        # Skip a tile that is currently too close to the horizon.
        if zenith_angles[i] > max_zenith_angle:
            log.info(
                'Tile {0} is too close to the horizon ({1:.f})'
                .format(tileID[i], 90 * u.deg - zenith_angles[i]))
            continue
        # Calculate the moon separation and altitude angles.
        moondist, moonalt, _ = desisurvey.avoidobject.moonLoc(dt, ra[i], dec[i])
        # Skip a tile that is currently too close to a moon above the horizon.
        if moonalt > 0 and moondist <= max_moondist:
            log.info(
                'Tile {0} is too close to moon with alt={1:.1f}, sep={2:.1f}.'
                .format(tileID[i], moonalt, moondist))
            continue
        # Skip a tile that is currently too close to any planets.
        if not desisurvey.avoidobject.avoidObject(dt, ra[i], dec[i]):
            log.info('Tile {0} is too close to a planet.'.format(tileID[i]))
            continue
        # Does this tile still needs more exposure?
        if progress.get_tile(tileID[i])['status'] < 2:
            found = True
            break

    if found:
        TILEID = tileID[i]
        RA = ra[i]
        DEC = dec[i]
        PASSNUM = passnum[i]
        Ebmv = tiledata['EBV_MED'][i]
        maxLen = 2.0*tiledata['EXPLEN'][i]
        DESsn2 = 100.0 # Some made-up number -> has to be the same as the reference in exposurecalc.py
        status = tiledata['STATUS'][i]
        exposure = -1.0 # Updated after observation
        obsSN2 = -1.0   # Idem
        target = {'tileID' : TILEID, 'RA' : RA, 'DEC' : DEC, 'PASS': PASSNUM,
                  'Program': program[i], 'Ebmv' : Ebmv, 'maxLen': maxLen,
                  'moon_illum_frac': moonfrac, 'MoonDist': moondist,
                  'MoonAlt': moonalt, 'DESsn2': DESsn2, 'Status': status,
                  'Exposure': exposure, 'obsSN2': obsSN2,
                  'obsConds': obsconds[i]}
    else:
        target = None

    return target, overhead
