"""Tabulate sun and moon ephemerides on each night of the survey calendar.
"""
from __future__ import print_function, division
from astropy.time import Time
import astropy.table
from datetime import datetime, timedelta
import numpy as np
import ephem
import desisurvey.kpno as kpno
import warnings
import math
import os.path


def getCalAll(startdate, enddate, num_moon_steps=32,
              verbose=True, use_cache=True):
    """Computes the nightly sun and moon ephemerides for the date range given.

    Args:
        startdate: datetime object for the beginning
        enddate: same, but for the end.
        num_moon_steps: number of steps for calculating moon altitude
            during the night, when it is up.
        verbose: print information to stdout.
        use_cache: use a previously cached table if available.
    Returns:
        Astropy table of sun and moon ephemerides.
    """
    # Build filename for saving the ephemerides.
    mjd_start = int(math.floor(Time(startdate).mjd))
    mjd_stop = int(math.ceil(Time(enddate).mjd))
    filename = 'ephem_{0}_{1}.fits'.format(mjd_start, mjd_stop)
    if use_cache and os.path.exists(filename):
        if verbose:
            print('Loading cached ephemerides from {0}'.format(filename))
        return astropy.table.Table.read(filename)

    # Allocate space for the data we will calculate.
    num_days = (enddate - startdate).days + 1
    data = np.empty(num_days, dtype=[
        ('MJDsunset', float), ('MJDsunrise', float), ('MJDetwi', float),
        ('MJDmtwi', float), ('MJDe13twi', float), ('MJDm13twi', float),
        ('MJDmoonrise', float), ('MJDmoonset', float), ('MoonFrac', float),
        ('MoonNightStart', float), ('MoonNightStop', float),
        ('MoonAlt', float, num_moon_steps), ('dirName', 'S8')])

    # Initialize the observer.
    mayall = ephem.Observer()
    mayall.lat = np.radians(kpno.mayall.lat_deg)
    mayall.lon = np.radians(kpno.mayall.west_lon_deg)
    # Disable atmospheric refraction corrections.
    mayall.pressure = 0.0
    # This throws a warning because of the early year, but it is harmless.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=astropy.utils.exceptions.AstropyUserWarning)
        mjd0 = Time(datetime(1900,1,1,12,0,0)).mjd

    # Loop over days.
    for day_offset in range(num_days):
        day = startdate + timedelta(days=day_offset)
        mayall.date = day
        row = data[day_offset]
        # Calculate sun rise/set with different horizons.
        mayall.horizon = '-0:34' # the value that the USNO uses.
        row['MJDsunset'] = mayall.next_setting(ephem.Sun()) + mjd0
        row['MJDsunrise'] = mayall.next_rising(ephem.Sun()) + mjd0
        # 13 deg twilight, adequate (?) for BGS sample.
        mayall.horizon = '-13'
        row['MJDe13twi'] = mayall.next_setting(ephem.Sun(), use_center=True) + mjd0
        row['MJDm13twi'] = mayall.next_rising(ephem.Sun(), use_center=True) + mjd0
        # 15 deg twilight, start of dark time if the moon is down.
        mayall.horizon = '-15'
        row['MJDetwi'] = mayall.next_setting(ephem.Sun(), use_center=True) + mjd0
        row['MJDmtwi'] = mayall.next_rising(ephem.Sun(), use_center=True) + mjd0
        # Moon.
        mayall.horizon = '-0:34' # the value that the USNO uses.
        row['MJDmoonrise'] = mayall.next_rising(ephem.Moon()) + mjd0
        if row['MJDmoonrise'] > row['MJDsunrise']:
            row['MJDmoonrise'] = mayall.previous_rising(ephem.Moon()) + mjd0
        mayall.date = row['MJDmoonrise'] - mjd0
        row['MJDmoonset'] = mayall.next_setting(ephem.Moon()) + mjd0
        # Calculate moon phase at the midpoint between moon rise and set.
        m0 = ephem.Moon()
        m0.compute(0.5 * (row['MJDmoonrise'] + row['MJDmoonset']) - mjd0)
        # Calculate the fraction of the moon's surface that is illuminated.
        row['MoonFrac'] = m0.moon_phase
        # Determine when the moon is up while the sun is down during this
        # night, if at all.
        row['MoonNightStart'] = max(row['MJDmoonrise'], row['MJDsunset'])
        row['MoonNightStop'] = min(max(row['MJDmoonset'], row['MJDsunset']),
                                   row['MJDsunrise'])
        # Tabulate the moon altitude at num_moon_steps equally spaced times
        # covering this interval.
        t_moon = np.linspace(row['MoonNightStart'], row['MoonNightStop'],
                             num_moon_steps)
        moon_alt = row['MoonAlt']
        for i, t in enumerate(t_moon):
            mayall.date = ephem.Date(t - mjd0)
            m0.compute(mayall)
            moon_alt[i] = math.degrees(float(m0.alt))
        # Build the night's directory name.
        row['dirName'] = ('{y:04d}{m:02d}{d:02d}'
                          .format(y=day.year, m=day.month, d=day.day))

    t = astropy.table.Table(data, meta=dict(name='Survey Ephemerides'))
    if verbose:
        print('Saving ephemerides to {0}'.format(filename))
    t.write(filename, overwrite=True)
    return t
