from __future__ import print_function, division
from astropy.time import Time
import astropy.table
from datetime import datetime, timedelta
import numpy as np
import ephem
import desisurvey.kpno as kpno
import warnings


def getCalAll(startdate, enddate):
    """Computes the nightly calendar for the date
       range given.

    Args:
        startdate: datetime object for the beginning
        enddate: same, but for the end.
    Returns:
        Astropy table of sun and moon ephemerides.
    """
    # Allocate space for the data we will calculate.
    num_days = (enddate - startdate).days + 1
    data = np.empty(num_days, dtype=[
        ('MJDsunset', float), ('MJDsunrise', float), ('MJDetwi', float),
        ('MJDmtwi', float), ('MJDe13twi', float), ('MJDm13twi', float),
        ('MJDmoonrise', float), ('MJDmoonset', float), ('MoonFrac', float),
        ('dirName', 'S8'), ('MJD_bright_start', float),
        ('MJD_bright_end', float)])

    # Initialize the observer.
    mayall = ephem.Observer()
    mayall.lat, mayall.lon = np.radians(kpno.mayall.lat_deg), np.radians(kpno.mayall.west_lon_deg)
    # Disable atmospheric refraction corrections.
    mayall.pressure = 0.0
    # This throws a warning because of the early year, but it is harmless.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=astropy.utils.exceptions.AstropyUserWarning)
        mjd0 = Time(datetime(1900,1,1,12,0,0)).mjd

    # Loop over days.
    for day_offset in range(num_days):
        day = startdate + timedelta(days=day_offset)
        mayall.date = day
        row = data[day_offset]
        # Sun
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
        row['MJDmoonset'] = mayall.next_setting(ephem.Moon()) + mjd0
        if row['MJDmoonrise'] > row['MJDsunrise']:
           row['MJDmoonrise'] = mayall.previous_rising(ephem.Moon()) + mjd0
        m0 = ephem.Moon()
        m0.compute(day)
        # Fraction of surface that is illuminated.
        row['MoonFrac'] = m0.moon_phase
        # Determine the start and end of bright time for this night.
        MJD_bright_start = row['MJDsunrise']
        MJD_bright_end = row['MJDsunset']
        if (row['MoonFrac'] > 0.6):
            if row['MJDmoonrise'] < row['MJDe13twi']:
                MJD_bright_start = row['MJDe13twi']
            else:
                MJD_bright_start = row['MJDmoonrise']
            if (row['MJDmoonset'] > row['MJDm13twi']):
                MJD_end_time = row['MJDm13twi']
            else:
                MJD_bright_end = row['MJDmoonset']
        else:
            # Calculate moon altitude every minute to find start / end times.
            t = row['MJDmoonrise'] - mjd0
            mayall.date = ephem.Date(t)
            m0.compute(mayall)
            moonalt = m0.alt
            while (moonalt < 30.0/row['MoonFrac'] and t < row['MJDmoonset'] - mjd0):
                # Increment by one minute.
                t += 1.0 / 1440.0
                mayall.date = ephem.Date(t)
                m0.compute(mayall)
                moonalt = m0.alt
            if t < row['MJDmoonset'] - mjd0:
                MJD_bright_start = t + mjd0
                while (moonalt >= 30.0/row['MoonFrac'] and t < row['MJDmoonset'] - mjd0):
                    # Increment by one minute.
                    t += 1.0/1440.0
                    mayall.date = ephem.Date(t)
                    m0.compute(mayall)
                    moonalt = m0.alt
                if t < row['MJDmoonset'] - mjd0:
                    MJD_bright_end = t + mjd0
                else:
                    MJD_bright_end = row['MJDmoonset']
        row['MJD_bright_start'] = MJD_bright_start
        row['MJD_bright_end'] = MJD_bright_end
        # Build the night's directory name.
        row['dirName'] = '{y:04d}{m:02d}{d:02d}'.format(y=day.year, m=day.month, d=day.day)

    return astropy.table.Table(data, meta=dict(name='Survey Ephemerides'))
