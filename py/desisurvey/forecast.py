"""Simple forecast of survey progress and margin.
"""
from __future__ import print_function, division, absolute_import

import collections

import numpy as np

import astropy.table
import astropy.units as u

import desimodel.io

import desisurvey.config
import desisurvey.ephem
import desisurvey.etc
import desisurvey.utils


class Forecast(object):
    """Compute a simple forecast of survey progress and margin.

    Based on config, ephemerides, tiles.

    Parameters
    ----------
    start_date : datetime.date
        Forecast for survey that starts on the evening of this date.
    stop_date : datetime.date
        Forecast for survey that stops on the morning of this date.
    use_twilight : bool
        Include twilight time in the forecast scheduled time?
    weather : array or None
        1D array of nightly weather factors (0-1) to use, or None to use
        :func:`desisurvey.plan.load_weather`. The array length must equal
        the number of nights in [start,stop). Values are fraction of the
        night with the dome open (0=never, 1=always). Use
        1 - :func:`desimodel.weather.dome_closed_fractions` to lookup
        suitable corrections based on historical weather data.
    design_hourangle : array or None
        1D array of design hour angles to use in degrees, or None to use
        :func:`desisurvey.plan.load_design_hourangle`.
    """
    def __init__(self, start_date=None, stop_date=None, use_twilight=False,
                 weather=None, design_hourangle=None):
        config = desisurvey.config.Configuration()
        if start_date is None:
            start_date = config.first_day()
        else:
            start_date = desisurvey.utils.get_date(start_date)
        if stop_date is None:
            stop_date = config.last_day()
        else:
            stop_date = desisurvey.utils.get_date(stop_date)
        self.num_nights = (stop_date - start_date).days
        if self.num_nights <= 0:
            raise ValueError('Expected start_date < stop_date.')

        self.use_twilight = use_twilight
        # Look up the tiles to observe.
        tiles = desisurvey.tiles.get_tiles()
        self.tiles = tiles
        if design_hourangle is None:
            self.design_hourangle = np.zeros(tiles.ntiles)
        else:
            if len(design_hourangle) != tiles.ntiles:
                raise ValueError('Array design_hourangle has wrong length.')
            self.design_hourangle = np.asarray(design_hourangle)
        # Get weather factors.
        if weather is None:
            self.weather = desisurvey.plan.load_weather(start_date, stop_date)
        else:
            self.weather = np.asarray(weather)
        if self.weather.shape != (self.num_nights,):
            raise ValueError('Array weather has wrong shape.')
        # Get the design hour angles.
        if design_hourangle is None:
            self.design_hourangle = desisurvey.plan.load_design_hourangle()
        else:
            self.design_hourangle = np.asarray(design_hourangle)
        if self.design_hourangle.shape != (tiles.ntiles,):
            raise ValueError('Array design_hourangle has wrong shape.')
        # Compute airmass at design hour angles.
        self.airmass = tiles.airmass(self.design_hourangle)
        airmass_factor = desisurvey.etc.airmass_exposure_factor(self.airmass)
        # Load ephemerides.
        ephem = desisurvey.ephem.get_ephem()
        # Compute the expected available and scheduled hours per program.
        scheduled = ephem.get_program_hours(include_twilight=use_twilight)
        available = scheduled * self.weather
        self.cummulative_days = np.cumsum(available, axis=1) / 24.
        # Calculate program parameters.
        ntiles, tsched, openfrac, dust, airmass, nominal = [], [], [], [], [], []
        for program in tiles.programs:
            tile_sel = tiles.program_mask[program]
            ntiles.append(np.count_nonzero(tile_sel))
            progindx = tiles.program_index[program]
            scheduled_sum = scheduled[progindx].sum()
            tsched.append(scheduled_sum)
            openfrac.append(available[progindx].sum() / scheduled_sum)
            dust.append(tiles.dust_factor[tile_sel].mean())
            airmass.append(airmass_factor[tile_sel].mean())
            nominal.append(getattr(config.nominal_exposure_time, program)().to(u.s).value)
        # Build a table of all forecasting parameters.
        df = collections.OrderedDict()
        self.df = df
        df['Number of tiles'] = np.array(ntiles)
        df['Scheduled time (hr)'] = np.array(tsched)
        df['Dome open fraction'] = np.array(openfrac)
        self.set_overheads()
        df['Nominal exposure (s)'] = np.array(nominal)
        df['Dust factor'] = np.array(dust)
        df['Airmass factor'] = np.array(airmass)
        self.set_factors()

    def summary(self, width=7, prec=5, separator='-'):
        """Print a summary table of the forecast parameters.
        """
        # Find the longest key and calculate the row length.
        nprog = len(self.tiles.programs)
        maxlen = np.max([len(key) for key in self.df])
        rowlen = maxlen + (1 + width) * nprog
        # Build a format string for each row.
        header = ' ' * maxlen + ' {{:>{}s}}'.format(width) * nprog
        row = '{{:>{}s}}'.format(maxlen) + ' {{:{}.{}g}}'.format(width, prec) * nprog
        # Print the header.
        print(header.format(*self.tiles.programs))
        print(separator * rowlen)
        # Print each row.
        for key, values in self.df.items():
            print(row.format(key, *values))
        print(separator * rowlen)

    def set_overheads(self, update_margin=True,
                      setup={'DARK': 200, 'GRAY': 200, 'BRIGHT': 150},
                      split={'DARK': 100, 'GRAY': 100, 'BRIGHT':  75},
                      dead ={'DARK':  20, 'GRAY': 100, 'BRIGHT':  10}):
        df = self.df
        df['Setup overhead / tile (s)'] = np.array([setup[p] for p in self.tiles.programs])
        df['Cosmic split overhead / tile (s)'] = np.array([split[p] for p in self.tiles.programs])
        df['Operations overhead / tile (s)'] = np.array([dead[p] for p in self.tiles.programs])
        df['Average available / tile (s)'] = (
            df['Scheduled time (hr)'] * df['Dome open fraction'] /
            # Avoid division by zero for a program with no tiles.
            np.maximum(1, df['Number of tiles']) * 3600 -
            df['Setup overhead / tile (s)'] -
            df['Cosmic split overhead / tile (s)'] -
            df['Operations overhead / tile (s)'])
        self.update()

    def set_factors(self, update_margin=True,
                       moon    = {'DARK': 1.00, 'GRAY': 1.10, 'BRIGHT': 1.33},
                       weather = {'DARK': 1.22, 'GRAY': 1.20, 'BRIGHT': 1.16}):
        df = self.df
        df['Moon factor'] = np.array([moon[p] for p in self.tiles.programs])
        df['Weather factor'] = np.array([weather[p] for p in self.tiles.programs])
        df['Average required / tile (s)'] = (
            df['Nominal exposure (s)'] *
            df['Dust factor'] *
            df['Airmass factor'] *
            df['Moon factor'] *
            df['Weather factor'])
        self.update()

    def update(self):
        df = self.df
        if 'Average available / tile (s)' not in df: return
        if 'Average required / tile (s)' not in df: return
        df['Exposure time margin (%)'] = 100 * (
            df['Average available / tile (s)'] /
            df['Average required / tile (s)'] - 1)
        self.pass_progress = np.zeros((self.tiles.npasses, self.num_nights))
        for program in self.tiles.programs:
            progidx = self.tiles.program_index[program]
            dtexp = (
                df['Average required / tile (s)'] +
                df['Setup overhead / tile (s)'] +
                df['Cosmic split overhead / tile (s)'] +
                df['Operations overhead / tile (s)']
                )[progidx] / 86400.
            # Calculate the mean time between exposures for this program.
            progress = self.cummulative_days[progidx] / dtexp
            # Compute progress assuming tiles are observed in pass order,
            # separated by exactly dtexp.
            ntiles_observed = 0
            for passnum in self.tiles.program_passes[program]:
                passidx = self.tiles.pass_index[passnum]
                ntiles = self.tiles.pass_ntiles[passnum]
                self.pass_progress[passidx] = np.clip(
                    progress - ntiles_observed, 0, ntiles)
                ntiles_observed += ntiles
