"""Simple forecast of survey progress and margin.
"""
from __future__ import print_function, division, absolute_import

import numpy as np
import pandas as pd

import astropy.table

import desimodel.io

import desisurvey.config
import desisurvey.ephemerides
import desisurvey.etc
import desisurvey.utils


class Forecast(object):
    """Compute a simple forecast of survey progress and margin.

    Based on config, ephemerides, tiles.
    """
    def __init__(self, use_twilight=False, tiles_file=None,
        nominal={'DARK': 1000., 'GRAY': 1000., 'BRIGHT': 300.}):
        self.use_twilight = use_twilight
        # Look up the tiles to observe.
        tiles = desisurvey.tiles.get_tiles(tiles_file)
        self.tiles = tiles
        # Load our configuration.
        config = desisurvey.config.Configuration()
        # Save design hour angles in degrees.
        surveyinit_t = astropy.table.Table.read(
            config.get_path('surveyinit.fits'))
        self.design_hour_angle = surveyinit_t['HA'].data.copy()
        # Compute airmass at design hour angles.
        self.airmass = tiles.airmass(self.design_hour_angle)
        airmass_factor = desisurvey.etc.airmass_exposure_factor(self.airmass)
        # Load ephemerides.
        ephem = desisurvey.ephemerides.Ephemerides()
        self.num_nights = ephem.num_nights
        # Compute the expected available hours per program,
        # with and without weather.
        scheduled = desisurvey.ephemerides.get_program_hours(
            ephem, apply_weather=False, include_twilight=use_twilight)
        available = desisurvey.ephemerides.get_program_hours(
            ephem, apply_weather=True, include_twilight=use_twilight)
        self.cummulative_days = np.cumsum(available, axis=1) / 24.
        # Calculate program parameters.
        ntiles, tsched, openfrac, dust, airmass = [], [], [], [], []
        for program in tiles.programs:
            tile_sel = tiles.program_mask[program]
            ntiles.append(np.count_nonzero(tile_sel))
            progindx = ephem.program_index[program]
            scheduled_sum = scheduled[progindx].sum()
            tsched.append(scheduled_sum)
            openfrac.append(available[progindx].sum() / scheduled_sum)
            dust.append(tiles.dust_factor[tile_sel].mean())
            airmass.append(airmass_factor[tile_sel].mean())
        # Build a table of all forecasting parameters.
        df = pd.DataFrame()
        self.df = df
        df['Number of tiles'] = ntiles
        df['Scheduled time (hr)'] = tsched
        df['Dome open fraction'] = openfrac
        self.set_overheads()
        df['Nominal exposure (s)'] = [nominal[p] for p in self.tiles.programs]
        df['Dust factor'] = dust
        df['Airmass factor'] = airmass
        self.set_factors()

    def summary(self):
        """Print a summary table of the forecast parameters.
        """
        df = self.df.transpose()
        df.rename({pidx: pname for pidx, pname in enumerate(self.tiles.programs)},
                  inplace=True, axis='columns')
        return df

    def set_overheads(self, update_margin=True,
                      setup={'DARK': 200, 'GRAY': 200, 'BRIGHT': 150},
                      split={'DARK': 100, 'GRAY': 100, 'BRIGHT':  75},
                      dead ={'DARK':  20, 'GRAY': 100, 'BRIGHT':  10}):
        df = self.df
        df['Setup overhead / tile (s)'] = [setup[p] for p in self.tiles.programs]
        df['Cosmic split overhead / tile (s)'] = [split[p] for p in self.tiles.programs]
        df['Operations overhead / tile (s)'] = [dead[p] for p in self.tiles.programs]
        df['Average available / tile (s)'] = (
            df['Scheduled time (hr)'] * df['Dome open fraction'] /
            df['Number of tiles'] * 3600 -
            df['Setup overhead / tile (s)'] -
            df['Cosmic split overhead / tile (s)'] -
            df['Operations overhead / tile (s)'])
        self.update()

    def set_factors(self, update_margin=True,
                       moon    = {'DARK': 1.00, 'GRAY': 1.10, 'BRIGHT': 1.33},
                       weather = {'DARK': 1.22, 'GRAY': 1.20, 'BRIGHT': 1.16}):
        df = self.df
        df['Moon factor'] = [moon[p] for p in self.tiles.programs]
        df['Weather factor'] = [weather[p] for p in self.tiles.programs]
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
