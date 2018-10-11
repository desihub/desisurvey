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

    def __init__(self, use_twilight=False,
        nominal={'DARK': 1000., 'GRAY': 1000., 'BRIGHT': 300.}):
        # Load our configuration.
        config = desisurvey.config.Configuration()
        valid_programs = list(config.programs.keys)
        # Read the tiles to forecast.
        tiles = desimodel.io.load_tiles(
            onlydesi=True, extra=False, tilesfile=config.tiles_file())
        tile_programs = np.unique(tiles['PROGRAM'])
        self.programs = [p for p in valid_programs if p in tile_programs]
        passnum = tiles['PASS']
        self.num_passes = len(np.unique(passnum))
        self.ntiles_per_pass = [
            np.count_nonzero(passnum == p) for p in range(self.num_passes)]
        self.program_passes = {
            p: np.unique(passnum[tiles['PROGRAM'] == p]) for p in tile_programs}
        dust_factor = desisurvey.etc.dust_exposure_factor(tiles['EBV_MED'])
        tile_dec = np.radians(tiles['DEC'])
        # Load design hour angles.
        surveyinit_t = astropy.table.Table.read(
            config.get_path('surveyinit.fits'))
        self.design_hour_angle = np.radians(surveyinit_t['HA'].data)
        # Compute airmass at design hour angles.
        latitude = np.radians(config.location.latitude())
        coef_A = np.sin(tile_dec) * np.sin(latitude)
        coef_B = np.cos(tile_dec) * np.cos(latitude)
        cosZ = coef_A + coef_B * np.cos(self.design_hour_angle)
        self.airmass = desisurvey.utils.cos_zenith_to_airmass(cosZ)
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
        for program in self.programs:
            tile_sel = tiles['PROGRAM'] == program
            ntiles.append(np.count_nonzero(tile_sel))
            pidx = desisurvey.ephemerides.program_name.index(program) - 1
            tsched.append(scheduled[pidx].sum())
            openfrac.append(available[pidx].sum() / scheduled[pidx].sum())
            dust.append(dust_factor[tile_sel].mean())
            airmass.append(airmass_factor[tile_sel].mean())
        # Build a table of all forecasting parameters.
        df = pd.DataFrame()
        self.df = df
        df['Number of tiles'] = ntiles
        print(df['Number of tiles'].dtype)
        df['Scheduled time (hr)'] = tsched
        df['Dome open fraction'] = openfrac
        self.set_overheads()
        df['Nominal exposure (s)'] = [nominal[p] for p in self.programs]
        df['Dust factor'] = dust
        df['Airmass factor'] = airmass
        self.set_factors()

    def summary(self):
        """Print a summary table of the forecast parameters.
        """
        df = self.df.transpose()
        df.rename({pidx: pname for pidx, pname in enumerate(self.programs)},
                  inplace=True, axis='columns')
        return df

    def set_overheads(self, update_margin=True,
                         setup={'DARK': 200, 'GRAY': 200, 'BRIGHT': 150},
                         split={'DARK': 100, 'GRAY': 100, 'BRIGHT':  75},
                         dead ={'DARK':  20, 'GRAY': 100, 'BRIGHT':  10},
                         ):
        df = self.df
        df['Setup overhead / tile (s)'] = [setup[p] for p in self.programs]
        df['Cosmic split overhead / tile (s)'] = [split[p] for p in self.programs]
        df['Operations overhead / tile (s)'] = [dead[p] for p in self.programs]
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
        df['Moon factor'] = [moon[p] for p in self.programs]
        df['Weather factor'] = [weather[p] for p in self.programs]
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
        self.pass_progress = np.zeros((self.num_passes, self.num_nights))
        for program in self.programs:
            pidx = desisurvey.ephemerides.program_name.index(program) - 1
            dtexp = (
                df['Average required / tile (s)'] +
                df['Setup overhead / tile (s)'] +
                df['Cosmic split overhead / tile (s)'] +
                df['Operations overhead / tile (s)']
                )[self.programs.index(program)] / 86400.
            # Calculate the mean time between exposures for this program.
            progress = self.cummulative_days[pidx] / dtexp
            # Compute progress assuming tiles are observed in pass order,
            # separated by exactly dtexp.
            ntiles_observed = 0
            for passnum in self.program_passes[program]:
                ntiles = self.ntiles_per_pass[passnum]
                self.pass_progress[passnum] = np.clip(
                    progress - ntiles_observed, 0, ntiles)
                ntiles_observed += ntiles