from __future__ import print_function, division

import copy
import os
import sys
import pkg_resources

import numpy as np

import astropy.table

import desimodel.io
import desiutil.log

import desisurvey.config
from desisurvey.utils import mjd2lst


SURVEY_NOMINAL_LENGTH = 5.0 * 365.25 #Survey duration

class surveyPlan:
    """
    Main class for survey planning
    """

    def __init__(self, MJDstart, MJDend, ephem, tilesubset=None, HA_assign=False):
        """Initialises survey by reading in the file desi_tiles.fits
        and populates the class members.

        Arguments:
            MJDstart: day of the (re-)start of the survey
            MJDend: day of the end of the survey
            ephem: Ephemerides covering MJDstart - MJDend

        Optional:
            tilesubset: array of integer tileids to use; ignore others
        """
        self.log = desiutil.log.get_logger()
        self.config = desisurvey.config.Configuration()
        self.ephem = ephem

        # Read in DESI tile data
        tiles = astropy.table.Table(
            desimodel.io.load_tiles(onlydesi=True, extra=False))

        # Restrict to a subset of tiles if requested.
        if tilesubset is not None:
            tiles = tiles[tilesubset]

        numtiles = len(tiles)

        # Drop un-needed columns.
        tiles.remove_columns(['IN_DESI', 'AIRMASS', 'STAR_DENSITY', 'EXPOSEFAC'])

        # Add some new columns (more will be added later).
        for name, dtype in (('GAL_CAP', np.int8), ('SUBLIST', np.int8),
                            ('PRIORITY', np.int32), ('STATUS', np.int32)):
            tiles[name] = np.zeros(numtiles, dtype=dtype)

        # Determine which galactic cap each tile is in: -1=south, +1=north.
        tiles['GAL_CAP'] = -1
        tiles['GAL_CAP'][(tiles['RA'] > 75.0) & (tiles['RA'] < 300.0)] = +1

        # Assign a sublist to each tile equal to pass for tiles in the
        # first-year full depth field, or else equal to pass+8.  The full
        # depth field covers 15 deg <= dec <= 25 deg in the NGC,
        # padded by 3 deg for the first pass in each program.
        dec = tiles['DEC']
        passnum = tiles['PASS']
        first_pass = (passnum == 0) | (passnum == 4) | (passnum == 5)
        dec_min = np.full(numtiles, 15.0)
        dec_max = np.full(numtiles, 25.0)
        dec_min[first_pass] -= 3.0
        dec_max[first_pass] += 3.0
        tiles['SUBLIST'] = passnum
        tiles['SUBLIST'][
            (tiles['GAL_CAP'] < 0) | (dec < dec_min) | (dec > dec_max)] += 8

        # Initialize the LST bins we will use for scheduling each night.
        self.nLST = 144
        self.LSTedges = np.linspace(0.0, 360.0, self.nLST + 1)
        self.LSTbins = 0.5 * (self.LSTedges[1:] + self.LSTedges[:-1])
        self.LSTres = self.LSTedges[1]

        self.tiles = tiles
        self.numtiles = numtiles

        # Add HA, LSTMIN, LSTMAX columns.
        self.assignHA(MJDstart, MJDend, compute=HA_assign)

        self.tiles.sort(('SUBLIST', 'DEC'))


    def assignHA(self, MJDstart, MJDend, compute=False):
        """Assigns optimal hour angles for the DESI tiles;
        can be re-run at any point during the survey to
        reoptimise the schedule.

        Args:
            MJDstart: float, time at which the assignment starts, this is the same input as
                      for surveySim.
            MJDend: float, time by which the _survey_ is expected to be completed, i.e. it should
                    be the MJDstart + time remaining in survey.

        Optional:
            compute: bool, False reads a pre-computed table; for development purposes only.
        """

        if compute:
            obs_dark = self.plan_ha(MJDstart, MJDend, False)
            obs_bright = self.plan_ha(MJDstart, MJDend, True)
            obs1 = astropy.table.Table([obs_dark['tileid'], obs_dark['beginobs'], obs_dark['endobs'], obs_dark['obstime'], obs_dark['ha']],
                                       names=('TILEID','BEGINOBS','ENDOBS','OBSTIME','HA'), dtype=('i4','f8','f8','f8','f8'))
            obs2 = astropy.table.Table([obs_bright['tileid'], obs_bright['beginobs'], obs_bright['endobs'], obs_bright['obstime'], obs_bright['ha']],
                                       names=('TILEID','BEGINOBS','ENDOBS','OBSTIME','HA'), dtype=('i4','f8','f8','f8','f8'))
            info = astropy.table.vstack([obs1, obs2], join_type="exact")
            #info.write("ha_check.dat", format="ascii")
        else:
            # Read in the pre-computed HA and begin/end LST range.
            info = astropy.table.Table.read(
                pkg_resources.resource_filename(
                    'desisurvey', 'data/tile-info.fits'), hdu=1)
            # Ignore most of the columns.
            info = info[['TILEID', 'HA', 'BEGINOBS', 'ENDOBS', 'OBSTIME']]

        # Join with our tiles table, matching on TILEID.
        self.tiles = astropy.table.join(
            self.tiles, info, keys='TILEID', join_type='left')
        if len(self.tiles) != self.numtiles:
            raise RuntimeError('Missing some tiles in tile-info.fits')
        # Rename new columns.
        self.tiles.rename_column('BEGINOBS', 'LSTMIN')
        self.tiles.rename_column('ENDOBS', 'LSTMAX')
        self.tiles.rename_column('OBSTIME', 'EXPLEN')


    def afternoonPlan(self, day_stats, date_string, tiles_observed):
        """Main decision making method

        Args:
            day_stats: row of tabulated ephmerides data for today
            date_string: string of the form YYYYMMDD
            tiles_observed: table with follwing columns: tileID, status

        Returns:
            string containg the filename for today's plan; it has the format
            obsplanYYYYMMDD.fits
        """
        nto = len(tiles_observed)

        # Copy the STATUS for previously observed tiles.
        if nto > 0:
            for status in set(tiles_observed['STATUS']):
                ii = (tiles_observed['STATUS'] == status)
                jj = np.in1d(self.tiles['TILEID'], tiles_observed['TILEID'][ii])
                self.tiles['STATUS'][jj] = status

        # Find all tiles with STATUS < 2
        finalTileList = self.tiles[self.tiles['STATUS'] < 2]

        # Assign tiles to LST bins
        planList0 = []
        lst_dusk = mjd2lst(day_stats['dusk'])
        lst_dawn = mjd2lst(day_stats['dawn'])
        lst_brightdusk = mjd2lst(day_stats['brightdusk'])
        lst_brightdawn = mjd2lst(day_stats['brightdawn'])
        LSTmoonrise = mjd2lst(day_stats['moonrise'])
        LSTmoonset = mjd2lst(day_stats['moonset'])
        LSTbrightstart = mjd2lst(day_stats['MJD_bright_start'])
        LSTbrightend = mjd2lst(day_stats['MJD_bright_end'])

        # Calculate LST of each tile in the range [0, 360).
        finalTileLST = finalTileList['RA'] + finalTileList['HA']
        assert np.min(finalTileLST) > -360.
        finalTileLST = np.fmod(finalTileLST + 360., 360.)

        # Select tiles assigned to each program.  The original code tested
        # for bits in OBSCONDITIONS but this is equivalent and faster.
        dark_tile = finalTileList['PROGRAM'] == 'DARK'
        gray_tile = finalTileList['PROGRAM'] == 'GRAY'
        bright_tile = finalTileList['PROGRAM'] == 'BRIGHT'

        # Check that each tile is assigned to exactly one program.
        assert np.all(dark_tile.astype(int) + gray_tile + bright_tile == 1)

        # Assign each tile to an LST bin.
        finalTileLSTbin = np.digitize(finalTileLST, self.LSTedges) - 1
        assert np.all(finalTileLSTbin >= 0)

        # Assign the program for each LST bin tonight.
        def inLSTWindow(start, stop):
            if start <= stop:
                return (self.LSTbins > start) & (self.LSTbins < stop)
            else:
                return (self.LSTbins < stop) | (self.LSTbins > start)

        bright_night = inLSTWindow(lst_brightdusk, lst_brightdawn)
        dark_night = inLSTWindow(lst_dusk, lst_dawn)
        moon_up = inLSTWindow(LSTmoonrise, LSTmoonset)
        bright = inLSTWindow(LSTbrightstart, LSTbrightend)
        dark =  dark_night & ~moon_up
        gray = dark_night & moon_up & ~bright

        # Add the bright twilight periods to the BRIGHT program.
        bright |= bright_night & ~dark_night

        # Check that each bin is assigned to at most one program.
        assert np.max(dark.astype(int) + bright + gray) == 1

        # Loop over LST bins
        for i in range(self.nLST):
            scheduled = []
            # DARK time
            if dark[i]:
                # Find all DARK tiles in this LST bin with STATUS < 2.
                found = np.where(dark_tile & (finalTileLSTbin == i) &
                                 (finalTileList['STATUS'] < 2))[0]
                # Schedule the first 5.
                scheduled.extend(found[:5])
                # If fewer than 5 dark tiles fall within this window, pad with grey
                if len(scheduled) < 5:
                    found = np.where(gray_tile & (finalTileLSTbin == i) &
                                     (finalTileList['STATUS'] < 2))[0]
                    scheduled.extend(found[:5 - len(scheduled)])
                # If fewer than 5 dark or grey tiles fall within this window,
                # pad with bright tiles.
                if len(scheduled) < 5:
                    found = np.where(bright_tile & (finalTileLSTbin == i) &
                                     (finalTileList['STATUS'] < 2))[0]
                    scheduled.extend(found[:5 - len(scheduled)])
            # GRAY time
            if gray[i]:
                # Find all GRAY tiles in this LST bin with STATUS < 2.
                found = np.where(gray_tile & (finalTileLSTbin == i) &
                                 (finalTileList['STATUS'] < 2))[0]
                # Schedule the first 5.
                scheduled.extend(found[:5])
                # If fewer than 5 grey tiles fall within this window, pad with
                # bright tiles.
                if len(scheduled) < 5:
                    found = np.where(bright_tile & (finalTileLSTbin == i) &
                                     (finalTileList['STATUS'] < 2))[0]
                    scheduled.extend(found[:5 - len(scheduled)])
            # BRIGHT time
            if bright[i]:
                # Find all BRIGHT tiles in this LST bin with STATUS < 2.
                found = np.where(bright_tile & (finalTileLSTbin == i) &
                                 (finalTileList['STATUS'] < 2))[0]
                # Schedule the first 5.
                scheduled.extend(found[:5])
            # Assign priorites to each scheduled tile.
            finalTileList['PRIORITY'][scheduled] = 3 + np.arange(len(scheduled))
            planList0.extend(scheduled)

        self.log.info('Afternoon plan contains {0} tiles.'
                      .format(len(planList0)))
        table = finalTileList[planList0]
        table.meta['MOONFRAC'] = day_stats['moon_illum_frac']
        filename = self.config.get_path('obsplan{0}.fits'.format(date_string))
        table.write(filename, overwrite=True)

        tilesTODO = len(planList0)

        return filename


####################################################################
# Below is a translation of Kyle's IDL code to compute hour angles #
####################################################################
    def plan_ha(self, survey_begin, survey_end, BGS=False):
        """Main driver of hour angle computations

            Args:
                survey_begin: MJD of (re-)start of survey
                survey_end: MJD of the expected end

            Optional:
                BGS: bool, true if bright sample
        """

        if BGS:
            exptime = 600.0
        else:
            exptime = 1000.0
        # First define general survey characteristics
        r_threshold = 1.54 # this is an initial guess for required SN2/pixel over r-band
        b_threshold = 0.7 # same for g-band, scaled relative to r-band throughout analysis, the ratio of r-b cannot change
        times = np.copy(self.LSTbins)
        scheduled_times = np.zeros(self.nLST) # available time at each LST bin over the full survey, after accounting for weather loss
        observed_times = np.zeros(self.nLST) # time spent observing at each LST bin, iteratively filled until optimal HA distribution is achieved
        sgcfraction_times = np.zeros(self.nLST) # the fraction of total time in each bin of LST, SGC
        ngcfraction_times = np.zeros(self.nLST) # the fraction of total time in each bin of LST, NGC
        weather=0.74*0.77    # SRD assumption: 74% open dome and 77% good seeing
        excess=1.01

        # There is some repeated code from the afternoon plan
        # which should be factored out.
        for night in self.ephem._table:
            lst_dusk = mjd2lst(night['dusk'])
            lst_dawn = mjd2lst(night['dawn'])
            lst_brightdusk = mjd2lst(night['brightdusk'])
            lst_brightdawn = mjd2lst(night['brightdawn'])
            LSTmoonrise = mjd2lst(night['moonrise'])
            LSTmoonset = mjd2lst(night['moonset'])
            LSTbrightstart = mjd2lst(night['MJD_bright_start'])
            LSTbrightend = mjd2lst(night['MJD_bright_end'])
            for i in range(self.nLST):
                if BGS:
                    if ( (inLSTwindow(self.LSTbins[i], lst_brightdusk, lst_brightdawn) and
                          not inLSTwindow(self.LSTbins[i], lst_dusk, lst_dawn)) or
                          inLSTwindow(self.LSTbins[i], LSTbrightstart, LSTbrightend) ):
                        scheduled_times[i] += 1.0
                else:
                    if ( inLSTwindow(self.LSTbins[i], lst_dusk, lst_dawn) and
                         not inLSTwindow(self.LSTbins[i], LSTmoonrise, LSTmoonset) ):
                        scheduled_times[i] += 1.0
                    if ( inLSTwindow(self.LSTbins[i], lst_dusk, lst_dawn) and
                         inLSTwindow(self.LSTbins[i], LSTmoonrise, LSTmoonset) and
                         not inLSTwindow(self.LSTbins[i], LSTbrightstart, LSTbrightend) ):
                        scheduled_times[i] += 1.0
        scheduled_times *= weather*self.LSTres
        remaining_times = np.copy(scheduled_times)

        surveystruct = {'exptime' : exptime/240.0,  # nominal exposure time (converted from seconds to degrees)
                        'overhead1' : 120.0 / 240.0,      # amount of time for cals and field acquisition (idem)
                        'overhead2' : 60.0 / 240.0,      # amount of time for readout (idem)
                        'survey_begin' : survey_begin,
                        'survey_end' : survey_end,
                        'res' : self.LSTres,
                        'avg_rsn' : 0.75, # SN2/pixel in r-band during nominal exposure time under average conditions, needs to be empirically determined
                        'avg_bsn' : 0.36, # same, for g-band
                        'alpha_red' : 1.25,          # power law for airmass dependence, r-band
                        'alpha_blue' : 1.25,         # same, for g-band
                        'r_threshold' : r_threshold,
                        'b_threshold' : b_threshold,
                        'weather' : weather,         # estimate of time available, after weather loss
                        'times' : times,             # time is in degrees
                        'scheduled_times' : scheduled_times,
                        'observed_times' : observed_times,
                        'remaining_times' : remaining_times,
                        'ngcfraction_times' : ngcfraction_times,
                        'sgcfraction_times' : sgcfraction_times,
                        'ngc_begin' : 75.0,           # estimate of bounds for NGC
                        'ngc_end' : 300.0,            # estimate of bounds for NGC
                        'platearea' : 1.4,           # area in sq degrees per unique tile
                        'surveyarea' : 14000.0}      # required survey area

        obs = self.compute_extinction(BGS)
        surveystruct['platearea'] = surveystruct['surveyarea'] / float( len(obs['tileid']) )

        # FIND MINIMUM AMOUNT OF TIME REQUIRED TO COMPLETE PLATES
        num_obs = len(obs['ra'])
        ha = np.zeros(num_obs, dtype='f8')
        for i in range(num_obs):
            self.filltimes(obs, surveystruct, ha[i], i)

        # FIND OPTIMAL HA FOR FOOTPRINT, ITERATE ONLY ONCE
        optimize = 1
        self.retile(obs, surveystruct, optimize)

        # ADJUST THRESHOLDS ONCE TO MATCH AVAILABLE LST DISTRIBUTION
        a = np.ravel(np.where(obs['obs_bit'] > 1))
        rel_area = len(a)*surveystruct['platearea']/surveystruct['surveyarea']
        obs_avg = np.mean(obs['obstime'][a])
        oh_avg = np.mean(obs['overhead'][a])
        if rel_area < 1.0 and rel_area > 0.0:
            t_scheduled = obs_avg - oh_avg
            t_required = obs_avg*rel_area - oh_avg
            surveystruct['r_threshold'] *= t_required/t_scheduled
            surveystruct['b_threshold'] *= t_required/t_scheduled
        if np.sum(surveystruct['remaining_times']) > 0.0:
            t_scheduled = np.sum(surveystruct['observed_times'])/num_obs - oh_avg
            t_required = np.sum(surveystruct['scheduled_times'])/num_obs - oh_avg
            surveystruct['r_threshold'] *= t_required/t_scheduled*excess
            surveystruct['b_threshold'] *= t_required/t_scheduled*excess
        obs['obs_bit'][:] = 0
        self.retile(obs, surveystruct, optimize)

        return obs

    def compute_extinction (self, BGS=False):

        if BGS:
            a = np.where(self.tiles['PASS'] > 4)
        else:
            a = np.where(self.tiles['PASS'] <= 4)
        subtiles = self.tiles[a]
        ntiles = len(subtiles)
        tileid = subtiles['TILEID']
        ra = subtiles['RA']
        dec = subtiles['DEC']
        ebv = subtiles['EBV_MED']

        layer = subtiles['PASS']
        program = subtiles['PROGRAM']
        obsconditions = subtiles['OBSCONDITIONS']

        i_increase = np.zeros(ntiles, dtype='f8')
        g_increase = np.zeros(ntiles, dtype='f8')
        glong = np.zeros(ntiles, dtype='f8')
        glat = np.zeros(ntiles, dtype='f8')
        overhead = np.zeros(ntiles, dtype='f8')

        # From http://arxiv.org/pdf/1012.4804v2.pdf Table 6
        # R_u = 4.239
        R_g = 3.303
        # R_r = 2.285
        R_i = 1.698
        # R_z = 1.263

        #glong, glat = equ2gal_J2000(ra, dec)
        i_increase = np.power(10.0, 0.8*R_i*ebv)
        g_increase = np.power(10.0, 0.8*R_g*ebv)

        ha = np.zeros(ntiles, dtype='f8')
        airmass = np.ones(ntiles, dtype='f8')
        obs_bit = np.zeros(ntiles, dtype='i4')
        obstime = np.zeros(ntiles, dtype='f8')
        red_sn = np.zeros(ntiles, dtype='f8')
        blue_sn = np.zeros(ntiles, dtype='f8')
        beginobs = np.zeros(ntiles, dtype='f8')
        endobs = np.zeros(ntiles, dtype='f8')

        obs = {'tileid' : tileid,
                'ra' : ra,
                'dec' : dec,
                'glong' : glong,
                'glat' : glat,
                'ha' : ha,
                'airmass' : airmass,
                'ebv' : ebv,
                'i_increase' : i_increase,
                'g_increase' : g_increase,
                'obs_bit' : obs_bit,
                'obstime' : obstime,
                'overhead' : overhead,
                'beginobs' : beginobs,
                'endobs' : endobs,
                'red_sn' : red_sn,
                'blue_sn' : blue_sn,
                'pass' : layer,
                'PROGRAM' : program,
                'OBSCONDITIONS' : obsconditions}

        return obs

    def retile (self, obs, surveystruct, optimize):

        # Re-initialise remaining and observed time arrays.
        surveystruct['remaining_times'] = np.copy(surveystruct['scheduled_times'])
        surveystruct['observed_times'] *= 0.0

        num_obs = len(obs['tileid'])
        times = surveystruct['times']
        num_times = len(times)
        rank_times = np.zeros(num_times, dtype='f8')
        rank_plates = np.zeros(num_obs, dtype='f8')
        dec = obs['dec']
        ra = obs['ra']

        ha_tmp = np.empty(num_obs, dtype='f8')
        airmass_tmp = np.empty(num_obs, dtype='f8')

        index1 = int(np.floor(self.nLST*surveystruct['ngc_begin']/360.0))
        index2 = int(np.floor(self.nLST*surveystruct['ngc_end']/360.0))
        ends =  0.5*surveystruct['scheduled_times'][index1] + 0.5*surveystruct['scheduled_times'][index2]
        ngctime = np.sum(surveystruct['scheduled_times'][index1:index2-1]) + ends
        sgctime = np.sum(surveystruct['scheduled_times'][0:index1-1]) + np.sum(surveystruct['scheduled_times'][index2+1:num_times-1]) + ends
        surveystruct['sgcfraction_times'][0:index1-1] = surveystruct['scheduled_times'][0:index1-1]/sgctime
        surveystruct['sgcfraction_times'][index2:num_times-1] = surveystruct['scheduled_times'][index2:num_times-1]/sgctime
        surveystruct['ngcfraction_times'][index1:index2-1] = surveystruct['scheduled_times'][index1:index2-1]/ngctime
        surveystruct['sgcfraction_times'][index1] = 0.5*surveystruct['scheduled_times'][index1]/sgctime
        surveystruct['ngcfraction_times'][index1] = 0.5*surveystruct['scheduled_times'][index1]/ngctime
        surveystruct['sgcfraction_times'][index2] = 0.5*surveystruct['scheduled_times'][index2]/sgctime
        surveystruct['ngcfraction_times'][index2] = 0.5*surveystruct['scheduled_times'][index2]/ngctime

        obs['obstime'][:] = 0.0
        sgcplates = np.where( (obs['ra'] < surveystruct['ngc_begin']) |
                              (obs['ra'] > surveystruct['ngc_end']) )
        ngcplates = np.where( (obs['ra'] > surveystruct['ngc_begin']) &
                              (obs['ra'] < surveystruct['ngc_end']) )

        # Start by filling the hardest regions with tiles, NGC then SGC

        dec = obs['dec'][ngcplates]
        ra = obs['ra'][ngcplates]
        orig_ha = np.copy(obs['ha'][ngcplates])
        tile = obs['tileid'][ngcplates]
        #obs_bit = obs['obs_bit'][ngcplates]

        nindices = index2-index1
        for i in range(nindices):
            ihalf = i//2
            if 2*ihalf == i:
                index = index1 + ihalf
                ha = times[index] - ra - 0.5*surveystruct['res']
            else:
                index = index2 - ihalf
                ha = times[index] - ra + 0.5*surveystruct['res']
            ha[ha >= 180.0] -= 360.0
            ha[ha <= -180.0] += 360.0
            num_reqplates = int(np.ceil( (surveystruct['ngcfraction_times'][index]*ngctime-surveystruct['observed_times'][index])/surveystruct['res'] ))
            if num_reqplates < 0: # Why is this possible?
                num_reqplates = 0
            airmass = airMassCalculator(ra, dec, ra+ha)
            orig_airmass = airMassCalculator(ra, dec, ra+orig_ha)
            rank_plates_tmp = np.power(airmass, surveystruct['alpha_red']*obs['i_increase'][ngcplates])
            obs_bit = obs['obs_bit'][ngcplates]
            if optimize:
                rank_plates_tmp -= np.power(orig_airmass, surveystruct['alpha_red']*obs['i_increase'][ngcplates])
            else:
                rank_plates_tmp[np.where( (obs_bit < 2) & (np.abs(ha) < 15.0) )] = 1000.0
            todo = np.where(obs_bit < 2)
            asize = len(todo[0])
            if asize == 0:
                break
            if asize < num_reqplates:
                num_reqplates = asize
            rank_plates = rank_plates_tmp[todo]
            tile0 = sort2arr(tile[todo],rank_plates)
            ha0 = sort2arr(ha[todo], rank_plates)
            for j in range(num_reqplates):
                j2 = np.ravel(np.where(obs['tileid'] == tile0[j]))[0]
                h = ha0[j]
                airmass = airMassCalculator(obs['ra'][j2], obs['dec'][j2], obs['ra'][j2]+h)
                red = surveystruct['avg_rsn']/np.power(airmass, surveystruct['alpha_red']/obs['i_increase'][j2])
                rtime = surveystruct['overhead1'] + surveystruct['exptime']*surveystruct['r_threshold']/red
                blue = surveystruct['avg_bsn']/np.power(airmass, surveystruct['alpha_blue'])/obs['g_increase'][j2]
                btime = surveystruct['overhead1'] + surveystruct['exptime']*surveystruct['b_threshold']/blue
                time = np.max([rtime, btime])
                ihalf = i//2
                if 2*ihalf == i:
                    h += 0.5*time
                else:
                    h -= 0.5*time
                obs['obs_bit'][j2] = 2
                self.filltimes(obs, surveystruct, h, j2)
                obs['ha'][j2] = h

        nindices = num_times-(index2-index1)

        dec = obs['dec'][sgcplates]
        ra = obs['ra'][sgcplates]
        orig_ha = np.copy(obs['ha'][sgcplates])
        tile = obs['tileid'][sgcplates]
        #obs_bit = obs['obs_bit'][sgcplates]

        for i in range(nindices):
            ihalf = i//2
            if 2*ihalf != i:
                index = index2 + ihalf
                if index < 0:
                    index += num_times
                if index >= num_times:
                    index -= num_times
                ha = times[index] - ra - 0.5*surveystruct['res']
            else:
                index = index1 - ihalf
                ha = times[index] - ra + 0.5*surveystruct['res']
            ha[ha >= 180.0] -= 360.0
            ha[ha <= -180.0] += 360.0
            num_reqplates = int(np.ceil((surveystruct['sgcfraction_times'][index]*sgctime - surveystruct['observed_times'][index])/surveystruct['res']))
            if num_reqplates < 0: # Why is this possible?
                num_reqplates = 0
            airmass = airMassCalculator(ra, dec, ha+ra)
            orig_airmass = airMassCalculator(ra, dec, orig_ha+ra)
            rank_plates = np.power(airmass, surveystruct['alpha_red']*obs['i_increase'][sgcplates])
            obs_bit = obs['obs_bit'][sgcplates]
            if optimize:
                rank_plates -= np.power(orig_airmass, surveystruct['alpha_red']*obs['i_increase'][sgcplates])
            else:
                rank_plates[np.where( (obs_bit < 2) & (np.abs(ha) < 15.0) )] = 1000.0
            todo = np.where(obs_bit < 2)
            asize = len(todo[0])
            if asize == 0:
                break
            num_reqplates = min([num_reqplates,asize])
            rank_plates = rank_plates[todo]
            tile0 = sort2arr(tile[todo],rank_plates)
            ha0 = sort2arr(ha[todo], rank_plates)

            for j in range(num_reqplates):
                j2 = np.ravel(np.where(obs['tileid'] == tile0[j]))[0]
                h = ha0[j]
                airmass = airMassCalculator(obs['ra'][j2], obs['dec'][j2], obs['ra'][j2]+h)
                red = surveystruct['avg_rsn']/np.power(airmass, surveystruct['alpha_red']/obs['i_increase'][j2])
                rtime = surveystruct['overhead1'] + surveystruct['exptime']*surveystruct['r_threshold']/red
                blue = surveystruct['avg_bsn']/np.power(airmass, surveystruct['alpha_blue']/obs['g_increase'][j2])
                btime = surveystruct['overhead1'] + surveystruct['exptime']*surveystruct['b_threshold']/blue
                time = np.max([rtime,btime])
                if 2*ihalf != i:
                     h += 0.5*time
                if 2*ihalf == i:
                    h -= 0.5*time
                obs['obs_bit'][j2] = 2
                self.filltimes(obs, surveystruct, h, j2)

    def filltimes(self, obs, surveystruct, ha, index):

        res = surveystruct['res']
        times = surveystruct['times']

        overhead = surveystruct['overhead1']
        airmass = airMassCalculator(obs['ra'][index], obs['dec'][index], ha+obs['ra'][index])
        red = surveystruct['avg_rsn'] / np.power(airmass, surveystruct['alpha_red']) / obs['i_increase'][index]
        rtime = surveystruct['exptime']*surveystruct['r_threshold']/red
        blue = surveystruct['avg_bsn'] / np.power(airmass, surveystruct['alpha_blue'])/obs['g_increase'][index]
        btime = surveystruct['exptime']*surveystruct['b_threshold']/blue
        if btime > 5.0 or rtime > 5.0:
            overhead += surveystruct['overhead2']
        rtime += overhead
        btime += overhead
        obs['overhead'][index] = overhead
        time = np.max([rtime,btime])

        obs['red_sn'][index] = red*(time-overhead)/surveystruct['exptime']
        obs['blue_sn'][index] = blue*(time-overhead)/surveystruct['exptime']
        obs['obstime'][index] = time
        obs['beginobs'][index] = obs['ra'][index] + ha - 0.5*time
        obs['endobs'][index] = obs['ra'][index] + ha + 0.5*time
        obs['airmass'][index] = airmass
        obs['ha'][index] = ha

        if obs['beginobs'][index] < 0.0 and obs['endobs'][index] < 0.0:
            obs['beginobs'][index] += 360.0
            obs['endobs'][index] += 360.0

        if obs['beginobs'][index] > 360.0 and obs['endobs'][index] > 360.0:
            obs['beginobs'][index] -= 360.0
            obs['endobs'][index] -= 360.0

        #fill in times over LST range
        num = len(surveystruct['times'])
        for i in range(num):
            if obs['beginobs'][index] <= surveystruct['times'][i]-0.5*res and obs['endobs'][index] >= surveystruct['times'][i]+0.5*res:
                surveystruct['remaining_times'][i] -= res
                surveystruct['observed_times'][i] += res
            if obs['beginobs'][index] > surveystruct['times'][i]-0.5*res and obs['beginobs'][index] < surveystruct['times'][i]+0.5*res:
                surveystruct['remaining_times'][i] -= (surveystruct['times'][i]+0.5*res-obs['beginobs'][index])
                surveystruct['observed_times'][i] += (surveystruct['times'][i]+0.5*res-obs['beginobs'][index])
            if obs['endobs'][index] > surveystruct['times'][i]-0.5*res and obs['endobs'][index] < surveystruct['times'][i]+0.5*res:
                surveystruct['remaining_times'][i] -= (-(surveystruct['times'][i]-0.5*res)+obs['endobs'][index])
                surveystruct['observed_times'][i] += (-(surveystruct['times'][i]-0.5*res)+obs['endobs'][index])

        if obs['beginobs'][index] < 0.0:
            t = np.floor(-obs['beginobs'][index]/res)
            it = int(t)
            if t > 0.0:
                surveystruct['remaining_times'][num-it:num-1] -= res
                surveystruct['observed_times'][num-it:num-1] += res
            surveystruct['remaining_times'][num-it-1] -= (-obs['beginobs'][index]-t*res)
            surveystruct['observed_times'][num-it-1] += (-obs['beginobs'][index]-t*res)
            obs['beginobs'][index] += 360.0

        if obs['endobs'][index] > 360.0:
            obs['endobs'][index] -= 360.0 * np.floor(obs['endobs'][index]/360.0)
            t = np.floor(obs['endobs'][index]/res)
            it = int(t)
            if t > 0.0:
                surveystruct['remaining_times'][0:it-1] -= res
                surveystruct['observed_times'][0:it-1] += res
            surveystruct['remaining_times'][it] -= (obs['endobs'][index]-t*res)
            surveystruct['observed_times'][it] += (obs['endobs'][index]-t*res)
