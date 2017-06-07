"""Plan an observing night during the previous afternoon.
"""
from __future__ import print_function, division

import pkg_resources

import numpy as np
from datetime import datetime, date, time

import astropy.table
import astropy.utils.exceptions
import astropy.units as u
from astropy.time import Time

import desimodel.io
import desiutil.log

import desisurvey.config
from desisurvey.utils import mjd2lst, inLSTwindow, cos_zenith_to_airmass, cos_zenith, sort2arr, is_monsoon


class surveyPlan:
    """Plan an observing night during the previous afternoon.

    Initialises survey by reading in the file desi_tiles.fits and populates the
    class members.

    Parameters
    ----------
    MJDstart : float
        Day of the (re-)start of the survey
    MJDend : float
        Day of the end of the survey
    ephem : desisurvey.ephemerides.Ephemerides
        Tabulated ephemerides covering MJDstart - MJDend.
    HA_assign : bool
        Calculate HA assignments if True, otherwise read from a file.
    """
    def __init__(self, MJDstart, MJDend, ephem, HA_assign=False):
        self.log = desiutil.log.get_logger()
        self.config = desisurvey.config.Configuration()
        self.ephem = ephem

        # Read in DESI tile data
        tiles = astropy.table.Table(
            desimodel.io.load_tiles(onlydesi=True, extra=False))
        numtiles = len(tiles)

        # Drop un-needed columns.
        tiles.remove_columns([
            'IN_DESI', 'AIRMASS', 'STAR_DENSITY', 'EXPOSEFAC'])

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
        tiles['SUBLIST'][(tiles['GAL_CAP'] < 0) | (dec < dec_min) | (dec > dec_max)] += 8

        # Initialize the LST bins we will use for scheduling each night.
        self.nLST = self.config.num_lst_bins()
        self.LSTedges = np.linspace(0.0, 360.0, self.nLST + 1)
        self.LSTbins = 0.5 * (self.LSTedges[1:] + self.LSTedges[:-1])
        self.LSTres = self.LSTedges[1]

        self.tiles = tiles
        self.numtiles = numtiles

        # Add HA, LSTMIN, LSTMAX columns.
        self.assignHA(MJDstart, MJDend, compute=HA_assign)

        self.tiles.sort(('SUBLIST', 'DEC'))

    def assignHA(self, MJDstart, MJDend, compute=False):
        """Assign optimal hour angles for the DESI tiles.

        Can be re-run at any point during the survey to
        reoptimise the schedule.

        Parameters
        ----------
        MJDstart : float
            Time at which the assignment starts.
        MJDend : float
            Time by which the survey is expected to be completed.
        compute : bool
            False reads a pre-computed table; for development purposes only.
        """
        if compute:
            t = Time(datetime.combine(self.config.last_day(), time(18,0,0)))
            nominalMJDend = t.mjd
            if nominalMJDend > MJDend:
                MJDstop = nominalMJDend
            else:
                MJDstop = MJDend
            obs_dark = self.plan_ha(MJDstart, MJDstop, "DARK")
            obs_gray = self.plan_ha(MJDstart, MJDstop, "GRAY")
            obs_bright = self.plan_ha(MJDstart, MJDstop, "BRIGHT")
            obs1 = astropy.table.Table(
                [obs_dark['tileid'], obs_dark['LSTMIN'], obs_dark['LSTMAX'],
                 obs_dark['EXPLEN'], obs_dark['ha']],
                names=('TILEID','LSTMIN','LSTMAX','EXPLEN','HA'),
                dtype=('i4','f8','f8','f8','f8'))
            obs2 = astropy.table.Table(
                [obs_gray['tileid'], obs_gray['LSTMIN'], obs_gray['LSTMAX'],
                 obs_gray['EXPLEN'], obs_gray['ha']],
                names=('TILEID','LSTMIN','LSTMAX','EXPLEN','HA'),
                dtype=('i4','f8','f8','f8','f8'))
            obs3 = astropy.table.Table(
                [obs_bright['tileid'], obs_bright['LSTMIN'],
                 obs_bright['LSTMAX'], obs_bright['EXPLEN'], obs_bright['ha']],
                names=('TILEID','LSTMIN','LSTMAX','EXPLEN','HA'),
                dtype=('i4','f8','f8','f8','f8'))
            info = astropy.table.vstack([obs1, obs2, obs3], join_type="exact")
            info.write("ha_check.dat", format="ascii")
        else:
            # Read in the pre-computed HA and begin/end LST range.
            info = astropy.table.Table.read(
                pkg_resources.resource_filename(
                    'desisurvey', 'data/tile-info.fits'), hdu=1)
            # Ignore most of the columns.
            info = info[['TILEID', 'HA', 'BEGINOBS', 'ENDOBS', 'OBSTIME']]
            # File has these entries in the wrong units.
            info['BEGINOBS'] *= 15.0
            info['ENDOBS'] *= 15.0
            # Rename new columns.
            info.rename_column('BEGINOBS', 'LSTMIN')
            info.rename_column('ENDOBS', 'LSTMAX')
            info.rename_column('OBSTIME', 'EXPLEN')

        # Join with our tiles table, matching on TILEID.
        self.tiles = astropy.table.join(
            self.tiles, info, keys='TILEID', join_type='left')
        if len(self.tiles) != self.numtiles:
            raise RuntimeError('Missing some tiles in tile-info.fits')

        import sys
        sys.exit()

    def afternoonPlan(self, day_stats, progress):
        """Main decision making method.

        Parameters
        ----------
        day_stats : astropy.table.Row
            Row of tabulated ephmerides data for tonight's observing.
        progress : desisurvey.progress.Progress
            Record of observations made so far.

        Returns
        -------
        string
            The filename for today's plan; it has the format
            obsplanYYYYMMDD.fits
        """
        # Get a list of previously observed tiles, including those which
        # have not yet reached their SNR**2 target (with status == 1).
        tiles_observed = progress.get_summary('observed')
        nto = len(tiles_observed)

        # Copy the STATUS for previously observed tiles.
        if nto > 0:
            for status in set(tiles_observed['status']):
                ii = (tiles_observed['status'] == status)
                jj = np.in1d(self.tiles['TILEID'], tiles_observed['tileid'][ii])
                self.tiles['STATUS'][jj] = status

        # Find all tiles that have never been observed. We eventually want
        # to also schedule re-observations of partial tiles, but this is not
        # working yet.
        finalTileList = self.tiles[self.tiles['STATUS'] < 1]

        # Assign tiles to LST bins
        planList0 = []
        lst_dusk = mjd2lst(day_stats['dusk'])
        lst_dawn = mjd2lst(day_stats['dawn'])
        lst_brightdusk = mjd2lst(day_stats['brightdusk'])
        lst_brightdawn = mjd2lst(day_stats['brightdawn'])
        LSTmoonrise = mjd2lst(day_stats['moonrise'])
        LSTmoonset = mjd2lst(day_stats['moonset'])
        LSTbrightstart = mjd2lst(day_stats['brightstart'])
        LSTbrightend = mjd2lst(day_stats['brightstop'])

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
                # If fewer than 5 dark tiles fall within this window,
                # pad with grey
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

        date = desisurvey.utils.get_date(day_stats['noon'] + 0.5)
        self.log.info('Afternoon plan for {0} contains {1} tiles.'
                      .format(date, len(planList0)))

        table = finalTileList[planList0]
        table.meta['MOONFRAC'] = day_stats['moon_illum_frac']
        filename = self.config.get_path(
            'obsplan{0}.fits'.format(date.strftime('%Y%m%d')))
        table.write(filename, overwrite=True)

        return filename

####################################################################
# Below is a translation of Kyle's IDL code to compute hour angles #
####################################################################
    def plan_ha(self, survey_begin, survey_end, program):
        """Main driver of hour angle computations

            Args:
                survey_begin: MJD of (re-)start of survey
                survey_end: MJD of the expected end
                program: "DARK", "GRAY" or "BRIGHT"
        """

        if program=="BRIGHT":
            exptime = self.config.nominal_exposure_time.BRIGHT().value
        elif program=="GRAY":
            exptime = self.config.nominal_exposure_time.GRAY().value
        elif program=="DARK":
            exptime = self.config.nominal_exposure_time.DARK().value
        else:
            print("ERROR[desisurvey.afternoonplan.plan_ha]: Unknown program")
        # First define general survey characteristics
        r_threshold = 1.54 # this is an initial guess for required SN2/pixel over r-band
        b_threshold = 0.7 # same for g-band, scaled relative to r-band throughout analysis, the ratio of r-b cannot change
        weather=0.74*0.77    # SRD assumption: 74% open dome and 77% good seeing
        excess=1.01
        times = np.copy(self.LSTbins)
        scheduled_times = np.zeros(self.nLST) # available time at each LST bin over the full survey, after accounting for weather loss
        observed_times = np.zeros(self.nLST) # time spent observing at each LST bin, iteratively filled until optimal HA distribution is achieved
        sgcfraction_times = np.zeros(self.nLST) # the fraction of total time in each bin of LST, SGC
        ngcfraction_times = np.zeros(self.nLST) # the fraction of total time in each bin of LST, NGC

        # There is some repeated code from the afternoon plan
        # which should be factored out.
        for night in self.ephem._table:
            if not is_monsoon(night['noon']) and not self.ephem.is_full_moon(night['noon']):
                lst_dusk = mjd2lst(night['dusk'])
                lst_dawn = mjd2lst(night['dawn'])
                lst_brightdusk = mjd2lst(night['brightdusk'])
                lst_brightdawn = mjd2lst(night['brightdawn'])
                LSTmoonrise = mjd2lst(night['moonrise'])
                LSTmoonset = mjd2lst(night['moonset'])
                LSTbrightstart = mjd2lst(night['brightstart'])
                LSTbrightend = mjd2lst(night['brightstop'])
                if program=="BRIGHT":
                    scheduled_times[(inLSTwindow(self.LSTbins, lst_brightdusk, lst_brightdawn) &
                                    ~inLSTwindow(self.LSTbins, lst_dusk, lst_dawn)) |
                                    inLSTwindow(self.LSTbins, LSTbrightstart, LSTbrightend)] += 1.0
                elif program=="DARK":
                    scheduled_times[inLSTwindow(self.LSTbins, lst_dusk, lst_dawn) &
                                    ~inLSTwindow(self.LSTbins, LSTmoonrise, LSTmoonset)] += 1.0
                elif program=="GRAY":
                    scheduled_times[inLSTwindow(self.LSTbins, lst_dusk, lst_dawn) &
                                    inLSTwindow(self.LSTbins, LSTmoonrise, LSTmoonset) &
                                    ~inLSTwindow(self.LSTbins, LSTbrightstart, LSTbrightend)] += 1.0
                else:
                    print("ERROR[desisurvey.afternoonplan.plan_ha]: Unknown program.\n")
        print("Scheduled times: ", np.sum(scheduled_times))
        scheduled_times *= weather*self.LSTres
        remaining_times = np.copy(scheduled_times)

        surveystruct = {'exptime' : exptime/240.0,  # nominal exposure time (converted from seconds to degrees)
                        'overhead1' : 120.0 / 240.0,      # Minimum amount of time for cals and field acquisition (idem)
                        'overhead2' : 120.0 / 240.0,      # amount of time for readout (idem)
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
                        'ngc_end' : 300.0}            # estimate of bounds for NGC

        obs = self.compute_extinction(program)

        # FIND MINIMUM AMOUNT OF TIME REQUIRED TO COMPLETE PLATES
        num_obs = len(obs['ra'])
        ha = np.zeros(num_obs, dtype='f8')
        for i in range(num_obs):
            self.filltimes(obs, surveystruct, ha[i], i)

        # FIND OPTIMAL HA FOR FOOTPRINT, ITERATE ONLY ONCE
        optimize = 1
        self.retile(obs, surveystruct, optimize)
        print("HA assigned for ", len(np.ravel(np.where(obs['EXPLEN']!=0.0))), " out of ", num_obs," tiles.")

        # ADJUST THRESHOLDS ONCE TO MATCH AVAILABLE LST DISTRIBUTION
        a = np.ravel(np.where(obs['obs_bit'] > 1))
        rel_area = len(a) / num_obs
        #obs_avg = np.mean(obs['EXPLEN'][a])
        #oh_avg = np.mean(obs['overhead'][a])
        #if rel_area < 1.0 and rel_area > 0.0:
        while rel_area < 1.0 and rel_area > 0.0:
            obs_avg = np.mean(obs['EXPLEN'][a])
            oh_avg = np.mean(obs['overhead'][a])
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
            a = np.ravel(np.where(obs['obs_bit'] > 1))
            rel_area = len(a) / num_obs
            print("HA assigned for ", len(a), " out of ", num_obs," tiles.")
            print("Unused available time: ", np.sum(surveystruct['remaining_times']))

        return obs

    def compute_extinction (self, program):

        if program=="DARK":
            a = np.where(self.tiles['PASS'] < 4)
        elif program=="GRAY":
            a = np.where(self.tiles['PASS'] == 4)
        elif program=="BRIGHT":
            a = np.where(self.tiles['PASS'] > 4)
        subtiles = self.tiles[a]
        ntiles = len(subtiles)
        tileid = subtiles['TILEID']
        ra = subtiles['RA']
        dec = subtiles['DEC']
        ebv = subtiles['EBV_MED']

        layer = subtiles['PASS']
        #program = subtiles['PROGRAM']
        #obsconditions = subtiles['OBSCONDITIONS']

        i_increase = np.zeros(ntiles, dtype='f8')
        g_increase = np.zeros(ntiles, dtype='f8')
        #glong = np.zeros(ntiles, dtype='f8')
        #glat = np.zeros(ntiles, dtype='f8')
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
                #'glong' : glong,
                #'glat' : glat,
                'ha' : ha,
                'airmass' : airmass,
                'ebv' : ebv,
                'i_increase' : i_increase,
                'g_increase' : g_increase,
                'obs_bit' : obs_bit,
                'EXPLEN' : obstime,
                'overhead' : overhead,
                'LSTMIN' : beginobs,
                'LSTMAX' : endobs,
                'red_sn' : red_sn,
                'blue_sn' : blue_sn,
                'pass' : layer,
                #'PROGRAM' : program,
                #'OBSCONDITIONS' : obsconditions
                }

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

        Mayall_lat_deg = self.config.location.latitude().to(u.deg)
        max_airmass = cos_zenith_to_airmass(np.cos(0.5*np.pi-self.config.min_altitude().to(u.rad).value))
        
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

        obs['EXPLEN'][:] = 0.0
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
            #airmass = airMassCalculator(ra, dec, ra+ha)
            #orig_airmass = airMassCalculator(ra, dec, ra+orig_ha)
            airmass = cos_zenith_to_airmass(cos_zenith(ha*u.deg, dec*u.deg, Mayall_lat_deg))
            num_ok_airmass = len( (np.where(airmass <= max_airmass))[0] )
            #ii = np.where(airmass <= max_airmass)
            #ha_temp = ha[ii]
            #print( len(ha_temp[ha_temp>180.0]), len(ha_temp[ha_temp<-180.0]) )
            orig_airmass = cos_zenith_to_airmass(cos_zenith(orig_ha*u.deg, dec*u.deg, Mayall_lat_deg))
            rank_plates_tmp = np.power(airmass, surveystruct['alpha_red'])*obs['i_increase'][ngcplates]
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
            if num_ok_airmass < num_reqplates:
                num_reqplates = num_ok_airmass
            rank_plates = rank_plates_tmp[todo]
            tile0 = sort2arr(tile[todo],rank_plates)
            ha0 = sort2arr(ha[todo], rank_plates)
            for j in range(num_reqplates):
                j2 = np.ravel(np.where(obs['tileid'] == tile0[j]))[0]
                h = ha0[j]
                #airmass = airMassCalculator(obs['ra'][j2], obs['dec'][j2], obs['ra'][j2]+h)
                airmass = cos_zenith_to_airmass(cos_zenith(h*u.deg, dec*u.deg, Mayall_lat_deg))
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

        #print("Now SGC\n")
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
            #airmass = airMassCalculator(ra, dec, ha+ra)
            #orig_airmass = airMassCalculator(ra, dec, orig_ha+ra)
            airmass = cos_zenith_to_airmass(cos_zenith(ha*u.deg, dec*u.deg, Mayall_lat_deg))
            num_ok_airmass = len( (np.where(airmass <= max_airmass))[0] )
            orig_airmass = cos_zenith_to_airmass(cos_zenith(orig_ha*u.deg, dec*u.deg, Mayall_lat_deg))
            rank_plates = np.power(airmass, surveystruct['alpha_red'])*obs['i_increase'][sgcplates]
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
            if num_ok_airmass < num_reqplates:
                num_reqplates = num_ok_airmass
            rank_plates = rank_plates[todo]
            tile0 = sort2arr(tile[todo],rank_plates)
            ha0 = sort2arr(ha[todo], rank_plates)

            for j in range(num_reqplates):
                j2 = np.ravel(np.where(obs['tileid'] == tile0[j]))[0]
                h = ha0[j]
                #airmass = airMassCalculator(obs['ra'][j2], obs['dec'][j2], obs['ra'][j2]+h)
                airmass = cos_zenith_to_airmass(cos_zenith(h*u.deg, dec*u.deg, Mayall_lat_deg))
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

        #print(ha)
        res = surveystruct['res']
        times = surveystruct['times']

        overhead = surveystruct['overhead1']
        #airmass = airMassCalculator(obs['ra'][index], obs['dec'][index], ha+obs['ra'][index])
        airmass = cos_zenith_to_airmass(cos_zenith(ha*u.deg, obs['dec'][index]*u.deg))
        #print(zrad(self.config.location.latitude().to(u.rad).value, np.radians(obs['dec'][index]), np.radians(ha)),
        #      airmass)
        red = surveystruct['avg_rsn'] / np.power(airmass, surveystruct['alpha_red']) / obs['i_increase'][index]
        rtime = surveystruct['exptime']*surveystruct['r_threshold']/red
        blue = surveystruct['avg_bsn'] / np.power(airmass, surveystruct['alpha_blue'])/obs['g_increase'][index]
        btime = surveystruct['exptime']*surveystruct['b_threshold']/blue
        if btime > 15.0 or rtime > 15.0:
            overhead += surveystruct['overhead2']
        rtime += overhead
        btime += overhead
        obs['overhead'][index] = overhead
        time = np.max([rtime,btime])

        obs['red_sn'][index] = red*(time-overhead)/surveystruct['exptime']
        obs['blue_sn'][index] = blue*(time-overhead)/surveystruct['exptime']
        obs['EXPLEN'][index] = time * 240.0 # Convert to seconds.
        obs['LSTMIN'][index] = obs['ra'][index] + ha - 0.5*time
        obs['LSTMAX'][index] = obs['ra'][index] + ha + 0.5*time
        obs['airmass'][index] = airmass
        obs['ha'][index] = ha
        #print(ha, obs['ra'][index], time, obs['beginobs'][index], obs['endobs'][index])
        if obs['LSTMIN'][index] < 0.0 and obs['LSTMAX'][index] < 0.0:
            obs['LSTMIN'][index] += 360.0
            obs['LSTMAX'][index] += 360.0

        if obs['LSTMIN'][index] > 360.0 and obs['LSTMAX'][index] > 360.0:
            obs['LSTMIN'][index] -= 360.0
            obs['LSTMAX'][index] -= 360.0

        #fill in times over LST range
        num = len(surveystruct['times'])
        t_b = obs['LSTMIN'][index]
        t_e = obs['LSTMAX'][index]
        #print (t_b, t_e)
        for i in range(num):
            t1 = surveystruct['times'][i]-0.5*res
            t2 = surveystruct['times'][i]+0.5*res
            if t_b <= t1 and t_e >= t2:
                surveystruct['remaining_times'][i] -= res
                surveystruct['observed_times'][i] += res
            if t_b > t1 and t_b < t2:
                surveystruct['remaining_times'][i] -= (t2-t_b)
                surveystruct['observed_times'][i] += (t2-t_b)
            if t_e > t1 and t_e < t2:
                surveystruct['remaining_times'][i] -= (-t1+t_e)
                surveystruct['observed_times'][i] += (-t1+t_e)

        if t_b < 0.0:
            t = np.floor(-t_b/res)
            it = int(t)
            if t > 0.0:
                surveystruct['remaining_times'][num-it:num-1] -= res
                surveystruct['observed_times'][num-it:num-1] += res
            #print(obs['beginobs'][index], t)
            surveystruct['remaining_times'][num-it-1] -= (-t_b-t*res)
            surveystruct['observed_times'][num-it-1] += (-t_b-t*res)
            obs['LSTMIN'][index] += 360.0

        if t_e > 360.0:
            obs['LSTMAX'][index] -= 360.0 #* np.floor(obs['endobs'][index]/360.0)
            t_e = obs['LSTMAX'][index]
            t = np.floor(t_e/res)
            it = int(t)
            if t > 0.0:
                surveystruct['remaining_times'][0:it-1] -= res
                surveystruct['observed_times'][0:it-1] += res
            surveystruct['remaining_times'][it] -= (t_e-t*res)
            surveystruct['observed_times'][it] += (t_e-t*res)
