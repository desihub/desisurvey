"""Calculate the nominal exposure time for specified observing conditions.

Use :func:`exposure_time` to combine all effects into an exposure time in
seconds, or call functions to calculate the individual exposure-time factors
associated with each effect.

The following effects are included: seeing, transparency, galactic dust
extinction, airmass, scattered moonlight.  The following effects are not yet
implemented: twilight sky brightness, clouds, variable OH sky brightness.
"""
from __future__ import print_function, division

import h5py 
import pickle
import numpy as np
from scipy.interpolate import interp1d
from itertools import chain, combinations_with_replacement

import astropy.utils.data
import astropy.units as u

import speclite.filters

import desiutil.log

import desisurvey.config
import desisurvey.tiles
        
from sklearn.gaussian_process import GaussianProcessRegressor as GPR


def seeing_exposure_factor(seeing):
    """Scaling of exposure time with seeing, relative to nominal seeing.

    The model is based on DESI simulations with convolutions of realistic
    atmospheric and instrument PSFs, for a nominal sample of DESI ELG
    targets (including redshift evolution of ELG angular size).

    The simulations predict SNR for the ELG [OII] doublet during dark-sky
    conditions at airmass X=1.  The exposure factor assumes exposure time
    scales with SNR ** -0.5.

    Parameters
    ----------
    seeing : float or array
        FWHM seeing value(s) in arcseconds.

    Returns
    -------
    float
        Multiplicative factor(s) that exposure time should be adjusted based
        on the actual vs nominal seeing.
    """
    seeing = np.asarray(seeing)
    if np.any(seeing <= 0):
        raise ValueError('Got invalid seeing value <= 0.')
    a, b, c = 12.95475751, -7.10892892, 1.21068726
    f_seeing =  (a + b * seeing + c * seeing ** 2) ** -2
    config = desisurvey.config.Configuration()
    nominal = config.nominal_conditions.seeing().to(u.arcsec).value
    f_nominal = (a + b * nominal + c * nominal ** 2) ** -2
    return f_seeing / f_nominal


def transparency_exposure_factor(transparency):
    """Scaling of exposure time with transparency relative to nominal.

    The model is that exposure time scales with 1 / transparency.

    Parameters
    ----------
    transparency : float or array
        Dimensionless transparency value(s) in the range [0-1].

    Returns
    -------
    float
        Multiplicative factor(s) that exposure time should be adjusted based
        on the actual vs nominal transparency.
    """
    transparency = np.asarray(transparency)
    if np.any(transparency <= 0):
        raise ValueError('Got invalid transparency value <= 0.')
    config = desisurvey.config.Configuration()
    nominal = config.nominal_conditions.transparency()
    return nominal / transparency


def dust_exposure_factor(EBV):
    """Scaling of exposure time with median E(B-V) relative to nominal.

    The model uses the SDSS-g extinction coefficient (3.303) from Table 6
    of Schlafly & Finkbeiner 2011.

    Parameters
    ----------
    EBV : float or array
        Median dust extinction value(s) E(B-V) for the tile area.

    Returns
    -------
    float
        Multiplicative factor(s) that exposure time should be adjusted based
        on the actual vs nominal dust extinction.
    """
    EBV = np.asarray(EBV)
    config = desisurvey.config.Configuration()
    EBV0 = config.nominal_conditions.EBV()
    Ag = 3.303 * (EBV - EBV0)
    return np.power(10.0, (2.0 * Ag / 2.5))


def airmass_exposure_factor(airmass):
    """Scaling of exposure time with airmass relative to nominal.

    The exponent 1.25 is based on empirical fits to BOSS exposure
    times. See eqn (6) of Dawson 2012 for details.

    Parameters
    ----------
    airmass : float or array
        Airmass value(s)

    Returns
    -------
    float
        Multiplicative factor(s) that exposure time should be adjusted based
        on the actual vs nominal airmass.
    """
    X = np.asarray(airmass)
    if np.any(X < 1):
        raise ValueError('Got invalid airmass value < 1.')
    config = desisurvey.config.Configuration()
    X0 = config.nominal_conditions.airmass()
    return np.power((X / X0), 1.25)


# Linear regression coefficients for converting scattered moon V-band
# magnitude into an exposure-time correction factor.
_moonCoefficients = np.array([
    -8.83964463188, -7372368.5041596508, 775.17763895781638,
    -20185.959363990656, 174143.69095766739])

# V-band extinction coefficient to use in the scattered moonlight model.
# See specsim.atmosphere.krisciunas_schaefer for details.
_vband_extinction = 0.15154

# surface brightness of the nominal dark sky at ~4500A.
_dark_sky_4500A = 1.519 

def moon_exposure_factor(moon_frac, moon_sep, moon_alt, airmass):
    """Calculate exposure time factor due to scattered moonlight.

    The returned factor is relative to dark conditions when the moon is
    below the local horizon.

    This factor is based on a study of SNR for ELG targets and designed to
    achieve a median SNR of 7 for a typical ELG [OII] doublet at the lower
    flux limit of 8e-17 erg/(cm2 s A), averaged over the expected ELG target
    redshift distribution 0.6 < z < 1.7.

    TODO:
    - Check the assumption that exposure time scales with SNR ** -0.5.
    - Check if this ELG-based analysis is also valid for BGS targets.

    For details, see the jupyter notebook doc/nb/ScatteredMoon.ipynb in
    this package.

    Parameters
    ----------
    moon_frac : float
        Illuminated fraction of the moon, in the range [0,1].
    moon_sep : float
        Separation angle between field center and moon in degrees, in the
        range [0,180].
    moon_alt : float
        Altitude angle of the moon above the horizon in degrees, in the
        range [-90,90].
    airmass : float
        Airmass used for observing this tile, must be >= 1.

    Returns
    -------
    float
        Dimensionless factor that exposure time should be increased to
        account for increased sky brightness due to scattered moonlight.
        Will be 1 when the moon is below the horizon.
    """
    if (moon_frac < 0) or (moon_frac > 1):
        raise ValueError('Got invalid moon_frac outside [0,1].')
    if (moon_sep < 0) or (moon_sep > 180):
        raise ValueError('Got invalid moon_sep outside [0,180].')
    if (moon_alt < -90) or (moon_alt > 90):
        raise ValueError('Got invalid moon_alt outside [-90,+90].')
    if airmass < 1:
        raise ValueError('Got invalid airmass < 1.')

    # No exposure penalty when moon is below the horizon.
    if moon_alt < 0:
        return 1.

    # Convert input parameters to those used in the specim moon model.
    moon_phase = np.arccos(2 * moon_frac - 1) / np.pi
    separation_angle = moon_sep * u.deg
    moon_zenith = (90 - moon_alt) * u.deg

    # Estimate the zenith angle corresponding to this observing airmass.
    # We invert eqn.3 of KS1991 for this (instead of eqn.14).
    obs_zenith = np.arcsin(np.sqrt((1 - airmass ** -2) / 0.96)) * u.rad

    # Calculate scattered moon V-band brightness at each pixel.
    #V = specsim.atmosphere.krisciunas_schaefer(
    #    obs_zenith, moon_zenith, separation_angle,
    #    moon_phase, _vband_extinction).value
    V = krisciunas_schaefer_refit(
        obs_zenith, moon_zenith, separation_angle,
        moon_phase, _vband_extinction).value

    # Evaluate the linear regression model.
    X = np.array((1, np.exp(-V), 1/V, 1/V**2, 1/V**3))
    return _moonCoefficients.dot(X)


def bright_exposure_factor(airmass, moon_frac, moon_sep, moon_alt, sun_sep, sun_alt):
    """ Calculate exposure time factor based on airmass, moon, and sun parameters. 

    :param moon_frac: 
        Illuminated fraction of the moon within range [0,1].

    :param moon_alt:
        Altitude angle of the moon above the horizon in degrees within range [-90,90].

    :param moon_sep:
        Array of separation angles between field center and moon in degrees within the
        range [0,180].

    :param sun_alt:
        Altitude angle of the sunin degrees

    :param sun_sep: 
        Arry of separations angles between field center and sun in degrees

    :param airmass:
        Array of airmass used for observing this tile, must be >= 1.

    :returns expfactors: 
        Dimensionless factors that exposure time should be increased to
        account for increased sky brightness due to scattered moonlight.
        Will be 1 when the moon is below the horizon.
    """
    # exposure_factor = 1 when moon is below the horizon and sun is below -20.
    if moon_alt < 0 and sun_alt < -20.:
        return np.ones(len(airmass))  

    # check inputs 
    moon_sep    = moon_sep.flatten() 
    sun_sep     = sun_sep.flatten()
    airmass     = airmass.flatten() 
    if (moon_frac < 0) or (moon_frac > 1):
        raise ValueError('Got invalid moon_frac outside [0,1].')
    if (moon_alt < -90) or (moon_alt > 90):
        raise ValueError('Got invalid moon_alt outside [-90,+90].')
    if (moon_sep.min() < 0) or (moon_sep.max() > 180):
        raise ValueError('Got invalid moon_sep outside [0,180].')
    if airmass.min() < 1:
        raise ValueError('Got invalid airmass < 1.')

    # check size of inputs  
    nexp = len(airmass) 
    assert len(moon_sep) == nexp
    assert len(sun_sep) == nexp
    
    exp_factors = _bright_exposure_factor_notwi(
            airmass, 
            np.repeat(moon_frac, nexp), 
            moon_sep, 
            np.repeat(moon_alt, nexp)) 

    if sun_alt >= -20.: 
        # w/ twilight contribution
        exp_factors += _bright_exposure_factor_twi(
                airmass, 
                sun_sep, 
                np.repeat(sun_alt, nexp))
    return np.clip(exp_factors, 1., None) 


# polynomial regression cofficients for estimating exposure time factor during
# non-twilight from airmass, moon_frac, moon_sep, moon_alt  
_notwiCoefficients_4500 = np.array([1.71573612e-09, -5.60790479e-01, -1.48073306e+00,
    1.15296186e-01, 4.52848807e-02,  6.87825142e-01, -1.09204980e-01,
    -2.29003929e-02, 5.43115007e-02,  3.73159137e+00, -1.02979745e-02,
    -7.54139533e-02, -1.05738392e-03, -1.40087568e-03, -1.61643490e-04,
    -4.83874245e-01, -1.27069698e+00,  3.10724722e-02,  6.37391242e-03,
    6.03557732e+00, 1.93135069e-02,  6.44469267e-02, -4.85686295e-04,
    -1.14612065e-03, -5.46053442e-04,  4.53593491e-01, -6.38120677e-02,
    1.30231834e-01, 6.98885410e-05, -7.61371808e-04, -9.32261381e-04,
    6.59488169e-06, 1.78548384e-05,  1.37397769e-05,  2.42228917e-06])  

def _bright_exposure_factor_notwi(airmass, moon_frac, moon_sep, moon_alt, wavelength=4500): 
    ''' third degree polynomial regression fit to exposure factor of  
    non-twilight bright sky given airmass and moon_conditions 

    :param airmass: 
        array of airmasses
    :param moon_frac: 
        array of moon illumination fractions
    :param moon_sep: 
        array of moon separations
    :param moon_alt: 
        array of moon altitudes 
    :param wavelength: 
        wavelength of the exposure factor (default: 4500) 
    :return fexp: 
        exposure factor for non-twlight bright sky 
    '''
    theta = np.atleast_2d(np.array([airmass, moon_frac, moon_sep, moon_alt]).T)

    combs = chain.from_iterable(combinations_with_replacement(range(4), i) for i in range(0, 4))

    theta_transform = np.empty((theta.shape[0], 35))
    for i, comb in enumerate(combs):
        theta_transform[:, i] = theta[:, comb].prod(1)

    if wavelength == 4500: 
        _notwiCoefficients = _notwiCoefficients_4500
        _notwiIntercept = -1.9417946048367711
    
    fexp = np.dot(theta_transform, _notwiCoefficients.T) + _notwiIntercept
    return fexp


def _bright_exposure_factor_twi(airmass, sun_sep, sun_alt, wavelength=4500):
    ''' linear regression fit to exposure factor correction from the twilight
    contirbution given airmass and sun conditions 

    :param airmass: 
        array of airmasses
    :param sun_sep: 
        array of sun separations
    :param sun_alt: 
        array of sun altitudes 
    :param wavelength: 
        wavelength of the exposure factor (default: 4500) 
    :return fexp: 
        exposure factor twilight correction
    '''
    theta = np.atleast_2d(np.array([airmass, sun_sep, sun_alt]).T)
        
    if wavelength == 4500:
        _twiCoefficients = np.array([1.37980334, -0.00460065,  0.17337445])
        _twiIntercept = 2.217660156188111

    return np.dot(theta, _twiCoefficients.T) + _twiIntercept


def krisciunas_schaefer_free(obs_zenith, moon_zenith, separation_angle, moon_phase,
                        vband_extinction, C_R, C_M0, C_M1):
    """Calculate the scattered moonlight surface brightness in V band.

    Based on Krisciunas and Schaefer, "A model of the brightness of moonlight",
    PASP, vol. 103, Sept. 1991, p. 1033-1039 (http://dx.doi.org/10.1086/132921).
    Equation numbers in the code comments refer to this paper.

    The function :func:`plot_lunar_brightness` provides a convenient way to
    plot this model's predictions as a function of observation pointing.

    Units are required for the angular inputs and the result has units of
    surface brightness, for example:

    >>> sb = krisciunas_schaefer(20*u.deg, 70*u.deg, 50*u.deg, 0.25, 0.15)
    >>> print(np.round(sb, 3))
    19.855 mag / arcsec2

    The output is automatically broadcast over input arrays following the usual
    numpy rules.

    This method has several caveats but the authors find agreement with data at
    the 8% - 23% level.  See the paper for details.

    Parameters
    ----------
    obs_zenith : astropy.units.Quantity
        Zenith angle of the observation in angular units.
    moon_zenith : astropy.units.Quantity
        Zenith angle of the moon in angular units.
    separation_angle : astropy.units.Quantity
        Opening angle between the observation and moon in angular units.
    moon_phase : float
        Phase of the moon from 0.0 (full) to 1.0 (new), which can be calculated
        as abs((d / D) - 1) where d is the time since the last new moon
        and D = 29.5 days is the period between new moons.  The corresponding
        illumination fraction is ``0.5*(1 + cos(pi * moon_phase))``.
    vband_extinction : float
        V-band extinction coefficient to use.

    Returns
    -------
    astropy.units.Quantity
        Observed V-band surface brightness of scattered moonlight.
    """
    moon_phase = np.asarray(moon_phase)
    if np.any((moon_phase < 0) | (moon_phase > 1)):
        raise ValueError(
            'Invalid moon phase {0}. Expected 0-1.'.format(moon_phase))
    # Calculate the V-band magnitude of the moon (eqn. 9).
    abs_alpha = 180. * moon_phase
    m = -12.73 + 0.026 * abs_alpha + 4e-9 * abs_alpha ** 4
    # Calculate the illuminance of the moon outside the atmosphere in
    # foot-candles (eqn. 8).
    Istar = 10 ** (-0.4 * (m + 16.57))
    # Calculate the scattering function (eqn.21).
    rho = separation_angle.to(u.deg).value
    f_scatter = (C_R * (1.06 + np.cos(separation_angle) ** 2) +
                 10 ** (C_M0 - rho / C_M1))
    # Calculate the scattering airmass along the lines of sight to the
    # observation and moon (eqn. 3).
    X_obs = (1 - 0.96 * np.sin(obs_zenith) ** 2) ** (-0.5)
    X_moon = (1 - 0.96 * np.sin(moon_zenith) ** 2) ** (-0.5)
    # Calculate the V-band moon surface brightness in nanoLamberts.
    B_moon = (f_scatter * Istar *
        10 ** (-0.4 * vband_extinction * X_moon) *
        (1 - 10 ** (-0.4 * (vband_extinction * X_obs))))
    # Convert from nanoLamberts to to mag / arcsec**2 using eqn.19 of
    # Garstang, "Model for Artificial Night-Sky Illumination",
    # PASP, vol. 98, Mar. 1986, p. 364 (http://dx.doi.org/10.1086/131768)
    return ((20.7233 - np.log(B_moon / 34.08)) / 0.92104 *
            u.mag / (u.arcsec ** 2))


def exposure_factor(airmass, moon_frac, moon_sep, moon_alt, sun_sep, sun_alt): 
    """Calculate the exposure factor for specified observing conditions
    """
    # airmass exposure factor
    f_airmass = airmass_exposure_factor(airmass)
    
    # bright time exposure factor
    f_bright = bright_exposure_factor(airmass, moon_frac, moon_sep, moon_alt,
            sun_sep, sun_alt) 
    return f_airmass * f_bright


def exposure_time(program, seeing, transparency, airmass, EBV,
                  moon_frac, moon_sep, moon_alt):
    """Calculate the total exposure time for specified observing conditions.

    The exposure time is calculated as the time required under nominal
    conditions multiplied by factors to correct for actual vs nominal
    conditions for seeing, transparency, dust extinction, airmass, and
    scattered moon brightness.

    Note that this function returns the total exposure time required to
    achieve the target SNR**2 at current conditions.  The caller is responsible
    for adjusting this value when some signal has already been acummulated
    with previous exposures of a tile.

    Parameters
    ----------
    program : 'DARK', 'BRIGHT' or 'GRAY'
        Which program to use when setting the target SNR**2.
    seeing : float or array
        FWHM seeing value(s) in arcseconds.
    transparency : float or array
        Dimensionless transparency value(s) in the range [0-1].
    EBV : float or array
        Median dust extinction value(s) E(B-V) for the tile area.
    airmass : float
        Airmass used for observing this tile.
    moon_frac : float
        Illuminated fraction of the moon, between 0-1.
    moon_sep : float
        Separation angle between field center and moon in degrees.
    moon_alt : float
        Altitude angle of the moon above the horizon in degrees.

    Returns
    -------
    astropy.unit.Quantity
        Estimated exposure time(s) with time units.
    """
    # Lookup the nominal exposure time for this program.
    config = desisurvey.config.Configuration()
    nominal_time = getattr(config.nominal_exposure_time, program)()

    # Calculate actual / nominal factors.
    f_seeing = seeing_exposure_factor(seeing)
    f_transparency = transparency_exposure_factor(transparency)
    f_dust = dust_exposure_factor(EBV)
    f_airmass = airmass_exposure_factor(airmass)
    f_moon = moon_exposure_factor(moon_frac, moon_sep, moon_alt, airmass)

    # Calculate the exposure time required at the specified condtions.
    actual_time = nominal_time * (
        f_seeing * f_transparency * f_dust * f_airmass * f_moon)
    assert actual_time > 0 * u.s

    return actual_time


class ExposureTimeCalculator(object):
    """Online Exposure Time Calculator.

    Track observing conditions (seeing, transparency, sky background) during
    an exposure using the :meth:`start`, :meth:`update` and :meth:`stop`
    methods.

    Exposure time tracking is configured by the following parameters:
     - nominal_exposure_time
     - new_field_setup
     - same_field_setup
     - cosmic_ray_split
     - min_exposures

    Note that this version applies an average correction for the moon
    during the GRAY and BRIGHT programs, rather than idividual corrections
    based on the moon parameters.  This will be fixed in a future version.

    Parameters
    ----------
    save_history : bool
        When True, records the history of internal calculations during an
        exposure, for debugging and plotting.
    """
    def __init__(self, save_history=False):
        self._snr2frac = 0.
        self._exptime = 0.
        self._active = False
        self.tileid = None
        # Lookup config parameters (with times converted to days).
        config = desisurvey.config.Configuration()
        self.NEW_FIELD_SETUP = config.new_field_setup().to(u.day).value
        self.SAME_FIELD_SETUP = config.same_field_setup().to(u.day).value
        self.MAX_EXPTIME = config.cosmic_ray_split().to(u.day).value
        self.MIN_NEXP = config.min_exposures()
        self.TEXP_TOTAL = {}
        for program in desisurvey.tiles.Tiles.PROGRAMS:
            self.TEXP_TOTAL[program] = getattr(config.nominal_exposure_time, program)().to(u.day).value
        # Temporary hardcoded exposure factors for moon-up observing.
        #self.TEXP_TOTAL['GRAY'] *= 1.1
        #self.TEXP_TOTAL['BRIGHT'] *= 1.33

        # Initialize model of exposure time dependence on seeing.
        self.seeing_coefs = np.array([12.95475751, -7.10892892, 1.21068726])
        self.seeing_coefs /= np.sqrt(self.weather_factor(1.1, 1.0))
        assert np.allclose(self.weather_factor(1.1, 1.0), 1.)
        # Initialize optional history tracking.
        self.save_history = save_history
        if save_history:
            self.history = dict(mjd=[], signal=[], background=[], snr2frac=[])

    def weather_factor(self, seeing, transp):
        """Return the relative SNR2 accumulation rate for specified conditions.

        This is the inverse of the instantaneous exposure factor due to seeing and transparency.

        Parameters
        ----------
        seeing : float
            Atmospheric seeing in arcseconds.
        transp : float
            Atmospheric transparency in the range (0,1).
        """
        return transp * (self.seeing_coefs[0] + seeing * (self.seeing_coefs[1] + seeing * self.seeing_coefs[2])) ** 2

    def estimate_exposure(self, program, snr2frac, exposure_factor, nexp_completed=0):
        """Estimate exposure time(s).

        Can be used to estimate exposures for one or many tiles from the same program.

        Parameters
        ----------
        program : str
            Name of the program to estimate exposure times for. Used to determine the nominal
            exposure time. All tiles must be from the same program.
        snr2frac : float or array
            Fractional SNR2 integrated so far for the tile(s) to estimate.
        exposure_factor : float or array
            Exposure-time factor for the tile(s) to estimate.
        nexp_completed : int or array
            Number of exposures completed so far for tile(s) to estimate.
        
        Returns
        -------
        tuple
            Tuple (texp_total, texp_remaining, nexp) of floats or arrays, where
            texp_total is the total time that would be required under current conditions,
            texp_remaining is the remaining time under current conditions taking the
            already accumulated SNR2 into account, and nexp is the estimated number
            of remaining exposures required.
        """
        # Estimate total exposure time required under current conditions.
        texp_total = self.TEXP_TOTAL[program] * exposure_factor
        # Estimate time remaining to reach snr2frac = 1.
        texp_remaining = texp_total * (1 - snr2frac)
        # Estimate the number of exposures required.
        nexp = np.ceil(texp_remaining / self.MAX_EXPTIME).astype(int)
        nexp = np.maximum(nexp, self.MIN_NEXP - nexp_completed)
        return texp_total, texp_remaining, nexp

    def could_complete(self, t_remaining, program, snr2frac, exposure_factor):
        """Determine which tiles could be completed.

        Completion refers to achieving SNR2 = 1, which might require
        multiple exposures.

        Used by :meth:`desisurvey.scheduler.Scheduler.next_tile` and uses
        :meth:`estimate_exposure`.

        Parameters
        ----------
        t_remaining : float
            Time remaining in units of days.
        program : str
            Program that the candidate tiles belong to (must be the same for all tiles).
        snr2frac : float or array
            Fractional SNR2 integrated so far for each tile to consider.
        exposure_factor : float or array
            Exposure-time factor for each tile to consider.

        Returns
        -------
        array
            1D array of booleans indicating which tiles (if any) could completed
            within the remaining time.
        """
        texp_total, texp_remaining, nexp = self.estimate_exposure(program, snr2frac, exposure_factor)
        # Estimate total time required for all exposures.
        t_required = self.NEW_FIELD_SETUP + self.SAME_FIELD_SETUP * (nexp - 1) + texp_remaining
        return t_required <= t_remaining

    def start(self, mjd_now, tileid, program, snr2frac, exposure_factor, seeing, transp, sky):
        """Start tracking an exposure.

        Must be called before using :meth:`update` to track changing conditions
        during the exposure.

        Parameters
        ----------
        mjd_now : float
            MJD timestamp when exposure starts.
        tileid : int
            ID of the tile being exposed. This is only used to recognize consecutive
            exposures of the same tile.
        program : str
            Name of the program the exposed tile belongs to.
        snr2frac : float
            Previous accumulated fractional SNR2 of the exposed tile.
        exposure_factor : float
            Exposure factor of the tile when the exposure starts, based on the
            current conditions specified by the remaining parameters.
        seeing : float
            Initial atmospheric seeing in arcseconds.
        transp : float
            Initial atmospheric transparency (0,1).
        sky : float
            Initial sky background level.
        """
        self.mjd_start = mjd_now
        self._snr2frac = self._snr2frac_start = snr2frac
        self.mjd_last = mjd_now
        self.mjd_start = mjd_now
        self._active = True
        if tileid == self.tileid:
            self.tile_nexp += 1
        else:
            self.tile_nexp = 1
        self.tileid = tileid
        self.texp_total, texp_remaining, nexp = self.estimate_exposure(
            program, snr2frac, exposure_factor, self.tile_nexp - 1)
        # Estimate SNR2 to integrate in the next exposure.
        self.snr2frac_target = snr2frac + (texp_remaining / nexp) / self.texp_total
        # Initialize signal and background rate factors.
        self.srate0 = self.weather_factor(seeing, transp)
        self.brate0 = sky
        self.signal = 0.
        self.background = 0.
        self.last_snr2frac = 0.
        self.should_abort = False
        if self.save_history:
            self.history['mjd'].append(mjd_now)
            self.history['signal'].append(0.)
            self.history['background'].append(0.)
            self.history['snr2frac'].append(snr2frac)
    
    def update(self, mjd_now, seeing, transp, sky):
        """Track changing conditions during an exposure.

        Must call :meth:`start` first to start tracking an exposure.

        Parameters
        ----------
        mjd_now : float
            Current MJD timestamp.
        seeing : float
            Estimate of average atmospheric seeing in arcseconds
            since last update (or start).
        transp : float
            Estimate of average atmospheric transparency
            since last update (or start).
        sky : float
            Estimate of average sky background level
            since last update (or start).

        Returns
        -------
        bool
            True if the exposure should continue integrating.
        """
        dt = mjd_now - self.mjd_last
        self.mjd_last = mjd_now
        srate = self.weather_factor(seeing, transp)
        brate = sky        
        self.signal += dt * srate / self.srate0
        #self.background += dt * (srate + brate) / (self.srate0 + self.brate0)
        self.background += dt * brate / self.brate0
        self._snr2frac = self._snr2frac_start + self.signal ** 2 / self.background / self.texp_total            
        if self.save_history:
            self.history['mjd'].append(mjd_now)
            self.history['signal'].append(self.signal)
            self.history['background'].append(self.background)
            self.history['snr2frac'].append(self._snr2frac)
        need_more_snr = self._snr2frac < self.snr2frac_target
        # Give up on this tile if SNR progress has dropped significantly since we started.
        self.should_abort = (self._snr2frac - self.last_snr2frac) / dt < 0.25 / self.texp_total
        self.last_snr2frac = self._snr2frac
        return need_more_snr and not self.should_abort

    def stop(self, mjd_now):
        """Stop tracking an exposure.

        After calling this method, use :attr:`exptime` to look up the exposure time.

        Parameters
        ----------
        mjd_now : float
            MJD timestamp when the current exposure was stopped.

        Returns
        -------
        bool
            True if this tile is "done" or False if another exposure of the
            same tile should be started immediately.  Note that "done" normally
            means the tile has reached its target SNR2, but could also mean
            that the SNR2 accumulation rate has fallen below some threshold
            so that it is no longer useful to continue exposing.
        """
        self._exptime = mjd_now - self.mjd_start
        self._active = False
        return self._snr2frac >= 1 or self.should_abort

    @property
    def snr2frac(self):
        """Integrated fractional SNR2 of tile currently being exposed.

        Includes signal accumulated in previous exposures. Initialized by
        :meth:`start`, updated by :meth:`update` and frozen by :meth:`stop`.
        """
        return self._snr2frac

    @property
    def exptime(self):
        """Exposure time in days recorded by last call to :meth:`stop`.
        """
        return self._exptime

    @property
    def active(self):
        """Are we tracking an exposure?

        Set True by :meth:`start` and False by :meth:`stop`.
        """
        return self._active
