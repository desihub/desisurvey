"""Calculate the nominal exposure time for specified observing conditions.

Use :func:`exposure_time` to combine all effects into an exposure time in
seconds, or call functions to calculate the individual exposure-time factors
associated with each effect.

The following effects are included: seeing, transparency, galactic dust
extinction, airmass, scattered moonlight.  The following effects are not yet
implemented: twilight sky brightness, clouds, variable OH sky brightness.
"""
from __future__ import print_function, division

import numpy as np

from itertools import chain, combinations_with_replacement

import astropy.units as u

import specsim.atmosphere

import desiutil.log

import desisurvey.config
import desisurvey.tiles


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
    V = specsim.atmosphere.krisciunas_schaefer(
        obs_zenith, moon_zenith, separation_angle,
        moon_phase, _vband_extinction).value

    # Evaluate the linear regression model.
    X = np.array((1, np.exp(-V), 1/V, 1/V**2, 1/V**3))
    return _moonCoefficients.dot(X)


def sky_exposure_factor(tref, airmass, moon_frac, moon_sep, moon_alt, sun_sep,
        sun_alt):
    """ Calculate exposure time scaling factor due to sky brightness. This is
    the factor that exposure time needs to be scaled by in order to get the
    same SNR as an exposure during nominal dark sky. It's calculated based on a
    sky brightness model fit to observed sky brightness from DESI SV1, DESI
    CMX, and BOSS. 

    For details, see the following jupyter notebooks: 
    - https://github.com/desi-bgs/bgs-cmxsv/blob/55850e44d65570da69de0788c652cff698416834/doc/nb/sv1_sky_model_fit.ipynb
    - https://github.com/desi-bgs/bgs-cmxsv/blob/55850e44d65570da69de0788c652cff698416834/doc/nb/sky_model_twilight_fit.ipynb

    Parameters
    ----------
    tnom : array 
        Array of nominal exposure time 
    airmass : array 
        Array of airmass used for observing this tile, must be >= 1.
    moon_frac : float 
        Illuminated fraction of the moon within range [0,1].
    moon_sep : array
        Array of separation angles between field center and moon in degrees within the
        range [0,180].
    moon_alt : float 
        Altitude angle of the moon above the horizon in degrees within range [-90,90].
    sun_sep : array 
        Arry of separations angles between field center and sun in degrees
    sun_alt : float 
        Altitude angle of the sunin degrees

    Returns
    -------
    array
        Dimensionless factors that exposure time should be increased to
        account for increased sky brightness during bright time.
    """
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

    # calculate exposure factor 
    config = desisurvey.config.Configuration()

    # sky brightness at 5000A for nominal dark sky 
    Isky5000_ref = 1.1282850428182252 # 1e-17 erg/s/cm^2/A/arcsec^2
    
    # exposure time for nominal exposure 
    tref = getattr(config.nominal_exposure_time, 'BRIGHT')().to(u.s).value
    
    # sky brightness at 5000A for observing conditions 
    Isky5000_exp = Isky5000_notwilight_regression(
            airmass,
            np.repeat(moon_frac, nexp), 
            moon_sep, 
            np.repeat(moon_alt, nexp))

     # add twilight contribution 
    if sun_alt >= -18.:
        Isky5000_exp += np.clip(Isky5000_twilight_regression(
                airmass, 
                sun_sep, 
                np.repeat(sun_alt, nexp)), 0, None) 
    
    # calculate exposure factor 
    fibflux5000_ref = Isky5000_ref * 1.862089 # 1e-17 erg/s/cm^2/A
    fibflux5000_exp = Isky5000_exp * 1.862089 # 1e-17 erg/s/cm^2/A
    
    # 0.0629735016982807 = 1e-17 x (photons per bin) x throughput) at 5000A 
    sky_photon_per_sec_ref = fibflux5000_ref * 0.0629735016982807
    sky_photon_per_sec_exp = fibflux5000_exp * 0.0629735016982807
    
    # (read noise)^2 at 5000A 
    RNsq5000 = 56.329457658891435 # photon^2 

    # solve the following: 
    # S x tref / sqrt(sky_ref * tref + RN^2) = S x texp / sqrt(sky_exp * texp + RN^2)
    texp = 0.5 * (
            (tref * np.sqrt(4 * sky_photon_per_sec_ref * RNsq5000 * tref + sky_photon_per_sec_exp**2 * tref**2 + 4 * RNsq5000**2))/(sky_photon_per_sec_ref * tref + RNsq5000) +
            (sky_photon_per_sec_exp * tref**2)/(sky_photon_per_sec_ref * tref + RNsq5000))
    return texp/tref 


def Isky5000_notwilight_regression(airmass, moon_frac, moon_sep, moon_alt): 
    ''' Calculate sky surface brightness at 5000A during bright time without
    twilight. 

    Surface brightness is based on a polynomial regression model fit to
    log(observed sky brightness). The observed sky brightnesses were compiled
    from 
    * DESI SV1 bright exposures with TRANSP > 0.9 and SUN_ALT < -18 
    * DESI CMX exposures with transparency > 0.95
    * BOSS exposures with sun alt < -18 
    
    For details, see jupyter notebook: 
    https://github.com/desi-bgs/bgs-cmxsv/blob/521211dccff7cb3b71b07d84522802acade26071/doc/nb/sv1_sky_model_fit.ipynb

    Parameters
    ----------
    airmass : array 
        Array of airmass used for observing this tile, must be >= 1.
    moon_frac : array 
        Array of illuminated fraction of the moon within range [0,1].
    moon_sep : array
        Array of separation angles between field center and moon in degrees within the
        range [0,180].
    moon_alt : array
        Array of altitude angle of the moon above the horizon in degrees within
        range [-90,90].

    Returns
    -------
    array
        Array of sky surface brightness at 5000A for bright time without
        twilight in units of erg/s/cm^2/A/arcsec^2
    '''
    # polynomial of order
    norder = 5 
    # polynomial regression cofficients for estimating exposure time factor during
    # non-twilight from airmass, moon_frac, moon_sep, moon_alt  
    coeffs = np.array([ 
        1.19728287e-02,  2.68146174e-02,  2.78840941e-02,  4.41723544e-03,
       -1.02111676e-01,  7.65110151e-03,  1.63878220e-02, -3.08958375e-02,
        3.85657345e-01,  6.83474920e-03,  3.06045766e-01,  2.55798622e-01,
        1.19306254e-03, -1.04242412e-02, -3.34780689e-03, -1.68835833e-02,
        1.11378109e-02, -1.75611187e-01, -2.38751513e-01,  1.46593863e-02,
       -4.36980545e-02, -1.78128533e-01,  5.85014404e-03,  1.37745149e-02,
       -1.81392009e-03, -1.27115538e-02,  9.35953790e-02, -1.52422946e-01,
       -1.74469624e-04, -6.01814594e-03, -2.13820870e-04, -7.40115962e-05,
       -4.43960687e-05,  1.31544250e-04,  5.23119315e-05, -3.07378710e-02,
        3.50689477e-02,  1.85015093e-01,  6.13788213e-02,  1.98622649e-02,
       -7.03566704e-02, -2.70010457e-01, -4.13256930e-03, -4.65235557e-03,
        2.08199644e-03,  4.06645589e-03,  2.08415057e-01,  4.87587825e-01,
       -7.34996335e-03,  1.37774943e-03,  6.26446800e-03,  2.14264776e-05,
       -1.34520777e-05, -1.28680576e-04, -2.98020923e-05,  1.41853618e-03,
       -2.22111586e-01, -5.77624465e-01, -1.09989601e-02, -2.77969195e-04,
        2.17332386e-03,  1.47785979e-04,  1.63001051e-04,  3.10888811e-05,
       -5.98993375e-05, -8.54879999e-08,  3.29709501e-07,  2.92447174e-08,
       -2.79763418e-07, -2.38343236e-08, -1.10769335e-01,  8.24805309e-02,
       -2.67759016e-02, -1.09365333e-03,  9.87385635e-03,  3.12620536e-02,
        9.15874867e-02, -2.52374361e-05, -4.92100655e-04, -4.21467759e-04,
        5.28695530e-03,  1.93471362e-01, -5.37647583e-02, -4.34387094e-04,
       -2.25656833e-03, -7.15039182e-04,  1.18896366e-05,  2.77215519e-05,
        2.81214300e-05, -2.48811976e-06,  1.29700918e-02, -1.92447646e-02,
       -2.44212957e-01, -5.99603226e-03, -2.76089664e-03,  9.14612070e-04,
        6.91296852e-05,  1.02360578e-04,  1.16371717e-05, -2.52971480e-05,
       -1.37227312e-07, -3.75516901e-07,  5.48164999e-08,  5.06994020e-07,
        1.93984829e-07,  5.32931890e-02,  5.79701139e-01,  4.34770814e-01,
       -6.79348983e-03, -1.31426633e-02, -6.14432600e-05,  1.39891173e-04,
        3.29198340e-04,  1.11812563e-04, -1.17664897e-05, -1.29847018e-06,
       -3.91089212e-06, -2.85449951e-06, -3.27679136e-07,  3.48519396e-07,
        3.29795490e-09,  7.65512533e-09,  6.37102357e-09,  5.95568585e-10,
       -2.63641980e-09, -1.30434882e-09])
    
    theta = np.atleast_2d(np.array([airmass, moon_frac, moon_alt, moon_sep]).T)

    combs = chain.from_iterable(combinations_with_replacement(range(4), i) for i in range(0, norder+1))

    theta_transform = np.empty((theta.shape[0], len(coeffs)))
    for i, comb in enumerate(combs):
        theta_transform[:, i] = theta[:, comb].prod(1)

    return np.exp(np.dot(theta_transform, coeffs.T)) 


def Isky5000_twilight_regression(airmass, sun_sep, sun_alt): 
    ''' Calculate twilight contribution to sky surface brightness at 5000A
    during bright time. 
    
    Surface brightness is based on a polynomial regression model that was fit
    to twilight contributions measured from SV1 exposures with sun alt > -18
    and TRANSP > 0.95 and BOSS exposures with sun altitutde > -18deg. For
    details, see jupyter notebook: 
    https://github.com/desi-bgs/bgs-cmxsv/blob/521211dccff7cb3b71b07d84522802acade26071/doc/nb/sky_model_twilight_fit.ipynb
    
    Parameters
    ----------
    airmass : array 
        Array of airmass used for observing this tile, must be >= 1.
    sun_sep : array 
        Arry of separations angles between field center and sun in degrees
    sun_alt : float 
        Altitude angle of the sun in degrees

    Returns
    -------
    array
        Array of twilight contribution to sky surface brightness at 5000A for
        bright time in units of erg/s/cm^2/A/arcsec^2
    '''
    norder = 1
    # coefficients fit in notebook
    # https://github.com/desi-bgs/bgs-cmxsv/blob/521211dccff7cb3b71b07d84522802acade26071/doc/nb/sky_model_twilight_fit.ipynb
    skymodel_coeff = np.array([3.50300842, 2.63453598, 0.40491282, 0.0096106 ])

    theta = np.atleast_2d(np.array([airmass, sun_alt, sun_sep]).T)

    combs = chain.from_iterable(combinations_with_replacement(range(3), i) for i in range(0, norder+1))
    theta_transform = np.empty((theta.shape[0], len(skymodel_coeff)))
    for i, comb in enumerate(combs):
        theta_transform[:, i] = theta[:, comb].prod(1)

    return np.clip(np.dot(theta_transform, skymodel_coeff.T), 0, None) 


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
        self.log = desiutil.log.get_logger()
        unknownprograms = []
        for program in desisurvey.tiles.Tiles.PROGRAMS:
            nomtime = getattr(config.nominal_exposure_time, program, None)
            if nomtime is None:
                unknownprograms.append(program)
                nomtime = 1000/24/60/60
            else:
                nomtime = nomtime().to(u.day).value
            self.TEXP_TOTAL[program] = nomtime
        if len(unknownprograms) > 0:
            self.log.warning(
                'Unrecognized program {}, '.format(' '.join(unknownprograms)) +
                'using default exposure time of 1000 s')
        moon_up_factor = getattr(config, 'moon_up_factor', None)
        if moon_up_factor:
            for cond in ['DARK', 'GRAY']:
                self.TEXP_TOTAL[cond] *= getattr(moon_up_factor, cond)()
        else:
            # Temporary hardcoded exposure factors for moon-up observing.
            self.TEXP_TOTAL['GRAY'] *= 1.1

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
