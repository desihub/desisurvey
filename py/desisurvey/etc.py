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
    if moon_alt < 0 and sun_alt < -18.:
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

    if sun_alt >= -18.: 
        # w/ twilight contribution
        exp_factors += _bright_exposure_factor_twi(
                airmass, 
                sun_sep, 
                np.repeat(sun_alt, nexp))
    return np.clip(exp_factors, 1., None) 


# polynomial regression cofficients for estimating exposure time factor during
# non-twilight from airmass, moon_frac, moon_sep, moon_alt  
_coeff_150 = np.array([
    0.00000000e+00, -7.65192309e-02,  4.50339775e-01, -1.97942832e-02,
    -3.41415322e-02, -6.12718783e-01,  2.92105504e-01,  6.35543300e-03,
    4.33218913e-02,  4.52094789e-02, -9.84060307e-03, -7.08736998e-03,
    1.87100312e-04,  1.29286060e-04,  2.29783430e-04,  4.82259115e-01,
    1.98321998e-01, -1.93484942e-02, -2.97180320e-02,  6.69777743e-02,
    -4.61085961e-03,  3.41750827e-02,  3.30879455e-04,  3.81656037e-04,
    9.06394046e-05,  1.49819306e-02, -1.36399405e-03,  1.96793987e-02,
    8.62206556e-05, -4.98643387e-04, -3.98201905e-04, -2.55149678e-06,
    -2.84521671e-06, -2.17588659e-06, -1.22082693e-06])
_coeff_170 = np.array([
    0.00000000e+00, -8.47161918e-02,  5.09140318e-01, -2.21689858e-02,
    -3.82829563e-02, -6.82547805e-01,  3.30443697e-01,  6.96121973e-03,
    4.85970310e-02,  5.16826651e-02, -1.11869516e-02, -7.84891835e-03,
    2.11236410e-04,  1.44634107e-04,  2.56482375e-04,  5.39172275e-01,
    2.17955039e-01, -2.16472707e-02, -3.33864806e-02,  7.37805327e-02,
    -5.05383302e-03,  3.82830633e-02,  3.70874089e-04,  4.29673186e-04,
    1.02787431e-04,  1.72136448e-02, -1.50046818e-03,  2.18334958e-02,
    9.65000425e-05, -5.57451902e-04, -4.44714387e-04, -2.86562898e-06,
    -3.20549339e-06, -2.45240669e-06, -1.37286608e-06]) 
_coeff_200 = np.array([ 
    0.00000000e+00, -9.58679635e-02,  5.93841512e-01, -2.55861615e-02,
    -4.41591474e-02, -7.79678593e-01,  3.85818939e-01,  7.79679323e-03,
    5.59921835e-02,  6.11104223e-02, -1.31110611e-02, -8.79745360e-03,
    2.46048016e-04,  1.67042274e-04,  2.94236583e-04,  6.18672422e-01,
    2.44202035e-01, -2.48791060e-02, -3.85417472e-02,  8.30483197e-02,
    -5.63610422e-03,  4.40148404e-02,  4.27190211e-04,  4.97703948e-04,
    1.20209680e-04,  1.93038858e-02, -1.68032918e-03,  2.47257673e-02,
    1.10776356e-04, -6.39467548e-04, -5.09258980e-04, -3.31065269e-06,
    -3.72136838e-06, -2.85115773e-06, -1.59214888e-06])
_coeff_220 = np.array([
    0.00000000e+00, -1.02634030e-01,  6.47694841e-01, -2.77657185e-02,
    -4.78539584e-02, -8.39732104e-01,  4.21104384e-01,  8.31458698e-03,
    6.05850892e-02,  6.71532006e-02, -1.43223633e-02, -9.31516873e-03,
    2.68225131e-04,  1.81542250e-04,  3.17943153e-04,  6.67947574e-01,
    2.59892026e-01, -2.68947718e-02, -4.17491712e-02,  8.87103338e-02,
    -5.97815265e-03,  4.75607734e-02,  4.62337315e-04,  5.40334638e-04,
    1.31226062e-04,  1.99295303e-02, -1.78664043e-03,  2.64513742e-02,
    1.19566281e-04, -6.90228023e-04, -5.49024884e-04, -3.58983421e-06,
    -4.04795876e-06, -3.10523364e-06, -1.73200204e-06]) 
_coeff_250 = np.array([ 
    0.00000000e+00, -1.11912051e-01,  7.24347149e-01, -3.08864148e-02,
    -5.30738429e-02, -9.23417456e-01,  4.71405783e-01,  9.04313900e-03,
    6.69974746e-02,  7.58053205e-02, -1.60278115e-02, -9.94667623e-03,
    2.99868335e-04,  2.02584664e-04,  3.51440427e-04,  7.36685710e-01,
    2.81171709e-01, -2.97227802e-02, -4.62324288e-02,  9.65747399e-02,
    -6.43403705e-03,  5.24964865e-02,  5.11648523e-04,  6.00281853e-04,
    1.46814556e-04,  1.98955305e-02, -1.93034573e-03,  2.87764134e-02,
    1.31742822e-04, -7.60961220e-04, -6.04217621e-04, -3.98317550e-06,
    -4.51152222e-06, -3.46790508e-06, -1.93190291e-06])
_coeff_270 = np.array([ 
    0.00000000e+00, -1.17579680e-01,  7.72686098e-01, -3.28687496e-02,
    -5.63486867e-02, -9.75308571e-01,  5.03159813e-01,  9.50093595e-03,
    7.09752381e-02,  8.12876610e-02, -1.70912364e-02, -1.02857976e-02,
    3.19877375e-04,  2.16119447e-04,  3.72477413e-04,  7.79317910e-01,
    2.94068329e-01, -3.14861601e-02, -4.90157292e-02,  1.01463331e-01,
    -6.70544226e-03,  5.55508404e-02,  5.42384271e-04,  6.37693165e-04,
    1.56587555e-04,  1.93351377e-02, -2.01786152e-03,  3.01726857e-02,
    1.39241535e-04, -8.04798184e-04, -6.38300921e-04, -4.22922199e-06,
    -4.80332352e-06, -3.69734761e-06, -2.05856404e-06])
_coeff_300 = np.array([
    0.00000000e+00, -1.25394980e-01,  8.41149934e-01, -3.56986952e-02,
    -6.09725109e-02, -1.04787098e+00,  5.48156473e-01,  1.01508052e-02,
    7.65335799e-02,  8.90819550e-02, -1.85806736e-02, -1.06941027e-02,
    3.48296663e-04,  2.35659273e-04,  4.02228481e-04,  8.38907583e-01,
    3.11773061e-01, -3.39628218e-02, -5.29068606e-02,  1.08348195e-01,
    -7.07151143e-03,  5.98111877e-02,  5.85524975e-04,  6.90223878e-04,
    1.70352797e-04,  1.78631701e-02, -2.13966070e-03,  3.20694562e-02,
    1.49651454e-04, -8.66050659e-04, -6.85776304e-04, -4.57557932e-06,
    -5.21619748e-06, -4.02340355e-06, -2.23884453e-06])


def _bright_exposure_factor_notwi(airmass, moon_frac, moon_sep, moon_alt): 
    ''' third degree polynomial regression fit to exposure factor of  
    non-twilight bright sky given airmass and moon_conditions. Exposure factor
    is calculated from the ratio of (sky brightness)/(nominal dark sky
    brightness) at 7000A. The coefficients are fit to DESI CMX and BOSS sky
    surface brightness. See
    https://github.com/changhoonhahn/feasiBGS/blob/97524545ad98df34c934d777f98761c5aea6a4c5/notebook/cmx/exposure_factor_refit.ipynb
    for details. 

    :param airmass: 
        array of airmasses
    :param moon_frac: 
        array of moon illumination fractions
    :param moon_sep: 
        array of moon separations
    :param moon_alt: 
        array of moon altitudes 
    :return fexp: 
        exposure factor for non-twlight bright sky 
    '''
    config = desisurvey.config.Configuration()
    tnom = getattr(config.nominal_exposure_time, 'BRIGHT')().to(u.s).value
    #print('nominal exposure time %.f' % tnom)
    assert tnom <= 300, 'BGS nominal exposure time longer than 300s not supported'

    tnoms = np.array([150, 170, 200, 220, 250, 270, 300])
    coeffs = [_coeff_150, _coeff_170, _coeff_200, _coeff_220, _coeff_250,
            _coeff_270, _coeff_300]
    inters = [1.8711798724721023, 1.9732660265121995, 2.1185346942680368,
            2.2103067178854525, 2.3407011847663415, 2.4230117145582475, 
            2.539948136676698]
    
    i_tnom = np.arange(len(tnoms))[(tnoms - tnom) >= 0][0]
    #print('we will use nominal exposure time %.f' % tnoms[i_tnom])

    notwiCoeff = coeffs[i_tnom]
    notwiInter = inters[i_tnom]
    
    theta = np.atleast_2d(np.array([airmass, moon_frac, moon_sep, moon_alt]).T)

    combs = chain.from_iterable(combinations_with_replacement(range(4), i) for i in range(0, 4))

    theta_transform = np.empty((theta.shape[0], len(notwiCoeff)))
    for i, comb in enumerate(combs):
        theta_transform[:, i] = theta[:, comb].prod(1)

    fexp = np.dot(theta_transform, notwiCoeff.T) + notwiInter
    return fexp


def _bright_exposure_factor_twi(airmass, sun_sep, sun_alt):
    ''' linear regression fit to exposure factor correction contribution from
    the twilight given airmass and sun conditions. 

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
        
    _twiCoefficients = np.array([1.1139712, -0.00431072, 0.16183842]) 
    _twiIntercept = 2.3278959318651733

    return np.dot(theta, _twiCoefficients.T) + _twiIntercept


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
