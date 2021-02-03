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
_coeff_150 = np.array([ 0.00000000e+00, -2.49708184e-02,  2.52486199e-01, -5.46363860e-02,
    -3.46385860e-02, -8.36386831e-02,  1.30105577e-01,  9.66533278e-03,
    -1.57508886e-02,  3.28351095e-01, -1.42550850e-02,  6.94343483e-02,
    6.09900406e-04,  7.10071258e-04,  6.99453704e-06,  5.68048827e-02,
    -8.17988318e-02, -1.02939149e-02,  1.95714560e-03,  4.73877963e-01,
    6.16920067e-03, -1.95719175e-02,  1.12020673e-04,  2.91442987e-04,
    3.32396877e-04, -8.34800862e-02, -1.98454417e-02,  6.46925683e-03,
    1.37811319e-04, -3.58565949e-04, -1.98283765e-04, -3.10800386e-06,
    -6.15677754e-06, -5.05207439e-06, -3.90575869e-07])
_coeff_170 = np.array([ 0.00000000e+00, -2.80592741e-02,  2.82519650e-01, -6.12337075e-02,
    -3.88977732e-02, -9.38911644e-02,  1.45640129e-01,  1.08522543e-02,
    -1.77011318e-02,  3.69075912e-01, -1.59220598e-02,  7.78362001e-02,
    6.82898179e-04,  7.97210414e-04,  9.66859468e-06,  6.36858560e-02,
    -9.14042425e-02, -1.15491329e-02,  2.24059564e-03,  5.32649118e-01,
    6.89977186e-03, -2.19650156e-02,  1.25688485e-04,  3.26261642e-04,
    3.72266191e-04, -9.30181217e-02, -2.22376944e-02,  7.12176891e-03,
    1.53774966e-04, -4.01348718e-04, -2.20093665e-04, -3.47892160e-06,
    -6.90089062e-06, -5.68062255e-06, -4.52580016e-07])
_coeff_200 = np.array([ 0.00000000e+00, -3.25370591e-02,  3.24502646e-01, -7.06229367e-02,
    -4.48817655e-02, -1.08639662e-01,  1.67330251e-01,  1.26047574e-02,
    -2.06369392e-02,  4.27089974e-01, -1.82413474e-02,  8.98795572e-02,
    7.85960945e-04,  9.21368404e-04,  1.34633923e-05,  7.32368512e-02,
    -1.04924141e-01, -1.33399319e-02,  2.74422079e-03,  6.16555294e-01,
    7.95012256e-03, -2.54404656e-02,  1.44795563e-04,  3.74838366e-04,
    4.28923997e-04, -1.07082915e-01, -2.56369357e-02,  7.97087025e-03,
    1.75981316e-04, -4.61519409e-04, -2.50772895e-04, -4.00060046e-06,
    -7.95325622e-06, -6.57333165e-06, -5.42867923e-07])
_coeff_220 = np.array([ 0.00000000e+00, -3.54090517e-02,  3.50535274e-01, -7.65461292e-02,
    -4.86059816e-02, -1.18034491e-01,  1.80761231e-01,  1.37481237e-02,
    -2.25858176e-02,  4.63677195e-01, -1.96728246e-02,  9.75288132e-02,
    8.50491083e-04,  9.99727418e-04,  1.57829882e-05,  7.91051191e-02,
    -1.13375346e-01, -1.44713509e-02,  3.12220581e-03,  6.69585634e-01,
    8.61955423e-03, -2.76741049e-02,  1.56627998e-04,  4.04884075e-04,
    4.64616282e-04, -1.16288416e-01, -2.77776118e-02,  8.45958480e-03,
    1.89694119e-04, -4.99036366e-04, -2.69974746e-04, -4.32601284e-06,
    -8.61296332e-06, -7.13458865e-06, -6.00526979e-07])
_coeff_250 = np.array([ 0.00000000e+00, -3.95359759e-02,  3.86855546e-01, -8.49418723e-02,
    -5.38149517e-02, -1.31459562e-01,  1.99474452e-01,  1.54171009e-02,
    -2.54726885e-02,  5.15464928e-01, -2.16613537e-02,  1.08436519e-01,
    9.41343709e-04,  1.11078207e-03,  1.88972724e-05,  8.72069228e-02,
    -1.25262911e-01, -1.60763023e-02,  3.73514437e-03,  7.44801918e-01,
    9.57735146e-03, -3.08917481e-02,  1.73102900e-04,  4.46689897e-04,
    5.15142065e-04, -1.29809522e-01, -3.08064535e-02,  9.09387478e-03,
    2.08767603e-04, -5.51634780e-04, -2.97076724e-04, -4.78259856e-06,
    -9.54251416e-06, -7.92673637e-06, -6.82507628e-07])
_coeff_270 = np.array([ 0.00000000e+00, -4.21633435e-02,  4.09380442e-01, -9.02263000e-02,
    -5.70512472e-02, -1.39967512e-01,  2.11065329e-01,  1.64954804e-02,
    -2.73615473e-02,  5.47991633e-01, -2.28895878e-02,  1.15338990e-01,
    9.98176958e-04,  1.18064967e-03,  2.07237842e-05,  9.21751932e-02,
    -1.32693266e-01, -1.70868888e-02,  4.16532521e-03,  7.92137409e-01,
    1.01853143e-02, -3.29461547e-02,  1.83297027e-04,  4.72546452e-04,
    5.46905287e-04, -1.38612904e-01, -3.27094559e-02,  9.46059035e-03,
    2.20571012e-04, -5.84401967e-04, -3.14105220e-04, -5.06732555e-06,
    -1.01243264e-05, -8.42295696e-06, -7.33962180e-07])
_coeff_300 = np.array([ 0.00000000e+00, -4.59178821e-02,  4.40857124e-01, -9.77103513e-02,
    -6.15791014e-02, -1.52082391e-01,  2.27245400e-01,  1.80577744e-02,
    -3.01264231e-02,  5.93932528e-01, -2.45997804e-02,  1.25159593e-01,
    1.07823168e-03,  1.27952933e-03,  2.31002043e-05,  9.90392788e-02,
    -1.43149683e-01, -1.85183472e-02,  4.82986528e-03,  8.59119827e-01,
    1.10525881e-02, -3.58913747e-02,  1.97510818e-04,  5.08587660e-04,
    5.91840950e-04, -1.51472896e-01, -3.53999993e-02,  9.94136614e-03,
    2.37046497e-04, -6.30379294e-04, -3.38234224e-04, -5.46729862e-06,
    -1.09441607e-05, -9.12223323e-06, -8.06264099e-07])


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
    inters = [2.5862319238450633, 2.777513688414931, 3.0493419627360243,
            3.2206029229505133, 3.4630959513567374, 3.6155886844539435, 
            3.8314019756655826]
    
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
