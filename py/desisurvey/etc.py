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

import astropy.time
import astropy.units as u
from astropy.coordinates import SkyCoord

import specsim.atmosphere

import desiutil.log

import desisurvey.config
import desisurvey.tiles
import desisurvey.ephem


def seeing_exposure_factor(seeing, sbprof='ELG'):
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

    sbprof: str
        source profile to use, one of PSF, ELG, BGS

    Returns
    -------
    float
        Multiplicative factor(s) that exposure time should be adjusted based
        on the actual vs nominal seeing.
    """
    if sbprof == 'FLT':
        return 1.0 + seeing*0
    if np.any(seeing <= 0):
        raise ValueError('Got invalid seeing value <= 0.')
    polydict = getattr(seeing_exposure_factor, 'polydict', None)
    if polydict is None:
        polydict = dict(
            PSF=np.poly1d([0.09886370, -0.55877988, -0.97075602, -0.44728180]),
            ELG=np.poly1d([0.02306297, -0.42495946, -0.82527905, -0.77608006]),
            BGS=np.poly1d([0.03412768, -0.36108102, -0.71753377, -1.56430281]),
            FLT=None)
        config = desisurvey.config.Configuration()
        nomseeing = config.nominal_conditions.seeing().to(u.arcsec).value
        for name in list(polydict.keys()):
            if polydict[name] is None:
                continue
            ffracnom = np.exp(polydict[name](np.log(nomseeing)))
            polydict[name+'norm'] = ffracnom
        seeing_exposure_factor.polydict = polydict
    poly = polydict[sbprof]
    seeing = np.clip(seeing, 0.5, 3.5)
    norm = polydict[sbprof+'norm']
    ffrac = np.exp(poly(np.log(seeing)))/norm
    return ffrac**(-2)


def transparency_exposure_factor(transparency):
    """Scaling of exposure time with transparency relative to nominal.

    The model is that exposure time scales with 1 / transparency**2.

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
    return (nominal / transparency)**2


def dust_exposure_factor(EBV):
    """Scaling of exposure time with median E(B-V) relative to nominal.

    The model uses the SDSS-g extinction coefficient (3.303) from Table 6
    of Schlafly & Finkbeiner 2011 by default, or config.ebv_coefficient if
    specified.

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
    coeff = getattr(config, 'ebv_coefficient', None)
    if coeff is not None:
        coeff = coeff()
    else:
        coeff = 3.303
    Ag = coeff * (EBV - EBV0)
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
    return np.power((X / X0), 1.75)


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


def sky_level(mjd, ra, dec, moon_ill=None, moon_DECRA=None, moon_ALTAZ=None, sun_DECRA=None, sun_ALTAZ=None): 
    ''' Calculate the sky level. Sky level is defined as sky surface brightness
    for given MJD, RA, and Dec over nominal sky sufrace brightness 

    Parameters
    ----------
    mjd : float
        date 
    ra : float 
        RA in deg
    dec : float
        Dec in degrees deg
    moon_ill : float, optional
        moon illumination 
    moon_DECRA : optional
        moon Dec and RA interpolator object from desisurvey.ephem
    moon_ALTAZ : optional
        moon Alt and AZ interpolator object from desisurvey.ephem
    sun_DECRA : optional
        sun Dec and RA interpolator object from desisurvey.ephem
    sun_ALTAZ : optional
        sun Alt and AZ interpolator object from desisurvey.ephem

    Returns
    -------
    sky level : float 
        current sky brightness over nominal sky brightness
    '''
    if moon_DECRA is None or moon_ALTAZ is None or sun_DECRA is None or sun_ALTAZ is None:
        ephem = desisurvey.ephem.get_ephem()
        night = desisurvey.utils.get_date(mjd) 
        night_ephem = ephem.get_night(night)

        # calculate moon altitude and separation 
        moon_DECRA = desisurvey.ephem.get_object_interpolator(night_ephem, 'moon', altaz=False)
        moon_ALTAZ = desisurvey.ephem.get_object_interpolator(night_ephem, 'moon', altaz=True)

        # sun moon altitude and separation 
        sun_DECRA = desisurvey.ephem.get_object_interpolator(night_ephem, 'sun', altaz=False)
        sun_ALTAZ = desisurvey.ephem.get_object_interpolator(night_ephem, 'sun', altaz=True)

        # get moon illumination 
        moon_ill = night_ephem['moon_illum_frac']

    if ra is None or dec is None: 
        # default values when there's no tile position yet 
        frame = desisurvey.utils.get_observer(astropy.time.Time(mjd, format='mjd'))
        # get RA and Dec at the zenith 
        altaz = SkyCoord(alt=90.*u.deg, az=0*u.deg,
                obstime=astropy.time.Time(mjd, format='mjd'), 
                frame='altaz', 
                location=frame.location)
        ra, dec = altaz.icrs.ra.to(u.deg).value, altaz.icrs.dec.to(u.deg).value

    # calculate airmass 
    airmass = desisurvey.utils.get_airmass(
            astropy.time.Time(mjd, format='mjd'), 
            ra * u.deg, 
            dec * u.deg)
    # moon ephem
    moon_alt, _ = moon_ALTAZ(mjd)     # moon altitude
    moon_dec, moon_ra = moon_DECRA(mjd)
    moon_sep = desisurvey.utils.separation_matrix( # separation 
            [moon_ra], [moon_dec], [ra], [dec])[0][0]
    # sun ephem
    sun_alt, _ = sun_ALTAZ(mjd) # altitude
    sun_dec, sun_ra = sun_DECRA(mjd)
    sun_sep = desisurvey.utils.separation_matrix([sun_ra], [sun_dec], [ra],
            [dec])[0][0]
    # sky brightness without twilight at 5000A for observing conditions 
    Isky5000_exp = Isky5000_notwilight_regression(airmass, moon_ill, moon_sep, moon_alt)
     # add twilight contribution 
    if sun_alt >= -18.:
        Isky5000_exp += Isky5000_twilight_regression(airmass, sun_sep, sun_alt)

    Isky5000_nom = 1.1282850428182252 # 1e-17 erg/s/cm^2/A/arcsec2
    
    #if moon_sep < 30. - moon_alt: print('moon sep < 30 - moon alt', Isky5000_exp) 
    #if moon_sep > 160. - 1.1 * moon_alt: print('moon sep > 160 - 1.1 moon alt', Isky5000_exp) 
    #if moon_sep > moon_alt + 250: print('moon sep > 250. + moon alt', Isky5000_exp) 
    #if moon_sep < moon_alt - 50: print('moon sep > moon alt - 50', moon_sep, moon_alt, Isky5000_exp) 
    return Isky5000_exp/Isky5000_nom


def Isky5000_notwilight_regression(airmass, moon_frac, moon_sep, moon_alt): 
    ''' Calculate sky surface brightness at 5000A during bright time without
    twilight. 

    Surface brightness is based on a polynomial regression model fit to
    log(observed sky brightness). The observed sky brightnesses were compiled
    from 
    * DESI SV1 bright exposures with TRANSP > 0.95 and SUN_ALT < -18 
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
    norder = 6 
    # polynomial regression cofficients for estimating exposure time factor during
    # non-twilight from airmass, moon_frac, moon_sep, moon_alt  
    coeffs = np.array([ 1.37122205e-02,  1.19640701e-02,  8.16266472e-03,  1.39280691e-01,
       -3.34998306e-05,  1.03669586e-02,  4.14594997e-03,  4.98021653e-02,
        2.04047595e-03,  2.65405851e-03,  9.64071111e-02,  5.05776641e-02,
       -1.83615882e-02, -3.08046695e-02, -6.42666661e-03,  1.85826555e-02,
       -1.31684802e-03, -6.31654630e-02, -2.75126470e-02, -6.66878948e-04,
        5.98373278e-02, -2.63741965e-02,  3.76787045e-02,  4.90090134e-02,
        1.88864338e-02, -4.62783691e-03,  2.14026758e-02, -2.84568954e-03,
        2.70850466e-02,  1.81878506e-02,  1.19064002e-03,  2.01685330e-05,
        6.32815029e-05,  2.19722530e-04, -2.84904174e-05,  4.19099793e-02,
       -3.35908059e-03, -4.04610260e-02,  5.02943104e-02, -2.51984239e-03,
       -1.07739033e-03, -1.30186514e-01, -4.01654696e-02, -3.46066336e-02,
       -1.18782966e-02, -1.50282808e-03,  2.07729906e-02, -6.47500952e-02,
       -3.04598385e-02, -1.57347975e-02,  2.54821102e-03,  3.64911542e-05,
       -4.62796429e-05, -3.23300406e-04, -1.78909655e-04, -1.24732804e-03,
        6.34377564e-02, -2.72372301e-01, -2.71942506e-02, -1.10716987e-02,
        4.76993374e-03, -3.56730604e-04, -1.03304340e-04, -1.68009283e-04,
       -4.62115541e-05,  2.50550445e-06,  9.68241078e-07,  8.13914082e-07,
        5.54375178e-07,  1.48456774e-06,  7.75842516e-02,  5.87414817e-04,
        7.64725150e-02,  2.20412705e-02, -3.13542091e-04, -2.85748098e-02,
       -5.79352294e-02,  1.68630867e-02,  8.05659390e-03, -9.36412652e-04,
        6.02392576e-03,  6.04139180e-02, -3.02037097e-02,  2.86976403e-02,
       -8.55838311e-04,  4.45485436e-03, -5.92335120e-05,  1.30608798e-04,
        2.76992145e-04,  1.73417298e-04,  8.46802477e-05,  1.44803613e-01,
       -1.41019852e-01, -8.82311141e-03,  2.40245147e-02,  8.55080444e-03,
        1.65751285e-04, -3.06026118e-04,  2.47806624e-05, -1.04175553e-04,
       -9.60076655e-07,  1.25270206e-06, -2.26510053e-07, -8.13034123e-07,
       -3.62994514e-07,  8.10275226e-03,  1.30265610e-01, -2.59539811e-02,
        2.81777445e-02, -2.72609442e-02, -6.15755933e-05, -6.96052078e-05,
        4.44190852e-04,  1.88926726e-04, -9.41703486e-05,  3.33875647e-06,
        4.69869178e-06,  3.38184992e-07,  3.36524261e-07,  7.88612964e-07,
       -2.53343718e-08, -6.01290739e-08, -3.55412440e-08, -9.68385387e-09,
       -2.70149823e-09, -7.00167017e-09,  1.20582063e-01,  1.44232169e-02,
       -5.99644863e-02, -5.46588272e-02,  3.79091286e-03,  1.13505888e-01,
        1.40498432e-01, -1.28698148e-03,  3.88926808e-03,  3.27254434e-03,
        1.84300473e-02,  7.48888214e-02,  3.77593511e-02, -1.68928339e-02,
       -1.38366855e-02, -7.65439127e-03,  3.79474387e-05, -7.28387298e-05,
       -1.86982036e-04, -8.33117928e-05,  2.20980133e-03,  2.16668147e-01,
        1.36526658e-01,  4.16746892e-03, -6.47206461e-03, -2.96650732e-03,
        8.65687911e-05,  4.72609545e-04,  4.72384634e-04,  1.51577611e-04,
       -3.88374762e-07, -1.41057120e-06, -1.18013283e-08,  1.54156990e-06,
        4.42399919e-07, -4.41145646e-04,  1.39000704e-01, -4.37408535e-01,
       -1.54311899e-02, -1.91755401e-03,  4.36138614e-03, -4.42645315e-05,
       -8.34403032e-06, -1.53210208e-04, -3.48132905e-05,  7.41531513e-07,
       -2.36866678e-06, -4.74525136e-06, -4.84825494e-06, -1.02298977e-06,
       -1.75128388e-09,  1.07399591e-08,  1.26208068e-08,  4.31576944e-09,
       -5.13099539e-09, -1.26290889e-09,  1.32565239e-02,  1.36316867e-01,
        4.99691249e-01, -1.74968493e-02, -3.34139883e-03, -2.42772081e-03,
        3.36178419e-04,  2.23896110e-04,  1.73634367e-04, -1.38246185e-06,
       -2.40280959e-06, -2.66317367e-06, -2.80539906e-06, -4.19012237e-07,
        4.36408043e-07, -5.34778782e-09, -2.87653770e-08, -2.16845793e-09,
        3.21431780e-08,  2.09245593e-08,  1.66330738e-09,  1.08938813e-10,
        3.53672248e-10,  3.43915997e-10,  7.83100372e-11, -2.92671982e-11,
       -5.14346513e-13,  1.39651631e-11])
    
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
    to twilight contributions measured from SV1 and BOSS exposures with sun
    altitutde > -18deg.

    For details, see jupyter notebook: 
    https://github.com/desi-bgs/bgs-cmxsv/blob/55850e44d65570da69de0788c652cff698416834/doc/nb/sky_model_twilight_fit.ipynb
    
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
    norder = 2
    # coefficients fit in notebook
    # https://github.com/desi-bgs/bgs-cmxsv/blob/521211dccff7cb3b71b07d84522802acade26071/doc/nb/sky_model_twilight_fit.ipynb
    skymodel_coeff = np.array([ 
        7.74536884e-01,  1.25866845e+00,  1.06073841e+00,  2.27595902e-01,
        9.12671470e-01, -4.16253157e-02, -2.26899688e-02,  3.52752773e-02,
        6.39485772e-03, -5.03964852e-04])

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
    nominal_time = getattr(config.programs, program).efftime()
    sbprof = getattr(config.programs, program).sbprof()

    # Calculate actual / nominal factors.
    f_seeing = seeing_exposure_factor(seeing, sbprof=sbprof)
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
        for program in desisurvey.tiles.get_tiles().programs:
            progconf = getattr(config.programs, program, None)
            if progconf is None:
                unknownprograms.append(program)
                nomtime = 1000/24/60/60
            else:
                nomtime = progconf.efftime().to(u.day).value
            self.TEXP_TOTAL[program] = nomtime
        if len(unknownprograms) > 0:
            self.log.warning(
                'Unrecognized program {}, '.format(' '.join(unknownprograms)) +
                'using default exposure time of 1000 s')

        # Initialize optional history tracking.
        self.save_history = save_history
        if save_history:
            self.history = dict(mjd=[], signal=[], background=[], snr2frac=[])

    def weather_factor(self, seeing, transp, sky_level, sbprof='ELG'):
        """Return the relative SNR2 accumulation rate for specified conditions.

        This is the inverse of the instantaneous exposure factor due to seeing and transparency.

        Parameters
        ----------
        seeing : float
            Atmospheric seeing in arcseconds.
        transp : float
            Atmospheric transparency in the range (0,1).
        sky_level : float
            sky_level relative to nominal
        """
        fac = transp**2
        fac /= seeing_exposure_factor(seeing, sbprof=sbprof)
        fac /= sky_level
        return fac

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
        self.srate0 = np.sqrt(self.weather_factor(seeing, transp, 1.0))
        # effective sky / nominal sky based on SurveySpeed calculation 
        self.brate0 = sky + 0.932/3.373 * (self.texp_total / 0.0115741) + 1.71 / 3.373
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
        srate = np.sqrt(self.weather_factor(seeing, transp, 1.0))
        # effective sky / nominal sky based on SurveySpeed calculation 
        brate = sky + 0.932/3.373 * (dt / 0.0115741) + 1.71 / 3.373
        self.signal += dt * srate / self.srate0
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
