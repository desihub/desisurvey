"""Calculate the nominal exposure time for specified observing conditions.
"""
from __future__ import print_function, division

import numpy as np

import astropy.units as u

import specsim.simulator

import desiutil.log

import desisurvey.config


def seeing_exposure_factor(seeing):
    """Scaling of exposure time with seeing, relative to nominal seeing.

    The model is based on SDSS imaging data.

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
    a, b, c = 4.6, -1.55, 1.15
    # Could drop the denominator since it cancels in the ratio.
    denom = (a - 0.25 * b * b / c)
    f_seeing =  (a + b * seeing + c * seeing ** 2) / denom
    config = desisurvey.config.Configuration()
    nominal = config.nominal_conditions.seeing().to(u.arcsec).value
    f_nominal = (a + b * nominal + c * nominal ** 2) / denom
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
    if np.any(transparency <= 1e-9):
        raise ValueError('Got unlikely transparency value < 1e-9.')
    config = desisurvey.config.Configuration()
    nominal = config.nominal_conditions.transparency()
    return nominal / transparency


def dust_exposure_factor(EBV):
    """Scaling of exposure time with median E(B-V) relative to nominal.

    The model assumes SDSS g band. Where does this model come from?

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
    Ag = 3.303 * (EBV - EBV0) # Use g-band
    return np.power(10.0, (2.0 * Ag / 2.5))


def airmass_exposure_factor(airmass):
    """Scaling of exposure time with airmass relative to nominal.

    Is this model based on SDSS or HETDEX data?

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
    config = desisurvey.config.Configuration()
    X0 = config.nominal_conditions.airmass()
    return np.power((X / X0), 1.25)


# A specsim moon model that will be created once, then cached here.
_moonModel = None

# Linear regression coefficients for converting scattered moon V-band
# magnitude into an exposure-time correction factor.
_moonCoefficients = np.array([
    -8.83964463188, -7372368.5041596508, 775.17763895781638,
    -20185.959363990656, 174143.69095766739])

def moon_exposure_factor(moonFrac, moonDist, moonAlt):
    """Calculate exposure time factor due to scattered moonlight.

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
    moonFrac : float
        Illuminated fraction of the moon, between 0-1.
    moonDist : float
        Separation angle between field center and moon in degrees.
    moonAlt : float
        Altitude angle of the moon above the horizon in degrees.

    Returns
    -------
    float
        Dimensionless factor that exposure time should be increased to
        account for increased sky brightness due to scattered moonlight.
        Will be 1 when the moon is below the horizon.
    """
    if moonAlt < 0:
        return 1.

    global _moonModel
    if not _moonModel:
        # Create a specim moon model.
        desi = specsim.simulator.Simulator('desi')
        _moonModel = desi.atmosphere.moon
        desiutil.log.get_logger().info(
            'Created a specsim moon model with EV={0}'
            .format(_moonModel._vband_extinction))

    # Convert input parameters to those used in the specim moon model.
    _moonModel.moon_phase = np.arccos(2 * moonFrac - 1) / np.pi
    _moonModel.moon_zenith = (90 - moonAlt) * u.deg
    _moonModel.separation_angle = moonDist * u.deg

    # Calculate the scattered moon V-band magnitude.
    V = _moonModel.scattered_V.value

    # Evaluate the linear regression model.
    X = np.array((1, np.exp(-V), 1/V, 1/V**2, 1/V**3))
    return _moonCoefficients.dot(X)


def expTimeEstimator(seeing, transparency, airmass, program, ebmv,
                     moonFrac, moonDist, moonAlt):
    """Calculate the nominal exposure time for specified observing conditions.

    Args:
        seeing: float, FWHM seeing in arcseconds.
        transparency: float, 0-1.
        airmass: float, air mass
        programm: string, 'DARK', 'BRIGHT' or 'GRAY'
        ebmv: float, E(B-V)
        moonFrac: float, Moon illumination fraction, between 0 (new) and 1 (full).
        moonDist: float, separation angle between field center and moon in degrees.
        moonAlt: float, moon altitude angle in degrees.

    Returns:
        float, estimated exposure time
    """
    exp_ref_dark = 1000.0   # Reference exposure time in seconds
    exp_ref_bright = 300.0  # Idem but for bright time programme
    exp_ref_grey = exp_ref_dark

    if program == "DARK":
        exp_ref = exp_ref_dark
    elif program == "BRIGHT":
        exp_ref = exp_ref_bright
    elif program == "GRAY":
        exp_ref = exp_ref_grey
    else:
        exp_ref = 0.0 # Replace with throwing an exception

    f_seeing = seeing_exposure_factor(seeing)
    f_transparency = transparency_exposure_factor(transparency)
    f_dust = dust_exposure_factor(ebmv)
    f_airmass = airmass_exposure_factor(airmass)
    f_moon = moon_exposure_factor(moonFrac, moonDist, moonAlt)

    f = f_seeing * f_transparency * f_dust * f_airmass * f_moon
    if f >= 0.0:
        value = exp_ref * f
    else:
        value = exp_ref
    return value
