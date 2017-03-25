from __future__ import print_function, division
import numpy as np
from astropy.time import Time
import astropy.units as u
import specsim.simulator
from surveysim.weather import weatherModule
from desisurvey.utils import radec2altaz

def expTimeEstimator(weatherNow, amass, program, ebmv, sn2, moonFrac, moonDist, moonAlt):
    """
    Estimates expusure length given current conditions.

    Args:
        weatherNow: dictionnary containing the following keys:
                    'Seeing', 'Transparency', 'OpenDome', 'Clouds'
        amass: float, air mass
        programm: string, 'DARK', 'BRIGHT' or 'GRAY'
        ebmv: float, E(B-V)
        sn2: float, desired (S/N)^2
        moonFrac: float, Moon illumination fraction, between 0 (new) and 1 (full).
        moonDist: float, separation angle between field center and moon in degrees.
        moonAlt: float, moon altitude angle in degrees.

    Returns:
        float, estimated exposure time
    """

    seeing_ref = 1.1 # Seeing value to which actual seeing is normalised
    exp_ref_dark = 1000.0   # Reference exposure time in seconds
    exp_ref_bright = 300.0  # Idem but for bright time programme
    exp_ref_grey = exp_ref_dark
    sn2_nom = 100.0 # Nominal sign-to-noise: again, made-up number

    if program == "DARK":
        exp_ref = exp_ref_dark
    elif program == "BRIGHT":
        exp_ref = exp_ref_bright
    elif program == "GRAY":
        exp_ref = exp_ref_grey
    else:
        exp_ref = 0.0 # Replace with throwing an exception
    seeing = weatherNow['Seeing']
    a = 4.6
    b = -1.55
    c = 1.15
    f_seeing =  (a+b*seeing+c*seeing*seeing) / (a-0.25*b*b/c)
    # Rescale value
    f11 = (a + b*1.1 + c*1.21)/(a-0.25*b*b/c)
    f_seeing /= f11
    if weatherNow['Transparency'] > 0.0:
        f_transparency = 1.0 / weatherNow['Transparency']
    else:
        f_transparency = 1.0e9

    #Ag=3.303*ebv[i]
    #Ai=1.698*ebv[i]
    #i_increase[i]=(10^(Ai/2.5))^2
    #g_increase[i]=(10^(Ag/2.5))^2

    Ag = 3.303*ebmv # Use g-band
    f_ebmv = np.power( 10.0, (2.0*Ag/2.5) )
    f_am = np.power(amass, 1.25)

    f_moon = moonExposureTimeFactor(moonFrac, moonDist, moonAlt)
    #print (f_am, f_seeing, f_transparency, f_ebmv, f_moon)
    f = f_am * f_seeing * f_transparency * f_ebmv * f_moon
    if f >= 0.0:
        value = exp_ref * f
    else:
        value = exp_ref
    return value


# A specsim moon model that will be created once, then cached here.
_moonModel = None

# Linear regression coefficients for converting scattered moon V-band
# magnitude into an exposure-time correction factor.
_moonCoefficients = np.array([
    -8.83964463188, -7372368.5041596508, 775.17763895781638,
    -20185.959363990656, 174143.69095766739])

def moonExposureTimeFactor(moonFrac, moonDist, moonAlt):
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
        print('Creating a specsim moon model.')
        desi = specsim.simulator.Simulator('desi')
        _moonModel = desi.atmosphere.moon

    # Convert input parameters to those used in the specim moon model.
    _moonModel.moon_phase = np.arccos(2 * moonFrac - 1) / np.pi
    _moonModel.moon_zenith = (90 - moonAlt) * u.deg
    _moonModel.separation_angle = moonDist * u.deg

    # Calculate the scattered moon V-band magnitude.
    V = _moonModel.scattered_V.value

    # Evaluate the linear regression model.
    X = np.array((1, np.exp(-V), 1/V, 1/V**2, 1/V**3))
    return _moonCoefficients.dot(X)


def airMassCalculator(ra, dec, lst): # Valid for small to moderate angles.
    """
    Calculates airmass given position and LST.  Uses formula from
    Rosenberg (1966)

    Args:
        ra: float (degrees)
        dec: float (degrees)
        lst: float (degrees)

    Returns:
        float, air mass
    """

    Alt, Az = radec2altaz(ra, dec, lst)
    cosZ = np.cos(np.radians(90.0-Alt))
    if isinstance(Alt, np.ndarray):
        amass = np.full(len(Alt), 1.0e99, dtype='f8')
        amass[np.where(Alt>0.0)] = 1.0/(cosZ + 0.025*np.exp(-11.0*cosZ))
    else:
        if Alt > 0.0:
            amass = 1.0/(cosZ + 0.025*np.exp(-11.0*cosZ))
        else:
            amass = 1.0e99

    return amass
