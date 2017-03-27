import numpy as np

# airmass seeing dependence &
#  komologorov seeing wavelength dep
def seeingWithAirmassAndLambda(airmass, filter, seeingAtZenith=1.0) :
    
    seeing = seeingAtZenith*airmass**(3./5.)

    refWavelength = 775 # i-band
    wavelength=filterEffWavelength(filter)
    komol = (wavelength/refWavelength)**-0.2
    seeing = seeing*komol
    return seeing

def filterEffWavelength(filter) :
    if filter == "u" : effLam = 380.
    elif filter == "g" : effLam = 475.
    elif filter == "r" : effLam = 635.
    elif filter == "i" : effLam = 775.
    elif filter == "z" : effLam = 925.
    elif filter == "y" : effLam = 1000.
    else : raise Exception ("no such filter {}".format(filter))
    return effLam

#
