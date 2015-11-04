import numpy as np

def transmission(airmass, filter, refAirmass=1.3) :
    extCoeff = extinctionModel (filter) 
    transmission = 10**(-0.4 * extCoeff*(airmass-refAirmass))
    return transmission

def dirtTransmission(zd) :
    deg_to_rad = 2*np.pi/360.
    transmission = 1.0
    if zd > (90*deg_to_rad): 
        transmission = 0.0
    return transmission

def lunarDirtTransmission(moon_sep) :
    deg_to_rad = 2*np.pi/360.
    transmission = 1.0
    if (moon_sep < 0.5*deg_to_rad) :
        transmission = 0.0
    return transmission

def extinctionModel (filter) :
    if filter == "u" : k=0.58
    elif filter == "g" : k=0.18
    elif filter == "r" : k=0.09
    elif filter == "i" : k=0.08
    elif filter == "z" : k=0.08
    elif filter == "y" : k=0.08
    else : raise Exception ("no such filter {}".format(filter))
    return k

