import ephem

def programnamer(moon_elevation, moon_illumination_fraction):
    moon_brightness = moon_elevation*moon_illumination_fraction
    
    if moon_elevation < 0:
        return 'dark'
    elif moon_brightness < 12.0:
        return 'grey'
    else:
        return 'bright'
    
def getprogramname(dateobs):
    kittpeak = ephem.Observer()
    kittpeak.lat = '31:57:50.5'
    kittpeak.lon = '-111:35:59.6'
    kittpeak.date = (dateobs-2415020)
    kittpeak.elevation = 2095.5
    
    moon = ephem.Moon()
    moon.compute(kittpeak)
    
    moonalt = str(moon.alt)
    moonalt_split = moonalt.split(':')
    moonalt = (float(moonalt_split[0])+float(moonalt_split[1])/60
               +float(moonalt_split[2])/3600)
    
    moonfrac = moon.phase/100
    
    progname = programnamer(moonalt, moonfrac)
    
    return progname
