import math
import ephem
import numpy as np
import desimodel.io
from astropy.table import Table
from astropy.io import fits

def get_plan(dateobs):
    
    #- Take the input modified Julian date (MJD) and convert to Dublian Julian Date (DJD)
    DJD = dateobs-15019.5
    
    #- Set up the observing location to be Kitt Peak
    kittpeak = ephem.Observer()
    kittpeak.lon = '-111:35:59.6'
    kittpeak.lat = '31:57:50.5'
    kittpeak.elevation = 2096
    kittpeak.date = DJD
    
    #- Use pyephem to get the time of next sunset and sunrise
    sun = ephem.Sun(kittpeak)
    sun_set = kittpeak.next_setting(sun)
    sun_rise = kittpeak.next_rising(sun)
    
    #- Convert rise/set times to sidereal times
    kittpeak.date = sun_set
    sunset_lst = kittpeak.sidereal_time()
    kittpeak.date = sun_rise
    sunrise_lst = kittpeak.sidereal_time()
    
    #- Convert the sidereal times to a more useable format
    #-    1. Takes the pyephem output and stores as a string
    setlst = str(sunset_lst)
    riselst = str(sunrise_lst)
    #-    2. Split the string at the colons
    setlst_split = setlst.split(':')
    riselst_split = riselst.split(':')
    #-    3. Convert the split strings into floats and the LST to degrees
    setlst = (float(setlst_split[0])+float(setlst_split[1])/60
              +float(setlst_split[2])/3600)*15
    riselst = (float(riselst_split[0])+float(riselst_split[1])/60
               +float(riselst_split[2])/3600)*15
    
    #- Establish range of observable time as 1 hour after sunset until 1 hour before sunrise
    start_lst = setlst+15
    stop_lst = riselst-15
    
    #- Read in the tiles file
    #- Currently the file simply needs to be in the same directory as the code
    #- This should be switched to use desimodel.io.load_tiles() once you have
    #- that setup on your system. Uncomment the associate import statement above
    #- and then the following line while removing the current load method
    
    t = desimodel.io.load_tiles()
    #t = Table.read('desi-tiles.fits',hdu=1)
    
    #- Trim the tiles that are very low in declination
    dectrim = np.where(t['DEC'] > -30)[0]
    t = t[dectrim]
    
    #- Trim the tiles to the ones that should be observable
    if (start_lst > stop_lst):
        igood1 = np.where((start_lst-15 < t['RA']))[0]
        igood2 = np.where((t['RA'] < stop_lst+15))[0]
        igood = np.concatenate((igood1,igood2),axis=0)
        t = t[igood]
    if (start_lst < stop_lst):
        igood = np.where((start_lst-15 < t['RA']) & (t['RA'] < stop_lst+15))[0]
        t = t[igood]
    
    #- Assign priorities to the observable tiles
    #- This is currently just a random integer between 1 and 5
    priority = np.random.randint(1,high=5,size=len(t))
    
    #- Assign a program name of 'dark' to all tiles for now
    program = np.chararray(len(t), itemsize=4)
    program[:] = 'dark'
    
    #- Assign exposure times to the observable tiles
    #- This is just a default value of 1800.0 seconds currently
    exposetime = np.full(len(t),1800.0)
    
    #- Assign hour angles to all the tiles
    #- This is just set to zero currently
    hourangle = np.full(len(t),0.0)
    
    #- Set up arrays to combine into new table to output in plan format
    #- Place holders for the galactic longitude and latitude are here for later
    a1 = t['TILEID']
    a2 = t['RA']
    a3 = t['DEC']
    #a4 = t['GLONG']
    #a5 = t['GLAT']
    a6 = hourangle
    a7 = t['AIRMASS']
    a8 = t['EBV_MED']
    a9 = exposetime
    a10 = (t['RA']+15+360) % 360 # Place holder for start LST, one hour before meridian
    a11 = (t['RA']+45+360) % 360 # Place holder for stop LST, one hour after meridian
    a12 = priority
    a13 = program
    
    #- Define the columns for the output file
    col1 = fits.Column(name='TILEID', format='J', array=a1)
    col2 = fits.Column(name='RA', format='D', array=a2)
    col3 = fits.Column(name='DEC', format='D', array=a3)
    #col4 = fits.Column(name='GLONG', format='D', array=a4)
    #col5 = fits.Column(name='GLAT', format='D', array=a5)
    col6 = fits.Column(name='HA', format='D', array=a6)
    col7 = fits.Column(name='AIRMASS', format='D', array=a7)
    col8 = fits.Column(name='EBV_MED', format='D', array=a8)
    col9 = fits.Column(name='OBSTIME', format='D', array=a9)
    col10 = fits.Column(name='BEGINOBS', format='D', array=a10)
    col11 = fits.Column(name='ENDOBS', format='D', array=a11)
    col12 = fits.Column(name='PRIORITY', format='D', array=a12)
    col13 = fits.Column(name='PROGRAMNAME', format='20A', array=a13)
    
    cols = fits.ColDefs([col1, col2, col3, col6, col7, col8, col9, col10, col11, col12, col13])
    
    #- Create the binary table
    tbhdu = fits.BinTableHDU.from_columns(cols)
    
    #- Create the file name
    filename = 'plan' + str(int(math.floor(dateobs))) + '.fits'
    print filename
    
    #- Write the output file
    tbhdu.writeto(filename)
    

#- Uncomment these two lines to run as stand alone
#dateobs = float(raw_input('Enter the date of observation (MJD): '))
#get_plan(dateobs)