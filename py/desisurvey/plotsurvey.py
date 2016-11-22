import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as ticker

def plotsurvey(filename='obslist_all.fits', plot_type='f', program='m'):
    """
    This function plots various quantities output from surveySim.

    Args:
        filename: string, file must be in obslist{YYYYMMDD|_all}.fits format
        plot_type: 'f', 'h', 't' or 'e'
        program: 'm' (main survey, i.e. dark+grey), 'b' (BGS), 'a' (all)
    """ 

    t = Table.read(filename, format='fits')

    if plot_type == 'f':
        fig, ax = plt.subplots()
        ra = t['RA']
        ra[ra>300.0] -= 360.0
        dec = t['DEC']
        mjd = t['MJD']
        mjd_start = np.min(mjd)
        mjd -= mjd_start

        if program=='a':
            i1 = np.where( mjd/365.0 < 1.0 )
            i2 = np.where( (mjd/365.0 >= 1.0) & (mjd/365.0 < 2.0) )
            i3 = np.where( (mjd/365.0 >= 2.0) & (mjd/365.0 < 3.0) )
            i4 = np.where( (mjd/365.0 >= 3.0) & (mjd/365.0 < 4.0) )
            i5 = np.where( (mjd/365.0 >= 4.0) & (mjd/365.0 < 5.0) )
        elif program=='m':
            i1 = np.where( (mjd/365.0 < 1.0) & (t['PROGRAM']!='BRIGHT') )
            i2 = np.where( ((mjd/365.0 >= 1.0) & (mjd/365.0 < 2.0)) & (t['PROGRAM']!='BRIGHT') )
            i3 = np.where( ((mjd/365.0 >= 2.0) & (mjd/365.0 < 3.0)) & (t['PROGRAM']!='BRIGHT') )
            i4 = np.where( ((mjd/365.0 >= 3.0) & (mjd/365.0 < 4.0)) & (t['PROGRAM']!='BRIGHT') )
            i5 = np.where( ((mjd/365.0 >= 4.0) & (mjd/365.0 < 5.0)) & (t['PROGRAM']!='BRIGHT') )
        elif program=='b':
            i1 = np.where( (mjd/365.0 < 1.0) & (t['PROGRAM']=='BRIGHT') )
            i2 = np.where( ((mjd/365.0 >= 1.0) & (mjd/365.0 < 2.0)) & (t['PROGRAM']=='BRIGHT') )
            i3 = np.where( ((mjd/365.0 >= 2.0) & (mjd/365.0 < 3.0)) & (t['PROGRAM']=='BRIGHT') )
            i4 = np.where( ((mjd/365.0 >= 3.0) & (mjd/365.0 < 4.0)) & (t['PROGRAM']=='BRIGHT') )
            i5 = np.where( ((mjd/365.0 >= 4.0) & (mjd/365.0 < 5.0)) & (t['PROGRAM']=='BRIGHT') )
        else:
            print("if set, program should be a, m or b; default is m.\n")
            return
        y1 = plt.scatter(ra[i1], dec[i1], c='r', marker='.')
        y2 = plt.scatter(ra[i2], dec[i2], c='b', marker='.')
        y3 = plt.scatter(ra[i3], dec[i3], c='g', marker='.')
        y4 = plt.scatter(ra[i4], dec[i4], c='y', marker='.')
        y5 = plt.scatter(ra[i5], dec[i5], c='m', marker='.')

        plt.xlabel('RA (deg)')
        plt.ylabel('DEC (deg)')
        plt.legend((y1, y2, y3, y4, y5), ('Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'), scatterpoints=1, loc=2)
        ticks = ax.get_xticks()
        ticks[ticks < 0] += 360
        ax.set_xticklabels([int(tick) for tick in ticks])

    elif plot_type == 'h':
        if (program=='m'):
            i = np.where( t['PROGRAM'] != 'BRIGHT')
        elif (program=='b'):
            i = np.where( t['PROGRAM'] == 'BRIGHT')
        elif (program=='a'):
            i = np.arange(len(t['PROGRAM']))
        else:
            print("if set, program should be a, m or b; default is m.\n")
            return

        plt.figure(1)
        plt.subplot(231)
        x = t['EXPTIME']
        n, bins, patches = plt.hist(x[i], 20, facecolor='0.5', alpha=0.75)
        plt.xlabel('Exposure time (seconds)')
        plt.ylabel('Count')

        plt.subplot(232)
        x = t['SEEING']
        n, bins, patches = plt.hist(x[i], 20, facecolor='0.5', alpha=0.75)
        plt.xlabel('Seeing (arcseconds)')
        plt.ylabel('Count')

        plt.subplot(233)
        x = t['LINTRANS']
        n, bins, patches = plt.hist(x[i], 20, facecolor='0.5', alpha=0.75)
        plt.xlabel('Linear transparency')
        plt.ylabel('Count')

        plt.subplot(234)
        x = t['AIRMASS']
        n, bins, patches = plt.hist(x[i], 20, facecolor='0.5', alpha=0.75)
        plt.xlabel('Airmass')
        plt.ylabel('Count')

        plt.subplot(235)
        y1 = t['MOONALT']
        y2 = y1[i]
        x1 = t['MOONFRAC']
        x2 = x1[i]
        x = x2.compress((y2>0.0).flat)
        n, bins, patches = plt.hist(x, 20, facecolor='0.5', alpha=0.75)
        plt.xlabel('Moon illumination fraction')
        plt.ylabel('Count')

        plt.subplot(236)
        y = t['MOONALT']
        y2 = y[i]
        x1 = t['MOONDIST']
        x2 = x1[i]
        x = x2.compress((y2>0.0).flat)
        n, bins, patches = plt.hist(x, 20, facecolor='0.5', alpha=0.75)
        plt.xlabel('Distance from the Moon (deg)')
        plt.ylabel('Count')

    elif plot_type == 't':
        if (program=='m'):
            i = np.where( t['PROGRAM'] != 'BRIGHT')
        elif (program=='b'):
            i = np.where( t['PROGRAM'] == 'BRIGHT')
        elif (program=='a'):
            i = np.arange(len(t['PROGRAM']))
        else:
            print("if set, program should be a, m or b; default is m.\n")
            return
        
        mjd = t['MJD']
        mjd_start = np.min(mjd)
        mjd -= mjd_start
        plt.figure(1)

        plt.subplot(221)
        y = t['MOONALT']
        plt.plot(mjd[i], y[i], linestyle='-', color='black')
        plt.xlabel('Days')
        plt.ylabel('Moon elevation (degrees)')

        plt.subplot(222)
        y = t['SEEING']
        plt.plot(mjd[i], y[i], linestyle='-', color='black')
        plt.xlabel('Days')
        plt.ylabel('Seeing (arcseconds)')

        plt.subplot(223)
        y = t['LINTRANS']
        plt.plot(mjd[i], y[i], linestyle='-', color='black')
        plt.xlabel('Days')
        plt.ylabel('Linear transparency')

        plt.subplot(224)
        if (program=='b'):
            x = mjd[t['PROGRAM']=='BRIGHT']
        elif (program=='m'):
            x = mjd[t['PROGRAM']!='BRIGHT']
        elif (program=='a'):
            x = mjd
        y = np.arange(len(x)) + 1
        #y = np.arange(len(mjd)) + 1
        #plt.plot(mjd[i], y[i], linestyle='-', color='black')
        plt.plot(x, y, linestyle='-', color='black')
        plt.xlabel('Days')
        plt.ylabel('Number of tiles observed')


    elif plot_type == 'e':
        y = t['EXPTIME']
        if (program=='m'):
            i = np.where( t['PROGRAM'] != 'BRIGHT')
        elif (program=='b'):
            i = np.where( t['PROGRAM'] == 'BRIGHT')
        elif (program=='a'):
            i = np.arange(len(y))
        else:
            print("if set, program should be a, m or b; default is m.\n")
            return
        plt.figure(1)

        plt.subplot(221)
        x = t['LINTRANS']
        plt.scatter(x[i], y[i], marker='.', color='black')
        plt.xlabel('Linear transparency')
        plt.ylabel('Exposure time (seconds)')

        plt.subplot(222)
        x = t['SEEING']
        plt.scatter(x[i], y[i], marker='.', color='black')
        plt.xlabel('Seeing (arcseconds)')
        plt.ylabel('Exposure time (seconds)')

        plt.subplot(223)
        x = t['EBMV']
        plt.scatter(x[i], y[i], marker='.', color='black')
        plt.xlabel('E(B-V)')
        plt.ylabel('Exposure time (seconds)')

        plt.subplot(224)
        x = t['AIRMASS']
        plt.scatter(x[i], y[i], marker='.', color='black')
        plt.xlabel('Airmass')
        plt.ylabel('Exposure time (seconds)')

    plt.show()

