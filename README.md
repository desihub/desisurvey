# desisurvey

Code for desi survey planning and implementation.  There is currently no driver in this package.  One must use the driver in surveysim; see instructions in the README file there.

A plotting utility is provided to look at the progression of the survey and various metrics.  To run the plotting tool:

	>>> from surveysim.plotsurvey import plotsurvey
	>>> plotsurvey("obslist{_all|YYYYMMDD}.fits", plot_type='t', program='m')

The default filename is ./obslist_all.fits; plot_type is either 'f' (footprint, default), 'h' (histograms), 't' (time evolution) or 'e' (exposure time); and program us either 'm' (main survey), 'b' (BGS) or 'a' (all).

