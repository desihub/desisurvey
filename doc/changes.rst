=====================
desisurvey change log
=====================

0.6.0 (unreleased)
------------------

* Add new config yaml file and python wrapper.
* Convert most (but not all) code to use new config machinery.
* Cleanup and reorganization of Ephemerides class.
* Add comparisons with independent JPL Horizons run to unit tests for
  AltAz transforms and ephemerides calculations.

0.5.0 (2017-04-13)
------------------

* Add new plot methods
* Bug fix to Az computation and airmass calculator
* Code reorganization

0.4.0 (2017-04-04)
------------------

This version was tagged for the 2% sprint data challenge.

* Add unit tests; fix afternoon planning tile updates and other minor bugs
* Fix off-by-one with YEARMMDD vs. MJD of sunset
* Add new plots module
* Refactor nightcal module into ephmerides

0.3.1 (2016-12-21)
------------------

* fixed E(B-V) scaling for exposure time (PR #12)

0.3.0 (2016-11-29)
------------------

First release after refactoring.

0.2.0 (2016-11-19)
------------------

Last version before repackaging of surveysim.
