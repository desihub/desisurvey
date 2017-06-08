=====================
desisurvey change log
=====================

0.7.1 (unreleased)
------------------

* No changes yet

0.7.0 (2017-06-05)
------------------

* Freeze IERS table used by astropy time, coordinates.
* Implement alternate greedy scheduler with optional policy weights.
* Implement iterative HA optimization and LST-driven scheduler.
* Add `plots.plot_planner()`
* Partial fix of RA=0/360 planning bug

0.6.0 (2017-05-10)
------------------

* Add new config yaml file and python wrapper.
* Convert all code to use new config machinery.
* Add new class Plan for future use in scheduling.
* Unify different output files with overlapping contents into single output
  managed by desisurvey.progress.
* Cleanup and reorganize the Ephemerides class.
* Add comparisons with independent JPL Horizons run to unit tests for
  AltAz transforms and ephemerides calculations.
* Add new plot utilities for Progress and Plan objects.
* Document and handle astropy IERS warnings about future times.
* Rename exposurecalc module to etc (exposure-time calculator).
* Update docstrings and imports, and remove unused code.

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
