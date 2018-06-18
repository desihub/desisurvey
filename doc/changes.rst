=====================
desisurvey change log
=====================

0.10.2 (unreleased)
-------------------

* Do not assume that input tile file includes all of DARK, BRIGHT, and GRAY
  tiles (PR `#83`_). 

.. _`#83`: https://github.com/desihub/desisurvey/pull/83

0.10.1 (2017-12-20)
-------------------

* Add Ephemerides method to calculate sun altitude during twilight.
* Set the ``EXTNAME`` keyword on the Table returned by ``Progress.get_exposures()``.

0.10.0 (2017-11-09)
-------------------

* Progress.get_exposures() updates:

  * includes FLAVOR and PROGRAM columns.
  * uses desimodel.footprint.pass2program if available.
  * standardized on UPPERCASE column names and NIGHT=YEARMMDD not YEAR-MM-DD.

0.9.3 (2017-10-09)
------------------

* Fixes #18, #49, #54.
* Improvements to surveymovie script.
* Add progress columns to track fiber assignment and planning.
* Add support for optional depth-first survey strategy.
* Docs now auto-generated at http://desisurvey.readthedocs.io/en/latest/

0.9.2 (2017-09-29)
------------------

* Implement fiber assignment policy via --fa-delay option to surveyplan.

0.9.1 (2017-09-20)
------------------

* Command line scripts --config-file option to override default config file.
* Fixes for bugs that occur when testing with a small subset of tiles.
* Changes $DESISURVEY -> $DESISURVEY_OUTPUT as output dir envvar name
* Remove astropy units from function signatures (for readthedocs).
* Add travis, coveralls and readthedocs automation.

0.9.0 (2017-09-11)
------------------

* Create surveyinit script to calculate initial HA assignments.
* Improve Optimizer algorithms (~10x faster, better initialization).
* Create surveymovie to visualize survey scheduling and progress.
* Rework surveyplan to track fiber assignment availability.
* Validate a set of observing rules consistent with the baseline strategy
  described in DESI-doc-1767-v3.

0.8.2 (2017-07-12)
------------------

* Fix flat vs. flatten for older versions of numpy (PR `#52`_).

.. _`#52`: https://github.com/desihub/desisurvey/pull/52

0.8.1 (2017-06-19)
------------------

* Fix unit tests broken in 0.8.0 (PR `#46`_).

.. _`#46`: https://github.com/desihub/desisurvey/pull/46

0.8.0 (2017-06-18)
------------------

* Implement LST-driven scheduling strategy.
* Create new optimize module for iterative HA optimization.
* Rename module plan -> schedule.
* Create new plan module to manage fiber-assignment groups and priorities.

0.7.0 (2017-06-05)
------------------

* Freeze IERS table used by astropy time, coordinates.
* Implement alternate greedy scheduler with optional policy weights.
* Add `plots.plot_scheduler()`
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
