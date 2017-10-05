==================================
DESI Survey Planning and Scheduler
==================================

Introduction
------------

This package provides the tools to plan and schedule observations of the 
predefined tiles in the DESI footprint, each associated with an
observing program (DARK/GRAY/BRIGHT). The algorithms in this package
do not deal with individual targets or their assignment to fibers.

See the `surveysim <https://github.com/desihub/surveysim>`_ package
for a driver that uses randomly generated weather conditions to
simulate a survey.  In particular, this
`tutorial <https://github.com/desihub/surveysim/blob/master/doc/tutorial.rst>`_
is a good starting point.

Full Documentation
------------------

Please visit `desisurvey on Read the Docs`_

.. image:: https://readthedocs.org/projects/desisurvey/badge/?version=latest
    :target: http://desisurvey.readthedocs.io/en/latest/
    :alt: Documentation Status

.. _`desisurvey on Read the Docs`: http://desisurvey.readthedocs.io/en/latest/

Travis Build Status
-------------------

.. image:: https://img.shields.io/travis/desihub/desisurvey.svg
    :target: https://travis-ci.org/desihub/desisurvey
    :alt: Travis Build Status


Test Coverage Status
--------------------

.. image:: https://coveralls.io/repos/desihub/desisurvey/badge.svg?service=github
    :target: https://coveralls.io/github/desihub/desisurvey
    :alt: Test Coverage Status

License
-------

desisurvey is free software licensed under a 3-clause BSD-style license.
For details see the ``LICENSE.rst`` file.
