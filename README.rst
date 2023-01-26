==================================
DESI Survey Planning and Scheduler
==================================

|Actions Status| |Coveralls Status| |Documentation Status|

.. |Actions Status| image:: https://github.com/desihub/desisurvey/workflows/CI/badge.svg
    :target: https://github.com/desihub/desisurvey/actions
    :alt: GitHub Actions CI Status

.. |Coveralls Status| image:: https://coveralls.io/repos/desihub/desisurvey/badge.svg
    :target: https://coveralls.io/github/desihub/desisurvey
    :alt: Test Coverage Status

.. |Documentation Status| image:: https://readthedocs.org/projects/desisurvey/badge/?version=latest
    :target: https://desisurvey.readthedocs.io/en/latest/
    :alt: Documentation Status

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

.. _`desisurvey on Read the Docs`: https://desisurvey.readthedocs.io/en/latest/

License
-------

desisurvey is free software licensed under a 3-clause BSD-style license.
For details see the ``LICENSE.rst`` file.
