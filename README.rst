========================================================
psy-maps: The psyplot plugin for visualizations on a map
========================================================

.. start-badges

.. list-table::
    :stub-columns: 1
    :widths: 10 90

    * - docs
      - |docs|
    * - tests
      - |travis| |appveyor| |requires| |coveralls|
    * - package
      - |version| |conda| |supported-versions| |supported-implementations| |zenodo|

.. |docs| image:: http://readthedocs.org/projects/psy-maps/badge/?version=latest
    :alt: Documentation Status
    :target: http://psy-maps.readthedocs.io/en/latest/?badge=latest

.. |travis| image:: https://travis-ci.org/Chilipp/psy-maps.svg?branch=master
    :alt: Travis
    :target: https://travis-ci.org/Chilipp/psy-maps

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/3jk6ea1n4a4dl6vk?svg=true
    :alt: AppVeyor
    :target: https://ci.appveyor.com/project/Chilipp/psy-maps

.. |coveralls| image:: https://coveralls.io/repos/github/Chilipp/psy-maps/badge.svg?branch=master
    :alt: Coverage
    :target: https://coveralls.io/github/Chilipp/psy-maps?branch=master

.. |requires| image:: https://requires.io/github/Chilipp/psy-maps/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/Chilipp/psy-maps/requirements/?branch=master

.. |version| image:: https://img.shields.io/pypi/v/psy-maps.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/psy-maps

.. |conda| image:: https://anaconda.org/conda-forge/psy-maps/badges/version.svg
    :alt: conda
    :target: https://conda.anaconda.org/conda-forge/psy-maps

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/psy-maps.svg?style=flat
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/psy-maps

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/psy-maps.svg?style=flat
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/psy-maps

.. |zenodo| image:: https://zenodo.org/badge/83305582.svg
    :alt: Zenodo
    :target: https://zenodo.org/badge/latestdoi/83305582


.. end-badges

Welcome to the psyplot plugin for visualizations on a map. This package uses the
cartopy_ package to project the plots that are made with the psy-simple_ plugin
to an earth-referenced grid. It's main plot methods are the
mapplot_ and mapvector_ plot methods which can plot
rectangular and triangular plots.

See the full documentation on
`readthedocs.org <http://psyplot.readthedocs.io/projects/psy-maps>`__ for all
`plot methods`_ and examples_.

.. _cartopy: http://scitools.org.uk/cartopy
.. _mapplot: http://psyplot.readthedocs.io/projects/psy-maps/en/latest/generated/psyplot.project.plot.mapplot.html#psyplot.project.plot.mapplot
.. _mapvector: http://psyplot.readthedocs.io/projects/psy-maps/en/latest/generated/psyplot.project.plot.mapvector.html#psyplot.project.plot.mapvector
.. _psy-simple: http://psyplot.readthedocs.io/projects/psy-simple/
.. _plot methods: http://psyplot.readthedocs.io/projects/psy-maps/en/latest/plot_methods
.. _examples: http://psyplot.readthedocs.io/projects/psy-maps/en/latest/examples
