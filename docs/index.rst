.. psy-maps documentation master file, created by
   sphinx-quickstart on Mon Jul 20 18:01:33 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _psy-maps:

psy-maps: The psyplot plugin for visualizations on a map
========================================================

Welcome to the psyplot plugin for visualizations on a map. This package uses the
cartopy_ package to project the plots that are made with the psy-simple_ plugin
to an earth-referenced grid. It's main plot methods are the
:attr:`~psyplot.project.ProjectPlotter.mapplot` and
:attr:`~psyplot.project.ProjectPlotter.mapvector` plot methods which can plot
rectangular and triangular 2-dimensional data.

See the :ref:`plot_methods` and :ref:`gallery_examples` for more information.

.. _cartopy: http://scitools.org.uk/cartopy
.. _psy-simple: http://psyplot.readthedocs.io/projects/psy-simple/

.. start-badges

.. only:: html and not epub

    .. list-table::
        :stub-columns: 1
        :widths: 10 90

        * - docs
          - |docs|
        * - tests
          - |travis| |appveyor| |requires| |codecov|
        * - package
          - |version| |conda| |supported-versions| |supported-implementations| |zenodo|

    .. |docs| image:: http://readthedocs.org/projects/psy-maps/badge/?version=latest
        :alt: Documentation Status
        :target: http://psy-maps.readthedocs.io/en/latest/?badge=latest

    .. |travis| image:: https://travis-ci.org/psyplot/psy-maps.svg?branch=master
        :alt: Travis
        :target: https://travis-ci.org/psyplot/psy-maps

    .. |appveyor| image:: ttps://ci.appveyor.com/api/projects/status/rd733xj3tfrk4tot/branch/master?svg=true
        :alt: AppVeyor
        :target: https://ci.appveyor.com/project/psyplot/psy-maps

    .. |codecov| image:: https://codecov.io/gh/psyplot/psy-maps/branch/master/graph/badge.svg
        :alt: Coverage
        :target: https://codecov.io/gh/psyplot/psy-maps

    .. |requires| image:: https://requires.io/github/psyplot/psy-maps/requirements.svg?branch=master
        :alt: Requirements Status
        :target: https://requires.io/github/psyplot/psy-maps/requirements/?branch=master

    .. |version| image:: https://img.shields.io/pypi/v/psy-maps.svg?style=flat
        :alt: PyPI Package latest release
        :target: https://pypi.python.org/pypi/psy-maps

    .. |conda| image:: https://anaconda.org/conda-forge/psy-maps/badges/version.svg
        :alt: conda
        :target: https://anaconda.org/conda-forge/psy-maps

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


Documentation
-------------

.. toctree::
    :maxdepth: 1

    installing
    plot_methods
    examples/index
    contribute
    api/psy_maps



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
