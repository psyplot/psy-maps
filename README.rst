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
      - |circleci| |appveyor| |codecov|
    * - package
      - |version| |conda| |github| |zenodo|
    * - implementations
      - |supported-versions| |supported-implementations|

.. |docs| image:: https://img.shields.io/github/deployments/psyplot/psy-maps/github-pages
    :alt: Documentation
    :target: http://psyplot.github.io/psy-maps/

.. |circleci| image:: https://circleci.com/gh/psyplot/psy-maps/tree/master.svg?style=svg
    :alt: CircleCI
    :target: https://circleci.com/gh/psyplot/psy-maps/tree/master

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/rd733xj3tfrk4tot/branch/master?svg=true
    :alt: AppVeyor
    :target: https://ci.appveyor.com/project/psyplot/psy-maps

.. |codecov| image:: https://codecov.io/gh/psyplot/psy-maps/branch/master/graph/badge.svg
    :alt: Coverage
    :target: https://codecov.io/gh/psyplot/psy-maps

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

.. |github| image:: https://img.shields.io/github/release/psyplot/psy-simple.svg
    :target: https://github.com/psyplot/psy-simple/releases/latest
    :alt: Latest github release


.. end-badges

Welcome to the psyplot plugin for visualizations on a map. This package uses the
cartopy_ package to project the plots that are made with the psy-simple_ plugin
to an earth-referenced grid. It's main plot methods are the
mapplot_ and mapvector_ plot methods which can plot
rectangular and triangular plots.

See the full documentation on
`psyplot.github.io/psy-maps/ <http://psyplot.github.io/psy-maps>`__ for all
`plot methods`_, and checkout the examples_.


.. _cartopy: http://scitools.org.uk/cartopy
.. _mapplot: http://psyplot.github.io/psy-maps/generated/psyplot.project.plot.mapplot.html#psyplot.project.plot.mapplot
.. _mapvector: http://psyplot.github.io/psy-maps/generated/psyplot.project.plot.mapvector.html#psyplot.project.plot.mapvector
.. _psy-simple: http://psyplot.github.io/psy-simple/
.. _plot methods: http://psyplot.github.io/psy-maps/en/latest/plot_methods
.. _examples: http://psyplot.github.io/examples/


Copyright
---------
Copyright © 2021 Helmholtz-Zentrum Hereon, 2020-2021 Helmholtz-Zentrum
Geesthacht, 2016-2021 University of Lausanne

psy-maps and is released under the GNU LGPL-3.O license.
See COPYING and COPYING.LESSER in the root of the repository for full
licensing details.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License version 3.0 as
published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU LGPL-3.0 license for more details.

You should have received a copy of the GNU LGPL-3.0 license
along with this program.  If not, see https://www.gnu.org/licenses/.
