.. SPDX-FileCopyrightText: 2021-2024 Helmholtz-Zentrum hereon GmbH
..
.. SPDX-License-Identifier: CC-BY-4.0

========================================================
psy-maps: The psyplot plugin for visualizations on a map
========================================================

.. start-badges

|CI|
|Code coverage|
|Latest Release|
|PyPI version|
|Code style: black|
|Imports: isort|
|PEP8|
|REUSE status|

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

.. |CI| image:: https://codebase.helmholtz.cloud/psyplot/psy-maps/badges/main/pipeline.svg
   :target: https://codebase.helmholtz.cloud/psyplot/psy-maps/-/pipelines?page=1&scope=all&ref=main
.. |Code coverage| image:: https://codebase.helmholtz.cloud/psyplot/psy-maps/badges/main/coverage.svg
   :target: https://codebase.helmholtz.cloud/psyplot/psy-maps/-/graphs/main/charts
.. |Latest Release| image:: https://codebase.helmholtz.cloud/psyplot/psy-maps/-/badges/release.svg
   :target: https://codebase.helmholtz.cloud/psyplot/psy-maps
.. |PyPI version| image:: https://img.shields.io/pypi/v/psy-maps.svg
   :target: https://pypi.python.org/pypi/psy-maps/
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |Imports: isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/
.. |PEP8| image:: https://img.shields.io/badge/code%20style-pep8-orange.svg
   :target: https://www.python.org/dev/peps/pep-0008/
.. |REUSE status| image:: https://api.reuse.software/badge/codebase.helmholtz.cloud/psyplot/psy-maps
   :target: https://api.reuse.software/info/codebase.helmholtz.cloud/psyplot/psy-maps
