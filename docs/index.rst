.. SPDX-FileCopyrightText: 2021-2024 Helmholtz-Zentrum hereon GmbH
..
.. SPDX-License-Identifier: CC-BY-4.0

.. psy-maps documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to psy-maps's documentation!
====================================

|CI|
|Code coverage|
|Latest Release|
|PyPI version|
|Code style: black|
|Imports: isort|
|PEP8|
|REUSE status|

.. rubric:: Psyplot plugin for visualization on a map

Welcome to the psyplot plugin for visualizations on a map. This package uses the
cartopy_ package to project the plots that are made with the psy-simple_ plugin
to an earth-referenced grid. It's main plot methods are the
:attr:`~psyplot.project.ProjectPlotter.mapplot` and
:attr:`~psyplot.project.ProjectPlotter.mapvector` plot methods which can plot
rectangular and triangular 2-dimensional data.

See the :ref:`plot_methods` for more information.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installing
   plot_methods
   api
   contributing


How to cite this software
-------------------------

.. card:: Please do cite this software!

   .. tab-set::

      .. tab-item:: APA

         .. citation-info::
            :format: apalike

      .. tab-item:: BibTex

         .. citation-info::
            :format: bibtex

      .. tab-item:: RIS

         .. citation-info::
            :format: ris

      .. tab-item:: Endnote

         .. citation-info::
            :format: endnote

      .. tab-item:: CFF

         .. citation-info::
            :format: cff


License information
-------------------
Copyright © 2021-2024 Helmholtz-Zentrum hereon GmbH

The source code of psy-maps is licensed under
LGPL-3.0-only.

If not stated otherwise, the contents of this documentation is licensed under
CC-BY-4.0.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


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
.. TODO: uncomment the following line when the package is registered at https://api.reuse.software
.. .. |REUSE status| image:: https://api.reuse.software/badge/codebase.helmholtz.cloud/psyplot/psy-maps
..    :target: https://api.reuse.software/info/codebase.helmholtz.cloud/psyplot/psy-maps
