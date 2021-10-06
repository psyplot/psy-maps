.. _install:

.. highlight:: bash

Installation
============

How to install
--------------

Installation using conda
^^^^^^^^^^^^^^^^^^^^^^^^
We highly recommend to use conda_ for installing psy-maps.

After downloading the `miniconda installer`_, you can install psy-maps simply
via::

    $ conda install -c conda-forge psy-maps

.. _miniconda installer: https://conda.io/en/latest/miniconda.html
.. _conda: http://conda.io/

Installation using pip
^^^^^^^^^^^^^^^^^^^^^^
If you do not want to use conda for managing your python packages, you can also
use the python package manager ``pip`` and install via::

    $ pip install psy-maps

Note however, that you should install cartopy_ beforehand.

.. _cartopy: http://scitools.org.uk/cartopy

Running the tests
-----------------
First, clone out the github_ repository. First, install pytest_ and create the
reference figures via::

    $ pytest --ref

After that, you can run::

    $ pytest


.. _pytest: https://pytest.org/latest/contents.html
.. _github: https://github.com/psyplot/psy-maps
