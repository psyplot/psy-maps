v1.3.1
======

Added
-----
- ``scipy`` is now a run dependency of psy-maps (see
  `#27 <https://github.com/psyplot/psy-maps/issues/27>`__)

Fixed
-----
- fixed a bug when using ``plot="poly"`` for data on non-standard projections
  (i.e. anything that is not ``cartopy.crs.PlateCarree``). See
  `#29 <https://github.com/psyplot/psy-maps/pull/29>`__).
- fixed plotting of data with 3D bounds (see
  `#30 <https://github.com/psyplot/psy-maps/pull/30>`__)

v1.3.0
======
New GUI widgets and better projection support

Added
-----
* The ``xgrid`` and ``ygrid`` formatoptions now have a new widget in the GUI
  (see `#17 <https://github.com/psyplot/psy-maps/pull/17>`__)
* The ``lsm`` formatoption now supports a multitude of different options. You
  can specify a land color, and ocean color and the coast lines color. These
  settings can now also be set through the psyplot GUI
  (see `#17 <https://github.com/psyplot/psy-maps/pull/17>`__).
* a new ``background`` formatoption has been implemented that allows to set the
  facecolor of the axes (i.e. the background color for the plot)
* compatibility for cartopy 0.18 (see `#14 <https://github.com/psyplot/psy-maps/pull/14>`__)
* all plotmethods now have a ``transpose`` formatoption that can be used if the
  order of dimensions in the data is ``(x, y)`` rather than ``(y, x)``
* the `transform` and `projection` formatoptions now automatically decode the
  ``'grid_mappings'`` attribute following the `CF-conventions <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#appendix-grid-mappings>`__,
  see `#5 <https://github.com/psyplot/psy-maps/pull/5>`__,
  `#10 <https://github.com/psyplot/psy-maps/pull/10>`__)
* the ``projection`` and ``transform`` formatoptions now also support a `rotated`
  value to use a rotated pole projection

Changed
-------
* A bug has been fixed visualizing unstructured grid cells at the poles (see
  `#23 <https://github.com/psyplot/psy-maps/pull/23>`__)
* the ``lsm`` formatoptions value is now a dictionary. Old values, such as
  the string ``'10m'`` or ``['10m', 1.0]`` are still valid and will be converted
  to a dictionary (see `#17 <https://github.com/psyplot/psy-maps/pull/17>`__).
* the value ``None`` for the ``map_extent`` formatoption now triggers a
  call of the :meth:`~matplotlib.axes._base.AxesBase.autoscale` of the axes,
  see `#12 <https://github.com/psyplot/psy-maps/pull/12>`__. Before, it was
  bound to the ``lonlatbox`` value which made problems for regional files
  (see `#11 <https://github.com/psyplot/psy-maps/pull/11>`__). To retain the
  old behaviour of the ``map_extent`` formatoption, use ``map_extent='data'``
* psy-maps has been moved from https://github.com/Chilipp/psy-maps to https://github.com/psyplot/psy-maps,
  see `#4 <https://github.com/psyplot/psy-maps/pull/4>`__
* the default values for the `transform` and `projection` formatoptions are now
  ``'cf'`` (see `#5 <https://github.com/psyplot/psy-maps/pull/5>`__)
* ``clat`` now always takes the mean latitude of the data if the formatoption
  value is None and the data does not span the entire latitudinal range. At the
  same time, ``clon`` now takes the mean longitude of the data if the
  formatoption value is None and the data does not span the entire longitudinal
  range (see `#8 <https://github.com/psyplot/psy-maps/pull/8>`__)
* a bug was fixed for displaying data in the statusbar if the coordinate has
  units in degree (see https://github.com/psyplot/psy-view/issues/6)

v1.2.0
======
Added
-----
* The ``mapplot`` plotmethod now also supports unstructured data of any shape
  (see `issue#6 <https://github.com/psyplot/psyplot/issues/6>`__)

Changed
-------
* The ``lonlatbox`` formatoption now selects a subset of the unstructured data
  to lower the size of the data array. Previously, data points outside the
  specified `lonlatbox` where simply set to NaN

v1.1.0
======
Added
-----
* Changelog
* ``stock_img`` formatoption for map plots (see the
  `docs <https://psyplot.readthedocs.io/projects/psy-maps/en/latest/api/psy_maps.plotters.html#psy_maps.plotters.FieldPlotter.stock_img>`__)
* Added ``'stereo'`` and ``'near'`` projections for the
  `projection <https://psyplot.readthedocs.io/projects/psy-maps/en/latest/api/psy_maps.plotters.html#psy_maps.plotters.FieldPlotter.projection>`__
  formatoption
* The ``lonlatbox`` and ``map_extent`` formatoption keywords now also accepts
  a combination of floats and strings
* When displaying all longitudes but not all latitudes, the
  ``map_extent`` keyword now adjusts the boundary of the map to keep it
  circular for Stereographic and Orthographic projections

Changed
-------
* Fixed bugs with displaying circumpolar data and stereographic and
  orthographic projections
