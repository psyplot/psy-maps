"""Test module of the :mod:`psyplot.plotter.maps` module"""
import os
import re
import unittest
import six
from functools import wraps
from itertools import starmap, repeat, chain
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from psyplot.utils import _TempBool
from psy_maps.plotters import (
    FieldPlotter, VectorPlotter, rcParams, CombinedPlotter, InteractiveList)
import test_base as tb
import _base_testing as bt
from psyplot import ArrayList, open_dataset
import psyplot.project as psy
from psyplot.compat.pycompat import filter

from test_base import bold


class MapReferences(object):
    """Abstract base class for map reference plots"""

    def ref_datagrid(self, close=True):
        """Create reference file for datagrid formatoption

        Create reference file for
        :attr:`~psyplot.plotter.simple.Simple2DPlotter.datagrid`
        formatoption"""
        if self.plot_type[:6] == 'simple':
            kwargs = dict(xlim=(0, 40), ylim=(0, 40))
        else:
            kwargs = dict(lonlatbox='Europe')
        sp = self.plot(**kwargs)
        sp.update(datagrid='k-')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('datagrid')))
        if close:
            sp.close(True, True, True)

    def ref_cmap(self, close=True):
        """Create reference file for cmap formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.simple.Simple2DPlotter.cmap`
        formatoption"""
        sp = self.plot(cmap='RdBu')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('cmap')))
        if close:
            sp.close(True, True, True)

    def ref_cbar(self, close=True):
        """Create reference file for cbar formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.simple.Simple2DPlotter.cbar`
        formatoption"""
        sp = self.plot(cbar=['fb', 'fr', 'fl', 'ft', 'b', 'r'])
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('cbar')))
        if close:
            sp.close(True, True, True)

    def ref_miss_color(self, close=True):
        """Create reference file for miss_color formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.simple.Simple2DPlotter.miss_color`
        formatoption"""
        if self.plot_type[:3] == 'map':
            kwargs = {'projection': 'ortho', 'grid_labels': False}
        else:
            kwargs = {}
        sp = self.plot(maskless=280, miss_color='0.9', **kwargs)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('miss_color')))
        if close:
            sp.close(True, True, True)

    def ref_cbarspacing(self, close=True):
        """Create reference file for cbarspacing formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.simple.Simple2DPlotter.cbarspacing`
        formatoption"""
        if self.plot_type.endswith('vector') or getattr(self, 'vector_mode',
                                                        False):
            kwargs = dict(
                bounds=np.arange(0, 1.45, 0.1).tolist() + np.linspace(
                    1.5, 13.5, 7, endpoint=True).tolist() + np.arange(
                        13.6, 15.05, 0.1).tolist(), color='absolute')
        else:
            kwargs = dict(bounds=list(range(235, 250)) + np.linspace(
                250, 295, 7, endpoint=True).tolist() + list(range(296, 310)))
        sp = self.plot(
            cbarspacing='proportional', cticks='rounded',
            **kwargs)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('cbarspacing')))
        if close:
            sp.close(True, True, True)

    def ref_lonlatbox(self, close=True):
        """Create reference file for lonlatbox formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.lonlatbox` formatoption"""
        sp = self.plot()
        sp.update(lonlatbox='Europe|India')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('lonlatbox')))
        if close:
            sp.close(True, True, True)

    def ref_map_extent(self, close=True):
        """Create reference file for map_extent formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.map_extent` formatoption"""
        sp = self.plot()
        sp.update(map_extent='Europe|India')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('map_extent')))
        if close:
            sp.close(True, True, True)

    def ref_lsm(self, close=True):
        """Create reference file for lsm formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.lsm` formatoption"""
        sp = self.plot()
        sp.update(lsm=False)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('lsm')))
        sp.update(lsm=['110m', 2.0])
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('lsm2')))
        if close:
            sp.close(True, True, True)

    def ref_projection(self, close=True):
        """Create reference file for projection formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.projection` formatoption"""
        import cartopy.crs as ccrs
        sp = self.plot()
        sp.update(projection='ortho', grid_labels=False)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('projection1')))
        sp.update(projection='northpole')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('projection2')))
        sp.update(projection=ccrs.LambertConformal())
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('projection3')))
        if close:
            sp.close(True, True, True)

    def ref_map_grid(self, close=True):
        """Create reference file for xgrid formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.xgrid` (and others)
        formatoption"""
        sp = self.plot()
        sp.update(xgrid=False, ygrid=False)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid1')))
        sp.update(xgrid='rounded', ygrid=['data', 20])
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid2')))
        sp.update(xgrid=True, ygrid=True, grid_color='w')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid3')))
        sp.update(xgrid=True, ygrid=True, grid_color='k', grid_settings={
            'linestyle': 'dotted'})
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid4')))
        if close:
            sp.close(True, True, True)


class FieldPlotterTest(tb.BasePlotterTest, MapReferences):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class"""

    plot_type = 'map'

    def plot(self, **kwargs):
        name = kwargs.pop('name', self.var)
        return psy.plot.mapplot(self.ncfile, name=name, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.ds = open_dataset(cls.ncfile)
        cls.data = ArrayList.from_dataset(
            cls.ds, t=0, z=0, name=cls.var, auto_update=True)[0]
        cls.plotter = FieldPlotter(cls.data)
        cls.create_dirs()

    @unittest.skip("axiscolor formatoption not implemented")
    def test_axiscolor(self):
        pass

    def test_extend(self):
        """Test extend formatoption"""
        self.update(extend='both')
        self.assertEqual(self.plotter.cbar.cbars['b'].extend, 'both')
        self.update(extend='min')
        self.assertEqual(self.plotter.cbar.cbars['b'].extend, 'min')
        self.update(extend='neither')
        self.assertEqual(self.plotter.cbar.cbars['b'].extend, 'neither')

    @property
    def _minmax_cticks(self):
        return np.round(
            np.linspace(self.data.min().values, self.data.max().values, 11,
                        endpoint=True), decimals=2).tolist()

    def test_cticks(self):
        """Test cticks, cticksize, ctickweight, ctickprops formatoptions"""
        cticks = self._minmax_cticks
        self.update(cticks='minmax')
        cbar = self.plotter.cbar.cbars['b']
        self.assertEqual(list(map(
            lambda t: float(t.get_text()), cbar.ax.get_xticklabels())), cticks)
        self.update(cticklabels='%3.1f')
        test_ticks = np.round(
            list(map(lambda t: float(t.get_text()),
                     cbar.ax.get_xticklabels())),
            1).tolist()
        self.assertAlmostArrayEqual(test_ticks, cticks, atol=0.1)
        self.update(cticksize=20, ctickweight='bold', ctickprops={
            'labelcolor': 'r'})
        texts = cbar.ax.get_xticklabels()
        n = len(texts)
        self.assertEqual([t.get_weight() for t in texts], [bold] * n)
        self.assertEqual([t.get_size() for t in texts], [20] * n)
        self.assertEqual([t.get_color() for t in texts], ['r'] * n)

    def test_clabel(self):
        """Test clabel, clabelsize, clabelweight, clabelprops formatoptions"""
        def get_clabel():
            return self.plotter.cbar.cbars['b'].ax.xaxis.get_label()
        self._label_test('clabel', get_clabel)
        label = get_clabel()
        self.update(clabelsize=22, clabelweight='bold',
                    clabelprops={'ha': 'left'})
        self.assertEqual(label.get_size(), 22)
        self.assertEqual(label.get_weight(), bold)
        self.assertEqual(label.get_ha(), 'left')

    def test_datagrid(self, *args):
        """Test datagrid formatoption"""
        self.update(lonlatbox='Europe', datagrid='k-')
        self.compare_figures(next(iter(args), self.get_ref_file('datagrid')))

    def test_cmap(self, *args):
        """Test colormap (cmap) formatoption"""
        self.update(cmap='RdBu')
        fname = next(iter(args), self.get_ref_file('cmap'))
        self.compare_figures(fname)
        self.update(cmap=plt.get_cmap('RdBu'))
        self.compare_figures(fname)

    def test_cbar(self, *args):
        """Test colorbar (cbar) formatoption"""
        self.update(cbar=['fb', 'fr', 'fl', 'ft', 'b', 'r'])
        self.compare_figures(next(iter(args), self.get_ref_file('cbar')))

    def test_bounds(self):
        """Test bounds formatoption"""
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.round(np.linspace(240, 310, 11, endpoint=True), 2).tolist())
        self.update(bounds='minmax')
        bounds = [241.03, 247.84, 254.65, 261.46, 268.27, 275.08, 281.9,
                  288.71, 295.52, 302.33, 309.14]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.round(np.linspace(245, 305, 5, endpoint=True), 2).tolist())

    def test_miss_color(self, *args):
        """Test miss_color formatoption"""
        # We have to change the projection because cartopy (0.13.0) does not
        # support the :meth:`matplotlib.colors.ColorMap.set_bad` method for
        # rectangular projections
        self.update(maskless=280, miss_color='0.9', projection='ortho',
                    grid_labels=False)
        self.compare_figures(next(iter(args), self.get_ref_file('miss_color')))

    def test_cbarspacing(self, *args):
        """Test cbarspacing formatoption"""
        self.update(
            cbarspacing='proportional', cticks='rounded',
            bounds=list(range(235, 250)) + np.linspace(
                250, 295, 7, endpoint=True).tolist() + list(range(296, 310)))
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('cbarspacing')))

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        self.update(lonlatbox='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-32.0, 97.0, -8.0, 81.0),
            repeat(5), repeat("Failed to set the extent to Europe and India!"))
            ))
        # test whether latitudes and longitudes succeded
        msg = "Failed to fit into lonlatbox limits for %s of %s."
        if isinstance(self.plotter.plot_data, InteractiveList):
            all_data = self.plotter.plot_data
        else:
            all_data = [self.plotter.plot_data]
        for data in all_data:
            self.assertGreaterEqual(data.lon.values.min(), -32.0,
                                    msg=msg % ('longitude', 'minimum'))
            self.assertLessEqual(data.lon.values.max(), 97.0,
                                 msg=msg % ('longitude', 'maximum'))
            self.assertGreaterEqual(data.lat.values.min(), -8.0,
                                    msg=msg % ('latitude', 'minimum'))
            self.assertLessEqual(data.lat.values.max(), 81.0,
                                 msg=msg % ('latitude', 'maximum'))
        self.compare_figures(next(iter(args), self.get_ref_file('lonlatbox')))

    def test_map_extent(self, *args):
        """Test map_extent formatoption"""
        self.update(map_extent='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-32.0, 97.0, -8.0, 81.0),
            repeat(5), repeat("Failed to set the extent to Europe and India!"))
            ))
        self.compare_figures(next(iter(args), self.get_ref_file('map_extent')))

    def test_lsm(self, *args):
        """Test land-sea-mask formatoption"""
        self.update(lsm=False)
        self.compare_figures(next(iter(args), self.get_ref_file('lsm')))
        self.update(lsm=['110m', 2.0])
        self.compare_figures(next(iter(args), self.get_ref_file('lsm2')))

    def test_projection(self, *args):
        """Test projection formatoption"""
        self.update(projection='ortho', grid_labels=False)
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('projection1')))
        self.update(projection='northpole')
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('projection2')))
        self.update(projection=ccrs.LambertConformal())
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('projection3')))

    def test_grid(self, *args):
        """Test xgrid, ygrid, grid_color, grid_labels, grid_settings fmts"""
        self.update(xgrid=False, ygrid=False)
        self.compare_figures(next(iter(args), self.get_ref_file('grid1')))
        self.update(xgrid='rounded', ygrid=['data', 20])
        self.compare_figures(next(iter(args), self.get_ref_file('grid2')))
        self.update(xgrid=True, ygrid=True, grid_color='w')
        self.compare_figures(next(iter(args), self.get_ref_file('grid3')))
        self.update(xgrid=True, ygrid=True, grid_color='k',
                    grid_settings={'linestyle': 'dotted'})
        self.compare_figures(next(iter(args), self.get_ref_file('grid4')))

    def test_clon(self):
        """Test clon formatoption"""
        self.update(clon=180.)
        self.assertEqual(self.plotter.ax.projection.proj4_params['lon_0'],
                         180.)
        self.update(clon='India')
        self.assertEqual(self.plotter.ax.projection.proj4_params['lon_0'],
                         82.5)

    def test_clat(self):
        """Test clat formatoption"""
        self.update(projection='ortho', clat=60., grid_labels=False)
        self.assertEqual(self.plotter.ax.projection.proj4_params['lat_0'],
                         60.)
        self.update(clat='India')
        self.assertEqual(self.plotter.ax.projection.proj4_params['lat_0'],
                         13.5)

    def test_grid_labelsize(self):
        """Test grid_labelsize formatoption"""
        self.update(grid_labelsize=20)
        texts = list(chain(self.plotter.xgrid._gridliner.xlabel_artists,
                           self.plotter.ygrid._gridliner.ylabel_artists))
        self.assertEqual([t.get_size() for t in texts], [20] * len(texts))


class FieldPlotterContourTest(FieldPlotterTest):

    plot_type = 'map_contour'

    @classmethod
    def setUpClass(cls):
        rcParams[FieldPlotter().plot.default_key] = 'contourf'
        rcParams[FieldPlotter().lonlatbox.default_key] = [-180, 180, -90, 90]
        super(FieldPlotterContourTest, cls).setUpClass()

    @unittest.skip('Extend keyword not implemented')
    def test_extend(self):
        pass

    @unittest.skip('miss_color keyword not implemented')
    def test_miss_color(self):
        pass

    @unittest.skip('miss_color keyword not implemented')
    def ref_miss_color(self):
        pass


class TestProjectedLonlatbox(unittest.TestCase):
    """A test class for testing the lonlatbox of a non-PlateCarree projection
    """

    def test_lonlatbox(self):
        sp = psy.plot.mapplot(os.path.join(bt.test_dir, 'Stockholm.nc'),
                              name='Population', transform='moll')
        ax = sp.plotters[0].ax
        self.assertEqual(np.round(ax.get_extent(), 2).tolist(),
                         [17.66, 18.39, 59.1, 59.59])
        sp.update(lonlatbox=[17.8, 18.2, 59.2, 59.4])
        self.assertEqual(
            np.round(ax.get_extent(ccrs.PlateCarree()), 2).tolist(),
            [17.8, 18.2, 59.2, 59.4])


class VectorPlotterTest(FieldPlotterTest, MapReferences):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class"""

    plot_type = 'mapvector'

    var = ['u', 'v']

    def plot(self, **kwargs):
        sp = psy.plot.mapvector(self.ncfile, name=[self.var], **kwargs)
        return sp

    @unittest.skip("miss_color formatoption not implemented")
    def ref_miss_color(self, close=True):
        pass

    def ref_arrowsize(self, close=True):
        """Create reference file for arrowsize formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.arrowsize` (and others)
        formatoption"""
        sp = self.plot(arrowsize=100.0)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('arrowsize')))
        if close:
            sp.close(True, True, True)

    def ref_density(self, close=True):
        """Create reference file for density formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.density` (and others)
        formatoption"""
        sp = self.plot()
        # We do not include the density in the initial plot because there the
        # map_extent is not already considered
        sp.update(density=0.5)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('density')))
        if close:
            sp.close(True, True, True)

    @classmethod
    def setUpClass(cls):
        cls.ds = open_dataset(cls.ncfile)
        rcParams[VectorPlotter().lonlatbox.default_key] = 'Europe'
        rcParams[VectorPlotter().color.default_key] = 'absolute'
        cls.data = ArrayList.from_dataset(
            cls.ds, t=0, z=0, name=[cls.var], auto_update=True)[0]
        cls.data.attrs['long_name'] = 'absolute wind speed'
        cls.data.name = 'wind'
        cls.plotter = VectorPlotter(cls.data)
        cls.create_dirs()
        cls._color_fmts = cls.plotter.fmt_groups['colors']

    @unittest.skip("Not supported")
    def test_maskless(self):
        pass

    @unittest.skip("Not supported")
    def test_maskgreater(self):
        pass

    @unittest.skip("Not supported")
    def test_maskleq(self):
        pass

    @unittest.skip("Not supported")
    def test_maskgeq(self):
        pass

    @unittest.skip("Not supported")
    def test_maskbetween(self):
        pass

    @unittest.skip("Not supported")
    def test_miss_color(self):
        pass

    def test_bounds(self):
        """Test bounds formatoption"""
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(0, 15, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [0.66, 1.74, 2.81, 3.89, 4.96, 6.04, 7.11, 8.19, 9.26, 10.34,
                  11.41]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(1.0, 10.0, 5, endpoint=True).tolist())

    def test_cbarspacing(self, *args):
        """Test cbarspacing formatoption"""
        self.update(
            cbarspacing='proportional', cticks='rounded', color='absolute',
            bounds=np.arange(0, 1.45, 0.1).tolist() + np.linspace(
                    1.5, 13.5, 7, endpoint=True).tolist() + np.arange(
                        13.6, 15.05, 0.1).tolist())
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('cbarspacing')))

    @unittest.skipIf(
        six.PY34, "The axes size changes using the arrowsize formatoption")
    def test_arrowsize(self, *args):
        """Test arrowsize formatoption"""
        self.update(arrowsize=100.0)
        self.compare_figures(next(iter(args), self.get_ref_file('arrowsize')))

    @unittest.skipIf(
        six.PY34, "The axes size changes when using the density formatoption")
    def test_density(self, *args):
        """Test density formatoption"""
        self.update(density=0.5)
        self.compare_figures(next(iter(args), self.get_ref_file('density')))

    @property
    def _minmax_cticks(self):
        speed = (self.plotter.plot_data.values[0]**2 +
                 self.plotter.plot_data.values[1]**2) ** 0.5
        speed = speed[~np.isnan(speed)]
        return np.round(
            np.linspace(speed.min(), speed.max(), 11, endpoint=True),
            decimals=2).tolist()


class StreamVectorPlotterTest(VectorPlotterTest):
    """Test case for stream plot of :class:`psyplot.plotter.maps.VectorPlotter`
    """

    @classmethod
    def setUpClass(cls):
        rcParams[VectorPlotter().plot.default_key] = 'stream'
        return super(StreamVectorPlotterTest, cls).setUpClass()

    def get_ref_file(self, identifier):
        return super(StreamVectorPlotterTest, self).get_ref_file(
            identifier + '_stream')

    def ref_arrowsize(self, *args):
        """Create reference file for arrowsize formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.arrowsize` (and others)
        formatoption"""
        sp = self.plot(arrowsize=2.0)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('arrowsize')))

    def ref_arrowstyle(self, *args):
        """Create reference file for arrowstyle formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.arrowstyle` (and others)
        formatoption"""
        sp = self.plot(arrowsize=2.0, arrowstyle='fancy')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('arrowstyle')))

    def test_arrowsize(self, *args):
        """Test arrowsize formatoption"""
        self.update(arrowsize=2.0)
        self.compare_figures(next(iter(args), self.get_ref_file('arrowsize')))

    def test_arrowstyle(self, *args):
        """Test arrowstyle formatoption"""
        self.update(arrowsize=2.0, arrowstyle='fancy')
        self.compare_figures(next(iter(args), self.get_ref_file('arrowstyle')))

    def test_density(self, *args):
        """Test density formatoption"""
        self.update(density=0.5)
        self.compare_figures(next(iter(args), self.get_ref_file('density')))


def _do_from_both(func):
    """Call the given `func` only from :class:`FieldPlotterTest and
    :class:`VectorPlotterTest`"""
    func.__doc__ = getattr(VectorPlotterTest, func.__name__).__doc__

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        getattr(FieldPlotterTest, func.__name__)(self, *args, **kwargs)
        if hasattr(self, 'plotter'):
            self.plotter.update(todefault=True)
        with self.vector_mode:
            getattr(VectorPlotterTest, func.__name__)(self, *args, **kwargs)

    return wrapper


def _in_vector_mode(func):
    """Call the given `func` only from:class:`VectorPlotterTest`"""
    func.__doc__ = getattr(VectorPlotterTest, func.__name__).__doc__

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.vector_mode:
            getattr(VectorPlotterTest, func.__name__)(self, *args, **kwargs)

    return wrapper


class _CombinedPlotterData(object):
    """Descriptor that returns the data"""
    # Note: We choose to use a descriptor rather than a usual property because
    # it shall also work for class objects and not only instances

    def __get__(self, instance, owner):
        if instance is None:
            return owner._data
        if instance.vector_mode:
            return instance._data[1]
        return instance._data[0]

    def __set__(self, instance, value):
        instance._data = value


class CombinedPlotterTest(VectorPlotterTest):
    """Test case for stream plot of :class:`psyplot.plotter.maps.CombinedPlotter`
    """

    plot_type = 'mapcombined'

    data = _CombinedPlotterData()

    var = ['t2m', ['u', 'v']]

    @property
    def vector_mode(self):
        """:class:`bool` indicating whether a vector specific formatoption is
        tested or not"""
        try:
            return self._vector_mode
        except AttributeError:
            self._vector_mode = _TempBool(False)
            return self._vector_mode

    @vector_mode.setter
    def vector_mode(self, value):
        self.vector_mode.value = bool(value)

    def compare_figures(self, fname, **kwargs):
        kwargs.setdefault('tol', 10)
        return super(CombinedPlotterTest, self).compare_figures(
            fname, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.ds = open_dataset(cls.ncfile)
        rcParams[CombinedPlotter().lonlatbox.default_key] = 'Europe'
        rcParams[CombinedPlotter().vcmap.default_key] = 'winter'
        cls._data = ArrayList.from_dataset(
            cls.ds, t=0, z=0, name=[cls.var], auto_update=True,
            prefer_list=True)[0]
        cls._data.attrs['long_name'] = 'Temperature'
        cls._data.attrs['name'] = 't2m'
        cls.plotter = CombinedPlotter(cls._data)
        cls.create_dirs()
        cls._color_fmts = cls.plotter.fmt_groups['colors']

    def tearDown(self):
        self._data.update(t=0, todefault=True, replot=True)

    def plot(self, **kwargs):
        if self.vector_mode:
            color_fmts = psy.plot.mapvector.plotter_cls().fmt_groups['colors']
            if color_fmts.intersection(kwargs):
                kwargs.setdefault('color', 'absolute')
            kwargs = self._rename_fmts(kwargs)
        sp = psy.plot.mapcombined(self.ncfile, name=[self.var],
                                  **kwargs)
        return sp

    def _rename_fmts(self, kwargs):
        def check_key(key):
            if not any(re.match('v' + key, fmt) for fmt in vcolor_fmts):
                return key
            else:
                return 'v' + key
        vcolor_fmts = {
            fmt for fmt in chain(
                psy.plot.mapcombined.plotter_cls().fmt_groups['colors'],
                ['ctick|clabel']) if fmt.startswith('v')}
        return {check_key(key): val for key, val in kwargs.items()}

    def update(self, *args, **kwargs):
        if self.vector_mode and (
                self._color_fmts.intersection(kwargs) or any(
                    re.match('ctick|clabel', fmt) for fmt in kwargs)):
            kwargs.setdefault('color', 'absolute')
            kwargs = self._rename_fmts(kwargs)
        super(VectorPlotterTest, self).update(*args, **kwargs)

    def get_ref_file(self, identifier):
        if self.vector_mode:
            identifier += '_vector'
        return super(CombinedPlotterTest, self).get_ref_file(identifier)

    @property
    def _minmax_cticks(self):
        if not self.vector_mode:
            return np.round(
                np.linspace(self.plotter.plot_data[0].values.min(),
                            self.plotter.plot_data[0].values.max(), 11,
                            endpoint=True), decimals=2).tolist()
        speed = (self.plotter.plot_data[1].values[0]**2 +
                 self.plotter.plot_data[1].values[1]**2) ** 0.5
        return np.round(
            np.linspace(speed.min(), speed.max(), 11, endpoint=True),
            decimals=2).tolist()

    def ref_density(self, close=True):
        """Create reference file for density formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.density` (and others)
        formatoption"""
        # we have to make sure, that the color keyword is not set to 'absolute'
        # because it does not work for quiver plots
        sp = self.plot(density=0.5, color='k')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('density')))
        if close:
            sp.close(True, True, True)

    @_do_from_both
    def ref_cbar(self, close=True):
        pass

    @unittest.skip('Buggy for unknown reason')
    def test_map_extent(self):
        # TODO: fix this
        pass

    def ref_cbarspacing(self, close=True):
        """Create reference file for cbarspacing formatoption"""
        kwargs = dict(bounds=list(range(245, 255)) + np.linspace(
                255, 280, 6, endpoint=True).tolist() + list(range(281, 290)))
        sp = self.plot(
            cbarspacing='proportional', cticks='rounded',
            **kwargs)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('cbarspacing')))
        with self.vector_mode:
            VectorPlotterTest.ref_cbarspacing(self, close=close)
        if close:
            sp.close(True, True, True)

    @_do_from_both
    def ref_cmap(self, close=True):
        pass

    def ref_miss_color(self, close=True):
        FieldPlotterTest.ref_miss_color(self, close)

    @_in_vector_mode
    def ref_arrowsize(self, *args, **kwargs):
        pass

    def _label_test(self, key, label_func, has_time=True):
        kwargs = {
            key: "Test plot at %Y-%m-%d, {tinfo} o'clock of %(long_name)s"}
        self.update(**kwargs)
        t_str = '1979-01-31, 18:00' if has_time else '%Y-%m-%d, %H:%M'
        self.assertEqual(
            u"Test plot at %s o'clock of %s" % (
                t_str, self.data.attrs.get('long_name', 'Temperature')),
            label_func().get_text())
        self._data.update(t=1)
        t_str = '1979-02-28, 18:00' if has_time else '%Y-%m-%d, %H:%M'
        self.assertEqual(
            u"Test plot at %s o'clock of %s" % (
                t_str, self.data.attrs.get('long_name', 'Temperature')),
            label_func().get_text())
        self._data.update(t=0)

    def test_miss_color(self, *args, **kwargs):
        FieldPlotterTest.test_miss_color(self, *args, **kwargs)

    @_do_from_both
    def test_cbar(self, *args, **kwargs):
        pass

    def test_cbarspacing(self, *args, **kwargs):
        """Test cbarspacing formatoption"""
        self.update(
            cbarspacing='proportional', cticks='rounded',
            bounds=list(range(245, 255)) + np.linspace(
                255, 280, 6, endpoint=True).tolist() + list(range(281, 290)))
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('cbarspacing')))
        self.plotter.update(todefault=True)
        with self.vector_mode:
            VectorPlotterTest.test_cbarspacing(self, *args, **kwargs)

    @_do_from_both
    def test_cmap(self, *args, **kwargs):
        pass

    @unittest.skipIf(
        six.PY34, "The axes size changes using the arrowsize formatoption")
    @_in_vector_mode
    def test_arrowsize(self):
        pass

    def test_bounds(self):
        """Test bounds formatoption"""
        # test bounds of scalar field
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(245, 290, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [248.07, 252.01, 255.96, 259.9, 263.85, 267.79, 271.74,
                  275.69, 279.63, 283.58, 287.52]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(250, 290, 5, endpoint=True).tolist())

        # test vector bounds
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(),
            np.linspace(0, 15, 11, endpoint=True).tolist())
        self.update(vbounds='minmax')
        bounds = [0.66, 1.74, 2.81, 3.89, 4.96, 6.04, 7.11, 8.19, 9.26, 10.34,
                  11.41]
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(), bounds)
        self.update(vbounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(),
            np.linspace(1.0, 10.0, 5, endpoint=True).tolist())

    def test_clabel(self):
        def get_clabel():
            return self.plotter.vcbar.cbars['b'].ax.xaxis.get_label()
        FieldPlotterTest.test_clabel(self)
        with self.vector_mode:
            self.update(color='absolute')
            self._label_test('vclabel', get_clabel)
            label = get_clabel()
            self.update(vclabelsize=22, vclabelweight='bold',
                        vclabelprops={'ha': 'left'})
            self.assertEqual(label.get_size(), 22)
            self.assertEqual(label.get_weight(), bold)
            self.assertEqual(label.get_ha(), 'left')


class CircumpolarFieldPlotterTest(FieldPlotterTest):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class for circumpolar
    grid"""

    grid_type = 'circumpolar'

    ncfile = os.path.join(bt.test_dir, 'circumpolar_test.nc')

    @classmethod
    def setUpClass(cls):
        rcParams['plotter.maps.projection'] = 'northpole'
        rcParams['plotter.maps.clat'] = 90
        super(CircumpolarFieldPlotterTest, cls).setUpClass()

    def ref_map_grid(self, close=True):
        """Create reference file for xgrid formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.xgrid` (and others)
        formatoption"""
        sp = self.plot(xgrid=False, ygrid=False)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid1')))
        sp.update(xgrid='rounded', ygrid=['data', 1000])
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid2')))
        sp.update(xgrid=True, ygrid=True, grid_color='w')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid3')))
        sp.update(xgrid=True, ygrid=True, grid_color='k', grid_settings={
            'linestyle': 'dotted'})
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid4')))
        if close:
            sp.close(True, True, True)

    def test_bounds(self):
        """Test bounds formatoption"""
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(240, 310, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [241.29, 247.68, 254.07, 260.47, 266.86, 273.25, 279.64,
                  286.03, 292.43, 298.82, 305.21]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(245, 305, 5, endpoint=True).tolist())

    def test_grid(self, *args):
        """Test xgrid, ygrid, grid_color, grid_labels, grid_settings fmts"""
        self.update(xgrid=False, ygrid=False)
        self.compare_figures(next(iter(args), self.get_ref_file('grid1')))
        self.update(xgrid='rounded', ygrid=['data', 1000])
        self.compare_figures(next(iter(args), self.get_ref_file('grid2')))
        self.update(xgrid=True, ygrid=True, grid_color='w')
        self.compare_figures(next(iter(args), self.get_ref_file('grid3')))
        self.update(xgrid=True, ygrid=True, grid_color='k',
                    grid_settings={'linestyle': 'dotted'})
        self.compare_figures(next(iter(args), self.get_ref_file('grid4')))

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            return coord.values[~np.isnan(data.values)]
        self.update(lonlatbox='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-55.6, 120.6,  -24.4, 85.9),
            repeat(1), repeat("Failed to set the extent to Europe and India!"))
            ))
        # test whether latitudes and longitudes succeded
        msg = "Failed to fit into lonlatbox limits for %s of %s."
        if isinstance(self.plotter.plot_data, InteractiveList):
            all_data = self.plotter.plot_data
        else:
            all_data = [self.plotter.plot_data]
        for data in all_data:
            self.assertGreaterEqual(get_unmasked(data.longitude).min(), -55.6,
                                    msg=msg % ('longitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.longitude).max(), 120.6,
                                 msg=msg % ('longitude', 'maximum'))
            self.assertGreaterEqual(get_unmasked(data.latitude).min(), -24.4,
                                    msg=msg % ('latitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.latitude).max(), 85.9,
                                 msg=msg % ('latitude', 'maximum'))
        self.compare_figures(next(iter(args), self.get_ref_file('lonlatbox')))

    def test_map_extent(self, *args):
        """Test map_extent formatoption"""
        self.update(map_extent='Europe|India')
        self.compare_figures(next(iter(args), self.get_ref_file('map_extent')))

    @unittest.skip('Not implemented for circumpolar grids')
    def ref_datagrid(self):
        pass

    @unittest.skip('Not implemented for circumpolar grids')
    def test_datagrid(self):
        pass


class CircumpolarVectorPlotterTest(VectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class for circumpolar
    grid"""

    grid_type = 'circumpolar'

    ncfile = os.path.join(bt.test_dir, 'circumpolar_test.nc')

    @classmethod
    def setUpClass(cls):
        rcParams['plotter.maps.projection'] = 'northpole'
        rcParams['plotter.maps.clat'] = 90
        rcParams['plotter.vector.arrowsize'] = 200
        rcParams['plotter.maps.lonlatbox'] = 'Europe'
        super(CircumpolarVectorPlotterTest, cls).setUpClass()

    def ref_map_grid(self, close=True):
        """Create reference file for xgrid formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.xgrid` (and others)
        formatoption"""
        sp = self.plot()
        sp.update(xgrid=False, ygrid=False)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid1')))
        sp.update(xgrid='rounded', ygrid=['data', 1000])
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid2')))
        sp.update(xgrid=True, ygrid=True, grid_color='w')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid3')))
        sp.update(xgrid=True, ygrid=True, grid_color='k', grid_settings={
            'linestyle': 'dotted'})
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid4')))
        if close:
            sp.close(True, True, True)

    def test_bounds(self):
        """Test bounds formatoption"""
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(0, 15, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [0.71, 1.74, 2.76, 3.79, 4.81, 5.84, 6.86, 7.89, 8.92, 9.94,
                  10.97]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.round(np.linspace(1.0, 9.5, 5, endpoint=True), 2).tolist())

    def test_grid(self, *args):
        """Test xgrid, ygrid, grid_color, grid_labels, grid_settings fmts"""
        self.update(xgrid=False, ygrid=False)
        self.compare_figures(next(iter(args), self.get_ref_file('grid1')))
        self.update(xgrid='rounded', ygrid=['data', 1000])
        self.compare_figures(next(iter(args), self.get_ref_file('grid2')))
        self.update(xgrid=True, ygrid=True, grid_color='w')
        self.compare_figures(next(iter(args), self.get_ref_file('grid3')))
        self.update(xgrid=True, ygrid=True, grid_color='k',
                    grid_settings={'linestyle': 'dotted'})
        self.compare_figures(next(iter(args), self.get_ref_file('grid4')))

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            return coord.values[~np.isnan(data[0].values)]
        self.update(lonlatbox='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-55.6, 120.6,  -24.4, 85.9),
            repeat(1), repeat("Failed to set the extent to Europe and India!"))
            ))
        # test whether latitudes and longitudes succeded
        msg = "Failed to fit into lonlatbox limits for %s of %s."
        if isinstance(self.plotter.plot_data, InteractiveList):
            all_data = self.plotter.plot_data
        else:
            all_data = [self.plotter.plot_data]
        for data in all_data:
            self.assertGreaterEqual(get_unmasked(data.longitude).min(), -55.6,
                                    msg=msg % ('longitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.longitude).max(), 120.6,
                                 msg=msg % ('longitude', 'maximum'))
            self.assertGreaterEqual(get_unmasked(data.latitude).min(), -24.4,
                                    msg=msg % ('latitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.latitude).max(), 85.9,
                                 msg=msg % ('latitude', 'maximum'))
        self.compare_figures(next(iter(args), self.get_ref_file('lonlatbox')))

    def test_map_extent(self, *args):
        """Test map_extent formatoption"""
        self.update(map_extent='Europe|India')
        self.compare_figures(next(iter(args), self.get_ref_file('map_extent')))

    @unittest.skip('Not implemented for circumpolar grids')
    def ref_datagrid(self):
        pass

    @unittest.skip('Not implemented for circumpolar grids')
    def test_datagrid(self):
        pass

    @unittest.skip('Not implemented for circumpolar grids')
    def test_density(self):
        pass

    @unittest.skip('Not implemented for circumpolar grids')
    def ref_density(self):
        pass


class CircumpolarCombinedPlotterTest(CombinedPlotterTest):
    """Test :class:`psyplot.plotter.maps.CombinedPlotter` class for circumpolar
    grid"""

    grid_type = 'circumpolar'

    ncfile = os.path.join(bt.test_dir, 'circumpolar_test.nc')

    @classmethod
    def setUpClass(cls):
        rcParams['plotter.maps.projection'] = 'northpole'
        rcParams['plotter.maps.clat'] = 90
        rcParams['plotter.vector.arrowsize'] = 200
        rcParams['plotter.maps.lonlatbox'] = 'Europe'
        super(CircumpolarCombinedPlotterTest, cls).setUpClass()

    def ref_map_grid(self, close=True):
        """Create reference file for xgrid formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.xgrid` (and others)
        formatoption"""
        sp = self.plot()
        sp.update(xgrid=False, ygrid=False)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid1')))
        sp.update(xgrid='rounded', ygrid=['data', 1000])
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid2')))
        sp.update(xgrid=True, ygrid=True, grid_color='w')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid3')))
        sp.update(xgrid=True, ygrid=True, grid_color='k', grid_settings={
            'linestyle': 'dotted'})
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid4')))
        if close:
            sp.close(True, True, True)

    def test_bounds(self):
        """Test bounds formatoption"""
        # test bounds of scalar field
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.round(np.linspace(250, 290, 11, endpoint=True), 2).tolist())
        self.update(bounds='minmax')
        bounds = [250.63, 254.38, 258.12, 261.87, 265.62, 269.36, 273.11,
                  276.85, 280.6, 284.35, 288.09]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(255, 290, 5, endpoint=True).tolist())

        # test vector bounds
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(),
            np.linspace(0, 15, 11, endpoint=True).tolist())
        self.update(vbounds='minmax')
        bounds = [0.71, 1.74, 2.76, 3.79, 4.81, 5.84, 6.86, 7.89, 8.92, 9.94,
                  10.97]
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(), bounds)
        self.update(vbounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(),
            np.round(np.linspace(1.0, 9.5, 5, endpoint=True), 2).tolist())

    @property
    def _minmax_cticks(self):
        if not self.vector_mode:
            arr = self.plotter.plot_data[0].values
            arr = arr[~np.isnan(arr)]
            return np.round(
                np.linspace(arr.min(), arr.max(), 11, endpoint=True),
                decimals=2).tolist()
        arr = self.plotter.plot_data[1].values
        speed = (arr[0]**2 + arr[1]**2) ** 0.5
        speed = speed[~np.isnan(speed)]
        return np.round(
            np.linspace(speed.min(), speed.max(), 11, endpoint=True),
            decimals=2).tolist()

    def test_grid(self, *args):
        """Test xgrid, ygrid, grid_color, grid_labels, grid_settings fmts"""
        self.update(xgrid=False, ygrid=False)
        self.compare_figures(next(iter(args), self.get_ref_file('grid1')))
        self.update(xgrid='rounded', ygrid=['data', 1000])
        self.compare_figures(next(iter(args), self.get_ref_file('grid2')))
        self.update(xgrid=True, ygrid=True, grid_color='w')
        self.compare_figures(next(iter(args), self.get_ref_file('grid3')))
        self.update(xgrid=True, ygrid=True, grid_color='k',
                    grid_settings={'linestyle': 'dotted'})
        self.compare_figures(next(iter(args), self.get_ref_file('grid4')))

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            arr = data if data.ndim == 2 else data[0]
            return coord.values[~np.isnan(arr.values)]
        self.update(lonlatbox='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-55.6, 120.6,  -24.4, 85.9),
            repeat(1), repeat("Failed to set the extent to Europe and India!"))
            ))
        # test whether latitudes and longitudes succeded
        msg = "Failed to fit into lonlatbox limits for %s of %s."
        if isinstance(self.plotter.plot_data, InteractiveList):
            all_data = self.plotter.plot_data
        else:
            all_data = [self.plotter.plot_data]
        for data in all_data:
            self.assertGreaterEqual(get_unmasked(data.longitude).min(), -55.6,
                                    msg=msg % ('longitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.longitude).max(), 120.6,
                                 msg=msg % ('longitude', 'maximum'))
            self.assertGreaterEqual(get_unmasked(data.latitude).min(), -24.4,
                                    msg=msg % ('latitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.latitude).max(), 85.9,
                                 msg=msg % ('latitude', 'maximum'))
        self.compare_figures(next(iter(args), self.get_ref_file('lonlatbox')))

    def test_map_extent(self, *args):
        """Test map_extent formatoption"""
        self.update(map_extent='Europe|India')
        self.compare_figures(next(iter(args), self.get_ref_file('map_extent')))

    @unittest.skip('Not implemented for circumpolar grids')
    def ref_datagrid(self):
        pass

    @unittest.skip('Not implemented for circumpolar grids')
    def test_datagrid(self):
        pass

    @unittest.skip('Not implemented for circumpolar grids')
    def test_density(self):
        pass

    @unittest.skip('Not implemented for circumpolar grids')
    def ref_density(self):
        pass


class IconFieldPlotterTest(FieldPlotterTest):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class for icon grid"""

    grid_type = 'icon'

    ncfile = os.path.join(bt.test_dir, 'icon_test.nc')

    def test_bounds(self):
        """Test bounds formatoption"""
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(240, 310, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [240.95, 247.79, 254.62, 261.46, 268.29, 275.13, 281.96,
                  288.8, 295.63, 302.46, 309.3]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(255, 305, 5, endpoint=True).tolist())

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            return coord.values[~np.isnan(data.values)]
        self.update(lonlatbox='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-32.0, 97.0, -8.0, 81.0),
            repeat(5), repeat("Failed to set the extent to Europe and India!"))
            ))
        # test whether latitudes and longitudes succeded
        msg = "Failed to fit into lonlatbox limits for %s of %s."
        if isinstance(self.plotter.plot_data, InteractiveList):
            all_data = self.plotter.plot_data
        else:
            all_data = [self.plotter.plot_data]
        for data in all_data:
            self.assertGreaterEqual(get_unmasked(data.clon).min(), -32.0,
                                    msg=msg % ('longitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.clon).max(), 97.0,
                                 msg=msg % ('longitude', 'maximum'))
            self.assertGreaterEqual(get_unmasked(data.clat).min(), -8.0,
                                    msg=msg % ('latitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.clat).max(), 81.0,
                                 msg=msg % ('latitude', 'maximum'))
        self.compare_figures(next(iter(args), self.get_ref_file('lonlatbox')))


class IconVectorPlotterTest(VectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class for icon grid"""

    grid_type = 'icon'

    ncfile = os.path.join(bt.test_dir, 'icon_test.nc')

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            return coord.values[~np.isnan(data[0].values)]
        self.update(lonlatbox='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-32.0, 97.0, -8.0, 81.0),
            repeat(5), repeat("Failed to set the extent to Europe and India!"))
            ))
        # test whether latitudes and longitudes succeded
        msg = "Failed to fit into lonlatbox limits for %s of %s."
        if isinstance(self.plotter.plot_data, InteractiveList):
            all_data = self.plotter.plot_data
        else:
            all_data = [self.plotter.plot_data]
        for data in all_data:
            self.assertGreaterEqual(get_unmasked(data.clon).min(), -32.0,
                                    msg=msg % ('longitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.clon).max(), 97.0,
                                 msg=msg % ('longitude', 'maximum'))
            self.assertGreaterEqual(get_unmasked(data.clat).min(), -8.0,
                                    msg=msg % ('latitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.clat).max(), 81.0,
                                 msg=msg % ('latitude', 'maximum'))
        self.compare_figures(next(iter(args), self.get_ref_file('lonlatbox')))

    @unittest.skip("Density for quiver plots of unstructered data is not "
                   "supported!")
    def ref_density(self):
        pass

    @unittest.skip("Density for quiver plots of unstructered data is not "
                   "supported!")
    def test_density(self):
        pass

    def test_bounds(self):
        """Test bounds formatoption"""
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(0, 15, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [0.24,  1.31,  2.38,  3.45,  4.51,  5.58,  6.65,  7.72,
                  8.79,  9.85, 10.92]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.round(np.linspace(1.0, 9.5, 5, endpoint=True), 2).tolist())


class IconCombinedPlotterTest(CombinedPlotterTest):
    """Test :class:`psyplot.plotter.maps.CombinedPlotter` class for icon grid
    """

    grid_type = 'icon'

    ncfile = os.path.join(bt.test_dir, 'icon_test.nc')

    @unittest.skip("Density for quiver plots of unstructered data is not "
                   "supported!")
    def ref_density(self):
        pass

    @unittest.skip("Density for quiver plots of unstructered data is not "
                   "supported!")
    def test_density(self):
        pass

    def test_bounds(self):
        """Test bounds formatoption"""
        # test bounds of scalar field
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(250, 290, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [252.61, 256.22, 259.83, 263.43, 267.04, 270.65, 274.26,
                  277.87, 281.48, 285.09, 288.7]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(255, 290, 5, endpoint=True).tolist())

        # test vector bounds
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(),
            np.linspace(0, 15, 11, endpoint=True).tolist())
        self.update(vbounds='minmax')
        bounds = [0.24,  1.31,  2.38,  3.45,  4.51,  5.58,  6.65,  7.72,
                  8.79,  9.85, 10.92]
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(), bounds)
        self.update(vbounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(),
            np.round(np.linspace(1.0, 9.5, 5, endpoint=True), 2).tolist())

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            arr = data if data.ndim == 1 else data[0]
            return coord.values[~np.isnan(arr.values)]
        self.update(lonlatbox='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-32.0, 97.0, -8.0, 81.0),
            repeat(5), repeat("Failed to set the extent to Europe and India!"))
            ))
        # test whether latitudes and longitudes succeded
        msg = "Failed to fit into lonlatbox limits for %s of %s."
        if isinstance(self.plotter.plot_data, InteractiveList):
            all_data = self.plotter.plot_data
        else:
            all_data = [self.plotter.plot_data]
        for data in all_data:
            self.assertGreaterEqual(get_unmasked(data.clon).min(), -32.0,
                                    msg=msg % ('longitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.clon).max(), 97.0,
                                 msg=msg % ('longitude', 'maximum'))
            self.assertGreaterEqual(get_unmasked(data.clat).min(), -8.0,
                                    msg=msg % ('latitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.clat).max(), 81.0,
                                 msg=msg % ('latitude', 'maximum'))
        self.compare_figures(next(iter(args), self.get_ref_file('lonlatbox')))

    @property
    def _minmax_cticks(self):
        if not self.vector_mode:
            arr = self.plotter.plot_data[0].values
            arr = arr[~np.isnan(arr)]
            return np.round(
                np.linspace(arr.min(), arr.max(), 11, endpoint=True),
                decimals=2).tolist()
        arr = self.plotter.plot_data[1].values
        speed = (arr[0]**2 + arr[1]**2) ** 0.5
        speed = speed[~np.isnan(speed)]
        return np.round(
            np.linspace(speed.min(), speed.max(), 11, endpoint=True),
            decimals=2).tolist()


class FieldPlotterTest2D(tb.TestBase2D, FieldPlotterTest):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class without time and
    vertical dimension"""

    var = 't2m_2d'


class VectorPlotterTest2D(tb.TestBase2D, VectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class without time and
    vertical dimension"""

    var = ['u_2d', 'v_2d']


class StreamVectorPlotterTest2D(tb.TestBase2D, StreamVectorPlotterTest):
    """Test case for stream plot of :class:`psyplot.plotter.maps.VectorPlotter`
    without time and vertical dimension"""

    var = ['u_2d', 'v_2d']


class CombinedPlotterTest2D(tb.TestBase2D, CombinedPlotterTest):
    """Test :class:`psyplot.plotter.maps.CombinedPlotter` class without time and
    vertical dimension"""

    var = ['t2m', ['u_2d', 'v_2d']]

    def _label_test(self, key, label_func, has_time=None):
        if has_time is None:
            has_time = not bool(self.vector_mode)
        CombinedPlotterTest._label_test(
            self, key, label_func, has_time=has_time)


class CircumpolarFieldPlotterTest2D(
        tb.TestBase2D, CircumpolarFieldPlotterTest):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class without time and
    vertical dimension for circumpolar grids"""

    var = 't2m_2d'


class CircumpolarVectorPlotterTest2D(
        tb.TestBase2D, CircumpolarVectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class without time and
    vertical dimension for circumpolar grids"""

    var = ['u_2d', 'v_2d']


class CircumpolarCombinedPlotterTest2D(
        tb.TestBase2D, CircumpolarCombinedPlotterTest):
    """Test :class:`psyplot.plotter.maps.CombinedPlotter` class without time and
    vertical dimension for circumpolar grids"""

    var = ['t2m', ['u_2d', 'v_2d']]

    def _label_test(self, key, label_func, has_time=None):
        if has_time is None:
            has_time = not bool(self.vector_mode)
        CombinedPlotterTest._label_test(
            self, key, label_func, has_time=has_time)


class IconFieldPlotterTest2D(tb.TestBase2D, IconFieldPlotterTest):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class for icon grid
    without time and vertical dimension"""

    var = 't2m_2d'


class IconVectorPlotterTest2D(tb.TestBase2D, IconVectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class for icon grid
    without time and vertical dimension"""

    var = ['u_2d', 'v_2d']


class IconCombinedPlotterTest2D(tb.TestBase2D, IconCombinedPlotterTest):
    """Test :class:`psyplot.plotter.maps.CombinedPlotter` class for icon grid
    without time and vertical dimension"""

    var = ['t2m', ['u_2d', 'v_2d']]

    def _label_test(self, key, label_func, has_time=None):
        if has_time is None:
            has_time = not bool(self.vector_mode)
        CombinedPlotterTest._label_test(
            self, key, label_func, has_time=has_time)


tests2d = [FieldPlotterTest2D, VectorPlotterTest2D, StreamVectorPlotterTest2D,
           CombinedPlotterTest2D, IconFieldPlotterTest2D,
           IconVectorPlotterTest2D, IconCombinedPlotterTest2D,
           CircumpolarCombinedPlotterTest2D, CircumpolarFieldPlotterTest2D,
           CircumpolarVectorPlotterTest2D]

# skip the reference creation functions of the 2D Plotter tests
for cls in tests2d:
    skip_msg = "Reference figures for this class are created by the %s" % (
        cls.__name__[:-2])
    for funcname in filter(lambda s: s.startswith('ref'), dir(cls)):
        setattr(cls, funcname, unittest.skip(skip_msg)(lambda self: None))
del cls


if __name__ == '__main__':
    bt.RefTestProgram()
