"""Test module of the :mod:`psyplot.plotter.maps` module"""
import os
import re
import unittest
from functools import wraps
from itertools import chain
import numpy as np
from psyplot.utils import _TempBool
from psy_maps.plotters import rcParams, CombinedPlotter
import _base_testing as bt
from psyplot import ArrayList, open_dataset
import psyplot.project as psy
import test_plotters_fieldplotter as tpf
import test_plotters_vectorplotter as tpv

from test_base import bold


def _do_from_both(func):
    """Call the given `func` only from :class:`tpf.FieldPlotterTest and
    :class:`tpv.VectorPlotterTest`"""
    func.__doc__ = getattr(tpv.VectorPlotterTest, func.__name__).__doc__

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        getattr(tpf.FieldPlotterTest, func.__name__)(self, *args, **kwargs)
        if hasattr(self, 'plotter'):
            self.plotter.update(todefault=True)
        with self.vector_mode:
            getattr(tpv.VectorPlotterTest, func.__name__)(self, *args, **kwargs)

    return wrapper


def _in_vector_mode(func):
    """Call the given `func` only from:class:`tpv.VectorPlotterTest`"""
    func.__doc__ = getattr(tpv.VectorPlotterTest, func.__name__).__doc__

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.vector_mode:
            getattr(tpv.VectorPlotterTest, func.__name__)(self, *args, **kwargs)

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


class CombinedPlotterTest(tpv.VectorPlotterTest):
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
        plotter = CombinedPlotter()
        rcParams[plotter.lonlatbox.default_key] = 'Europe'
        rcParams[plotter.cticklabels.default_key] = '%0.6g'
        rcParams[plotter.vcmap.default_key] = 'winter'
        rcParams[plotter.xgrid.default_key] = False
        rcParams[plotter.ygrid.default_key] = False
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
        if 'density' in kwargs:
            sp = psy.plot.mapcombined(self.ncfile, name=[self.var], draw=True)
            sp.update(**kwargs)
        else:
            sp = psy.plot.mapcombined(self.ncfile, name=[self.var], **kwargs)
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
        super(tpv.VectorPlotterTest, self).update(*args, **kwargs)

    def get_ref_file(self, identifier):
        if self.vector_mode:
            identifier += '_vector'
        return super(CombinedPlotterTest, self).get_ref_file(identifier)

    def transpose_data(self):
        self.plotter.data = self._data.copy()
        self.plotter.data[0] = self.plotter.data[0].T
        self.plotter.data[1] = self.plotter.data[1].transpose(
            'variable', *self.plotter.data[1].dims[1:][::-1])

    def test_transpose(self):
        try:
            self.transpose_data()
            self.update(transpose=True)
            for raw, arr in zip(self.plotter.plot.iter_raw_data,
                                self.plotter.plot.iter_data):
                self.assertEqual(arr.dims[-2:], raw.dims[-2:][::-1])
        finally:
            self.plotter.data = self._data

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

    def ref_density(self, close=True, *args, **kwargs):
        """Create reference file for density formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.density` (and others)
        formatoption"""
        # we have to make sure, that the color keyword is not set to 'absolute'
        # because it does not work for quiver plots
        sp = self.plot(density=0.5, color='k', *args, **kwargs)
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
            tpv.VectorPlotterTest.ref_cbarspacing(self, close=close)
        if close:
            sp.close(True, True, True)

    @_do_from_both
    def ref_cmap(self, close=True):
        pass

    def ref_miss_color(self, close=True):
        tpf.FieldPlotterTest.ref_miss_color(self, close)

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
        tpf.FieldPlotterTest.test_miss_color(self, *args, **kwargs)

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
            tpv.VectorPlotterTest.test_cbarspacing(self, *args, **kwargs)

    @_do_from_both
    def test_cmap(self, *args, **kwargs):
        pass

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
        tpf.FieldPlotterTest.test_clabel(self)
        with self.vector_mode:
            self.update(color='absolute')
            self._label_test('vclabel', get_clabel)
            label = get_clabel()
            self.update(vclabelsize=22, vclabelweight='bold',
                        vclabelprops={'ha': 'left'})
            self.assertEqual(label.get_size(), 22)
            self.assertEqual(label.get_weight(), bold)
            self.assertEqual(label.get_ha(), 'left')


class CombinedPlotterTest2D(bt.TestBase2D, CombinedPlotterTest):
    """Test :class:`psyplot.plotter.maps.CombinedPlotter` class without time and
    vertical dimension"""

    var = ['t2m', ['u_2d', 'v_2d']]

    def _label_test(self, key, label_func, has_time=None):
        if has_time is None:
            has_time = not bool(self.vector_mode)
        CombinedPlotterTest._label_test(
            self, key, label_func, has_time=has_time)


if __name__ == '__main__':
    unittest.main()
