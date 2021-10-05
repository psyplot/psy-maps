"""Test module of the :mod:`psyplot.plotter.maps` module"""
import os
import unittest
from itertools import starmap, repeat, chain
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcol
from psy_maps.plotters import FieldPlotter, rcParams, InteractiveList
import test_base as tb
import _base_testing as bt
from psyplot import ArrayList, open_dataset
import psyplot.project as psy

from test_base import bold


class MapReferences:
    """Abstract base class for map reference plots"""

    def ref_datagrid(self, close=True):
        """Create reference file for datagrid formatoption

        Create reference file for
        :attr:`~psyplot.plotter.simple.Simple2DPlotter.datagrid`
        formatoption"""
        if self.plot_type[:6] == 'simple':
            kwargs = dict(xlim=(0, 40), ylim=(0, 40))
        else:
            kwargs = dict(lonlatbox='Europe', map_extent='data')
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
        sp = self.plot(
            cbar=['fb', 'fr', 'fl', 'ft', 'b', 'r'], xgrid=False, ygrid=False
        )
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
        sp.update(lonlatbox='Europe|India', map_extent='data')
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

    def ref_lsm(self):
        """Create reference file for lsm formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.lsm` formatoption"""
        for i, val in enumerate(
                [False, ['110m', 2.0], {'land': '0.5'}, {'ocean': '0.5'},
                 {'land': '0.5', 'ocean': '0.8'},
                 {'land': '0.5', 'coast': 'r'},
                 {'land': '0.5', 'linewidth': 5.0}, {'linewidth': 5.0},
                ], 1):
            with self.plot(lsm=val) as sp:
                sp.export(os.path.join(
                    bt.ref_dir,
                    self.get_ref_file('lsm{}'.format(i if i-1 else ''))))

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
        sp.update(projection='rotated', lonlatbox='Europe', map_extent='data')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('projection4')))
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
        plotter = FieldPlotter()
        rcParams[plotter.cticklabels.default_key] = '%0.9g'
        cls.ds = open_dataset(cls.ncfile)
        cls.data = ArrayList.from_dataset(
            cls.ds, t=0, z=0, name=cls.var, auto_update=True)[0]
        cls.plotter = FieldPlotter(cls.data)
        cls.create_dirs()

    @unittest.skip("axiscolor formatoption not implemented")
    def test_axiscolor(self):
        pass

    def test_background(self):
        self.update(background='0.5')
        bc = mcol.to_rgba(self.plotter.ax.background_patch.get_facecolor())
        self.assertEqual(bc, (0.5, 0.5, 0.5, 1.0))

    def test_extend(self):
        """Test extend formatoption"""
        self.update(extend='both')
        self.assertEqual(self.plotter.cbar.cbars['b'].extend, 'both')
        self.update(extend='min')
        self.assertEqual(self.plotter.cbar.cbars['b'].extend, 'min')
        self.update(extend='neither')
        self.assertEqual(self.plotter.cbar.cbars['b'].extend, 'neither')

    def transpose_data(self):
        self.plotter.data = self.data.T
        self.plotter.data.psy.base = self.data.psy.base

    def test_transpose(self):
        try:
            self.transpose_data()
            self.update(transpose=True)
            for raw, arr in zip(self.plotter.plot.iter_raw_data,
                                self.plotter.plot.iter_data):
                self.assertEqual(arr.dims[-2:], raw.dims[-2:][::-1])
        finally:
            self.plotter.data = self.data

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
        self.assertAlmostArrayEqual(
            list(map(lambda t: float(t.get_text()),
                     cbar.ax.get_xticklabels())),
            cticks, atol=0.1)
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
        self.update(lonlatbox='Europe', datagrid='k-', map_extent='data')
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
        self.update(
            cbar=['fb', 'fr', 'fl', 'ft', 'b', 'r'], xgrid=False, ygrid=False
        )
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
        self.update(lonlatbox='Europe|India', map_extent='data')
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
        self.update(lsm={'land': '0.5'})
        self.compare_figures(next(iter(args), self.get_ref_file('lsm3')))
        self.update(lsm={'ocean': '0.5'})
        self.compare_figures(next(iter(args), self.get_ref_file('lsm4')))
        self.update(lsm={'land': '0.5', 'ocean': '0.8'})
        self.compare_figures(next(iter(args), self.get_ref_file('lsm5')))
        self.update(lsm={'land': '0.5', 'coast': 'r'})
        self.compare_figures(next(iter(args), self.get_ref_file('lsm6')))
        self.update(lsm={'land': '0.5', 'linewidth': 5.0})
        self.compare_figures(next(iter(args), self.get_ref_file('lsm7')))
        self.update(lsm={'linewidth': 5.0})
        self.compare_figures(next(iter(args), self.get_ref_file('lsm8')))

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
        self.update(projection='rotated', lonlatbox='Europe',
                    map_extent='data')
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('projection4')))

    def test_grid(self, *args):
        """Test xgrid, ygrid, grid_color, grid_labels, grid_settings fmts"""
        self.update(xgrid=False, ygrid=False)
        self.compare_figures(next(iter(args), self.get_ref_file('grid1')), tol=10)
        self.update(xgrid='rounded', ygrid=['data', 20])
        self.compare_figures(next(iter(args), self.get_ref_file('grid2')), tol=10)
        self.update(xgrid=True, ygrid=True, grid_color='w')
        self.compare_figures(next(iter(args), self.get_ref_file('grid3')), tol=10)
        self.update(xgrid=True, ygrid=True, grid_color='k',
                    grid_settings={'linestyle': 'dotted'})
        self.compare_figures(next(iter(args), self.get_ref_file('grid4')), tol=10)

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
        self.update(xgrid=True, ygrid=True, grid_labelsize=20)
        try:
            texts = list(chain(self.plotter.xgrid._gridliner.xlabel_artists,
                               self.plotter.ygrid._gridliner.ylabel_artists))
        except AttributeError:
            texts = list(chain(self.plotter.xgrid._gridliner.label_artists,
                               self.plotter.ygrid._gridliner.label_artists))
            texts = [t[-1] for t in texts]
        self.assertEqual([t.get_size() for t in texts], [20] * len(texts))


class FieldPlotterTest2D(bt.TestBase2D, FieldPlotterTest):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class without time and
    vertical dimension"""

    var = 't2m_2d'


def test_lonlatbox_projected():
    sp = psy.plot.mapplot(os.path.join(bt.test_dir, 'Stockholm.nc'),
                            name='Population', transform='moll',
                            map_extent='data')
    ax = sp.plotters[0].ax
    assert (
        np.round(ax.get_extent(ccrs.PlateCarree()), 2).tolist()
        == [17.66, 18.39, 59.1, 59.59]
    )
    sp.update(lonlatbox=[17.8, 18.2, 59.2, 59.4])
    assert (
        np.round(ax.get_extent(ccrs.PlateCarree()), 2).tolist()
        == [17.8, 18.2, 59.2, 59.4]
    )


def test_rotated_pole_poly():
    """Test function for https://github.com/psyplot/psy-maps/issues/28"""
    test_file = os.path.join(bt.test_dir, "rotated-pole-test.nc")
    # select rlon and rlat manually to make sure we do not use the coordinates
    decoder_kws = {"x": "rlon", "y": "rlat"}
    with psy.plot.mapplot(test_file, plot="poly", decoder=decoder_kws) as sp:
        plotter = sp.plotters[0]
        minx, maxx = plotter.ax.get_xlim()
        miny, maxy = plotter.ax.get_ylim()
        assert abs(minx - -32.2) < 2
        assert abs(maxx - 22) < 2
        assert abs(miny - -27.14) < 2
        assert abs(maxy - 25.6) < 2


def test_plot_poly_3D_bounds():
    """Test plotting the polygons with 3D bounds."""
    fname = os.path.join(bt.test_dir, "rotated-pole-test.nc")
    with psy.plot.mapplot(fname, plot='poly') as sp:
        assert sp[0].ndim == 2
        plotter = sp.plotters[0]
        xmin, xmax = plotter.ax.get_xlim()
        ymin, ymax = plotter.ax.get_ylim()
        assert abs(xmax - xmin - 53) < 2
        assert abs(ymax - ymin - 52) < 2


def test_datagrid_3D_bounds():
    """Test plotting the datagrid with 3D bounds."""
    fname = os.path.join(bt.test_dir, "rotated-pole-test.nc")
    with psy.plot.mapplot(fname, datagrid='k-') as sp:
        assert sp[0].ndim == 2
        plotter = sp.plotters[0]
        xmin, xmax = plotter.ax.get_xlim()
        ymin, ymax = plotter.ax.get_ylim()
        assert abs(xmax - xmin - 53) < 2
        assert abs(ymax - ymin - 52) < 2


def test_plot_curvilinear_datagrid(tmpdir):
    """Test if the there is no datagrid plotted over land

    This implicitly checks, if grid cells at the boundary are warped correctly.
    The file ``'curvilinear-with-bounds.nc'`` contains a variable on a
    curvilinear grid that is only defined over the ocean (derived from MPI-OM).
    Within this test, we focus on a region over land far away from
    the ocean (Czech Republic) where there are no grid cells. If the datagrid
    is plotted correctly, it should be all white.
    """
    from matplotlib.testing.compare import compare_images
    fname = os.path.join(bt.test_dir, 'curvilinear-with-bounds.nc')
    # make a white plot without datagrid
    kws = dict(plot=None, xgrid=False, ygrid=False, map_extent='Czech Republic')
    with psy.plot.mapplot(fname, **kws) as sp:
        sp.export(str(tmpdir / "ref.png"))  # this is all white
    # now draw the datagrid, it should still be empty (as the input file only
    # defines the data over the ocean)
    with psy.plot.mapplot(fname, datagrid='k-', **kws) as sp:
        sp.export(str(tmpdir / "test.png"))  # this should be all white, too
    results = compare_images(
        str(tmpdir / "ref.png"), str(tmpdir / "test.png"), tol=1)
    assert results is None, results


if __name__ == '__main__':
    unittest.main()
