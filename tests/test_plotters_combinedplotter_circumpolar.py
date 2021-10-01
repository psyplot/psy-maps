"""Test module of the :mod:`psyplot.plotter.maps` module"""
import os
import unittest
from itertools import starmap, repeat
import numpy as np
import cartopy.crs as ccrs
from psy_maps.plotters import rcParams, InteractiveList
import _base_testing as bt
import test_plotters_combinedplotter as tpc


class CircumpolarCombinedPlotterTest(tpc.CombinedPlotterTest):
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
        self.update(lonlatbox='Europe|India', map_extent='data')
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


class CombinedPlotterTest2D(bt.TestBase2D, tpc.CombinedPlotterTest):
    """Test :class:`psyplot.plotter.maps.CombinedPlotter` class without time and
    vertical dimension"""

    var = ['t2m', ['u_2d', 'v_2d']]

    def _label_test(self, key, label_func, has_time=None):
        if has_time is None:
            has_time = not bool(self.vector_mode)
        tpc.CombinedPlotterTest._label_test(
            self, key, label_func, has_time=has_time)


if __name__ == '__main__':
    unittest.main()
