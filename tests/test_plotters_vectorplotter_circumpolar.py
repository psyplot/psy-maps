"""Test module of the :mod:`psyplot.plotter.maps` module"""
import os
import unittest
from itertools import starmap, repeat
import numpy as np
import cartopy.crs as ccrs
from psy_maps.plotters import rcParams, InteractiveList
import _base_testing as bt
import test_plotters_vectorplotter as tpv


class CircumpolarVectorPlotterTest(tpv.VectorPlotterTest):
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
        rcParams['plotter.maps.xgrid'] = False
        rcParams['plotter.maps.ygrid'] = False
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


class CircumpolarVectorPlotterTest2D(
        bt.TestBase2D, CircumpolarVectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class without time and
    vertical dimension for circumpolar grids"""

    var = ['u_2d', 'v_2d']


if __name__ == '__main__':
    unittest.main()
