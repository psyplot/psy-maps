"""Test module of the :mod:`psyplot.plotter.maps` module"""
import os
import unittest
from itertools import starmap, repeat
import numpy as np
import cartopy.crs as ccrs
from psy_maps.plotters import InteractiveList
import _base_testing as bt
import test_plotters_vectorplotter as tpv


class IconVectorPlotterTest(tpv.VectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class for icon grid"""

    grid_type = 'icon'

    ncfile = os.path.join(bt.test_dir, 'icon_test.nc')

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            return coord.values[~np.isnan(data[0].values)]
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

    def test_transpose(self):
        raw = next(self.plotter.plot.iter_raw_data)
        xcoord = raw.psy.get_coord('x').name
        ycoord = raw.psy.get_coord('y').name
        self.update(transpose=True)
        for raw, arr in zip(self.plotter.plot.iter_raw_data,
                            self.plotter.plot.iter_data):
            self.assertEqual(self.plotter.plot.xcoord.name, ycoord)
            self.assertEqual(self.plotter.plot.ycoord.name, xcoord)

    def test_bounds(self):
        """Test bounds formatoption"""
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(0.5, 9.5, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [0.66, 1.51, 2.36, 3.21, 4.05, 4.9, 5.75, 6.59, 7.44, 8.29,
                  9.14]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.round(np.linspace(1.0, 8.0, 5, endpoint=True), 2).tolist())


class IconVectorPlotterTest2D(bt.TestBase2D, IconVectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class for icon grid
    without time and vertical dimension"""

    var = ['u_2d', 'v_2d']


if __name__ == '__main__':
    unittest.main()