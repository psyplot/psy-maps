"""Test module of the :mod:`psyplot.plotter.maps` module"""
import os
import unittest
from itertools import starmap, repeat
import numpy as np
import cartopy.crs as ccrs
from psy_maps.plotters import InteractiveList
import _base_testing as bt
import test_plotters_combinedplotter as tpc


class IconCombinedPlotterTest(tpc.CombinedPlotterTest):
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
        bounds = [253.62, 257.12, 260.62, 264.12, 267.63, 271.13, 274.63,
                  278.13, 281.64, 285.14, 288.64]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(260, 290, 5, endpoint=True).tolist())

        # test vector bounds
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(),
            np.linspace(0.5, 9.5, 11, endpoint=True).tolist())
        self.update(vbounds='minmax')
        bounds = [0.66, 1.51, 2.36, 3.21, 4.05, 4.9, 5.75, 6.59, 7.44, 8.29,
                  9.14]
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(), bounds)
        self.update(vbounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(),
            np.round(np.linspace(1.0, 8.0, 5, endpoint=True), 2).tolist())

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            arr = data if data.ndim == 1 else data[0]
            return coord.values[~np.isnan(arr.values)]
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

    def test_transpose(self):
        raw = next(self.plotter.plot.iter_raw_data)
        xcoord = raw.psy.get_coord('x').name
        ycoord = raw.psy.get_coord('y').name
        self.plotter.update(transpose=True)

        for raw, arr in zip(self.plotter.plot.iter_raw_data,
                            self.plotter.plot.iter_data):
            self.assertEqual(self.plotter.plot.xcoord.name, ycoord)
            self.assertEqual(self.plotter.plot.ycoord.name, xcoord)

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


if __name__ == '__main__':
    unittest.main()
