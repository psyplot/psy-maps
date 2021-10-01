"""Test module of the :mod:`psyplot.plotter.maps` module"""
import os
import unittest
from itertools import starmap, repeat
import numpy as np
import cartopy.crs as ccrs
from psy_maps.plotters import InteractiveList
import _base_testing as bt
import test_plotters_fieldplotter as tpf


class IconFieldPlotterTest(tpf.FieldPlotterTest):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class for icon grid"""

    grid_type = 'icon'

    ncfile = os.path.join(bt.test_dir, 'icon_test.nc')

    def test_bounds(self):
        """Test bounds formatoption"""
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(240, 310, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [243.76, 250.04, 256.31, 262.58, 268.85, 275.12, 281.39,
                  287.66, 293.94, 300.21, 306.48]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(255, 305, 5, endpoint=True).tolist())

    def test_transpose(self):
        raw = next(self.plotter.plot.iter_raw_data)
        xcoord = raw.psy.get_coord('x').name
        ycoord = raw.psy.get_coord('y').name
        self.plotter.update(transpose=True)

        for raw, arr in zip(self.plotter.plot.iter_raw_data,
                            self.plotter.plot.iter_data):
            self.assertEqual(self.plotter.plot.xcoord.name, ycoord)
            self.assertEqual(self.plotter.plot.ycoord.name, xcoord)

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            return coord.values[~np.isnan(data.values)]
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

    def ref_pole(self):
        """Test whether the grid cells are correctly displayed at the pole"""
        sp = self.plot()
        sp.update(projection="northpole", lonlatbox=[-180, 180, 80, 90],
                  cmap='viridis', datagrid='r-')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('pole')))
        sp.close(True, True, True)

    def test_pole(self):
        """Test whether the grid cells are correctly displayed at the pole"""
        self.update(projection="northpole", lonlatbox=[-180, 180, 80, 90],
                    cmap='viridis', datagrid='r-')
        self.compare_figures(self.get_ref_file('pole'))


if __name__ == '__main__':
    unittest.main()
