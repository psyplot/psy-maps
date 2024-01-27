"""Test module of the :mod:`psyplot.plotter.maps` module"""

# SPDX-FileCopyrightText: 2016-2024 University of Lausanne
# SPDX-FileCopyrightText: 2020-2021 Helmholtz-Zentrum Geesthacht
# SPDX-FileCopyrightText: 2021-2024 Helmholtz-Zentrum hereon GmbH
#
# SPDX-License-Identifier: LGPL-3.0-only

import os
import unittest
from itertools import repeat, starmap

import _base_testing as bt
import cartopy.crs as ccrs
import numpy as np
import test_plotters_fieldplotter as tpf

from psy_maps.plotters import InteractiveList


class IconEdgeFieldPlotterTest(tpf.FieldPlotterTest):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class for icon grid"""

    grid_type = "icon_edge"

    ncfile = os.path.join(bt.test_dir, "icon_edge_test.nc")

    def test_bounds(self):
        """Test bounds formatoption"""
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(240, 310, 11, endpoint=True).tolist(),
        )
        self.update(bounds="minmax")
        bounds = [
            242.48,
            249.06,
            255.64,
            262.21,
            268.79,
            275.37,
            281.94,
            288.52,
            295.1,
            301.67,
            308.25,
        ]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds
        )
        self.update(bounds=["rounded", 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(255, 305, 5, endpoint=True).tolist(),
        )

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""

        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            return coord.values[~np.isnan(data.values)]

        self.update(lonlatbox="Europe|India", map_extent="data")
        ax = self.plotter.ax
        list(
            starmap(
                self.assertAlmostEqual,
                zip(
                    ax.get_extent(ccrs.PlateCarree()),
                    (-32.0, 97.0, -8.0, 81.0),
                    repeat(5),
                    repeat("Failed to set the extent to Europe and India!"),
                ),
            )
        )
        # test whether latitudes and longitudes succeded
        msg = "Failed to fit into lonlatbox limits for %s of %s."
        if isinstance(self.plotter.plot_data, InteractiveList):
            all_data = self.plotter.plot_data
        else:
            all_data = [self.plotter.plot_data]
        for data in all_data:
            self.assertGreaterEqual(
                get_unmasked(data.elon).min(),
                -32.0,
                msg=msg % ("longitude", "minimum"),
            )
            self.assertLessEqual(
                get_unmasked(data.elon).max(),
                97.0,
                msg=msg % ("longitude", "maximum"),
            )
            self.assertGreaterEqual(
                get_unmasked(data.elat).min(),
                -8.0,
                msg=msg % ("latitude", "minimum"),
            )
            self.assertLessEqual(
                get_unmasked(data.elat).max(),
                81.0,
                msg=msg % ("latitude", "maximum"),
            )
        self.compare_figures(next(iter(args), self.get_ref_file("lonlatbox")))


if __name__ == "__main__":
    unittest.main()
