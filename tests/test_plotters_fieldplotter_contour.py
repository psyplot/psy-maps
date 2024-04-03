"""Test module of the :mod:`psyplot.plotter.maps` module"""

# SPDX-FileCopyrightText: 2016-2024 University of Lausanne
# SPDX-FileCopyrightText: 2020-2021 Helmholtz-Zentrum Geesthacht
# SPDX-FileCopyrightText: 2021-2024 Helmholtz-Zentrum hereon GmbH
#
# SPDX-License-Identifier: LGPL-3.0-only

import unittest

import test_plotters_fieldplotter as tpf

from psy_maps.plotters import FieldPlotter, rcParams


class FieldPlotterContourTest(tpf.FieldPlotterTest):
    plot_type = "map_contour"

    @classmethod
    def setUpClass(cls):
        plotter = FieldPlotter()
        rcParams[plotter.plot.default_key] = "contour"
        rcParams[plotter.lonlatbox.default_key] = [-180, 180, -90, 90]
        rcParams[plotter.xgrid.default_key] = False
        rcParams[plotter.ygrid.default_key] = False
        # to avoid https://github.com/SciTools/cartopy/issues/1673,
        # we use a global map_extent
        rcParams[plotter.map_extent.default_key] = "global"
        super(FieldPlotterContourTest, cls).setUpClass()

    @unittest.skip("Extend keyword not implemented")
    def test_extend(self):
        pass

    @unittest.skip("miss_color keyword not implemented")
    def test_miss_color(self):
        pass

    @unittest.skip("miss_color keyword not implemented")
    def ref_miss_color(self):
        pass


if __name__ == "__main__":
    unittest.main()
