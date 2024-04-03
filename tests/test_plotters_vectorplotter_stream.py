"""Test module of the :mod:`psyplot.plotter.maps` module"""

# SPDX-FileCopyrightText: 2016-2024 University of Lausanne
# SPDX-FileCopyrightText: 2020-2021 Helmholtz-Zentrum Geesthacht
# SPDX-FileCopyrightText: 2021-2024 Helmholtz-Zentrum hereon GmbH
#
# SPDX-License-Identifier: LGPL-3.0-only

import os
import unittest

import _base_testing as bt
import test_plotters_vectorplotter as tpv

from psy_maps.plotters import VectorPlotter, rcParams


class StreamVectorPlotterTest(tpv.VectorPlotterTest):
    """Test case for stream plot of :class:`psyplot.plotter.maps.VectorPlotter`"""

    @classmethod
    def setUpClass(cls):
        plotter = VectorPlotter()
        rcParams[plotter.plot.default_key] = "stream"
        rcParams[plotter.xgrid.default_key] = False
        rcParams[plotter.ygrid.default_key] = False
        return super(StreamVectorPlotterTest, cls).setUpClass()

    def get_ref_file(self, identifier):
        return super(StreamVectorPlotterTest, self).get_ref_file(
            identifier + "_stream"
        )

    def ref_arrowsize(self, *args):
        """Create reference file for arrowsize formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.arrowsize` (and others)
        formatoption"""
        sp = self.plot(arrowsize=2.0)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file("arrowsize")))

    def ref_arrowstyle(self, *args):
        """Create reference file for arrowstyle formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.arrowstyle` (and others)
        formatoption"""
        sp = self.plot(arrowsize=2.0, arrowstyle="fancy")
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file("arrowstyle")))

    def test_arrowsize(self, *args):
        """Test arrowsize formatoption"""
        self.update(arrowsize=2.0)
        self.compare_figures(next(iter(args), self.get_ref_file("arrowsize")))

    def test_arrowstyle(self, *args):
        """Test arrowstyle formatoption"""
        self.update(arrowsize=2.0, arrowstyle="fancy")
        self.compare_figures(next(iter(args), self.get_ref_file("arrowstyle")))

    def test_density(self, *args):
        """Test density formatoption"""
        self.update(density=0.5)
        self.compare_figures(next(iter(args), self.get_ref_file("density")))


class StreamVectorPlotterTest2D(bt.TestBase2D, StreamVectorPlotterTest):
    """Test case for stream plot of :class:`psyplot.plotter.maps.VectorPlotter`
    without time and vertical dimension"""

    var = ["u_2d", "v_2d"]


if __name__ == "__main__":
    unittest.main()
