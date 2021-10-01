"""Test module of the :mod:`psyplot.plotter.maps` module"""
import unittest
import test_base as tb
import test_plotters_vectorplotter_circumpolar as tpvc


class CircumpolarVectorPlotterTest2D(
        tb.TestBase2D, tpvc.CircumpolarVectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class without time and
    vertical dimension for circumpolar grids"""

    var = ['u_2d', 'v_2d']


if __name__ == '__main__':
    unittest.main()
