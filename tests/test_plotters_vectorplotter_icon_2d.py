"""Test module of the :mod:`psyplot.plotter.maps` module"""
import unittest
import test_base as tb
import test_plotters_vectorplotter_icon as tpvi


class IconVectorPlotterTest2D(tb.TestBase2D, tpvi.IconVectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class for icon grid
    without time and vertical dimension"""

    var = ['u_2d', 'v_2d']


if __name__ == '__main__':
    unittest.main()
