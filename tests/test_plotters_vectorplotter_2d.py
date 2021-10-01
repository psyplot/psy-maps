"""Test module of the :mod:`psyplot.plotter.maps` module"""
import unittest
import test_base as tb
import test_plotters_vectorplotter as tpv


class VectorPlotterTest2D(tb.TestBase2D, tpv.VectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class without time and
    vertical dimension"""

    var = ['u_2d', 'v_2d']


if __name__ == '__main__':
    unittest.main()
