"""Test module of the :mod:`psyplot.plotter.maps` module"""
import unittest
import test_base as tb
import test_plotters_vectorplotter_stream as tpvs


class StreamVectorPlotterTest2D(tb.TestBase2D, tpvs.StreamVectorPlotterTest):
    """Test case for stream plot of :class:`psyplot.plotter.maps.VectorPlotter`
    without time and vertical dimension"""

    var = ['u_2d', 'v_2d']


if __name__ == '__main__':
    unittest.main()
