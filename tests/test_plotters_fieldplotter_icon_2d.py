"""Test module of the :mod:`psyplot.plotter.maps` module"""
import unittest
import test_base as tb
import test_plotters_fieldplotter_icon as tpfi


class IconFieldPlotterTest2D(tb.TestBase2D, tpfi.IconFieldPlotterTest):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class for icon grid
    without time and vertical dimension"""

    var = 't2m_2d'


if __name__ == '__main__':
    unittest.main()
