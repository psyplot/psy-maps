"""Test module of the :mod:`psyplot.plotter.maps` module"""
import unittest
import test_base as tb
import test_plotters_fieldplotter as tpf


class FieldPlotterTest2D(tb.TestBase2D, tpf.FieldPlotterTest):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class without time and
    vertical dimension"""

    var = 't2m_2d'


if __name__ == '__main__':
    unittest.main()
