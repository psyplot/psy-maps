"""Test module of the :mod:`psyplot.plotter.maps` module"""
import unittest
import test_base as tb
import test_plotters_fieldplotter_circumpolar as tpfc


class CircumpolarFieldPlotterTest2D(
        tb.TestBase2D, tpfc.CircumpolarFieldPlotterTest):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class without time and
    vertical dimension for circumpolar grids"""

    var = 't2m_2d'


if __name__ == '__main__':
    unittest.main()
