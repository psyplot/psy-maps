"""Test module of the :mod:`psyplot.plotter.baseplotter` module"""
import unittest
import test_base as tb


class BasePlotterTest2D(tb.TestBase2D, tb.BasePlotterTest):
    """Test :class:`psyplot.plotter.baseplotter.BasePlotter` class without time
    and vertical dimension"""

    var = 't2m_2d'


if __name__ == '__main__':
    unittest.main()
