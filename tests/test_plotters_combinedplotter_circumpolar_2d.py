"""Test module of the :mod:`psyplot.plotter.maps` module"""
import unittest
import test_base as tb
import test_plotters_combinedplotter as tpc
import test_plotters_combinedplotter_circumpolar as tpcc


class CircumpolarCombinedPlotterTest2D(
        tb.TestBase2D, tpcc.CircumpolarCombinedPlotterTest):
    """Test :class:`psyplot.plotter.maps.CombinedPlotter` class without time and
    vertical dimension for circumpolar grids"""

    var = ['t2m', ['u_2d', 'v_2d']]

    def _label_test(self, key, label_func, has_time=None):
        if has_time is None:
            has_time = not bool(self.vector_mode)
        tpc.CombinedPlotterTest._label_test(
            self, key, label_func, has_time=has_time)


if __name__ == '__main__':
    unittest.main()
