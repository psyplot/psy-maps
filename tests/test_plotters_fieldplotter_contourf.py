"""Test module of the :mod:`psyplot.plotter.maps` module"""
import unittest
from psy_maps.plotters import FieldPlotter, rcParams
import _base_testing as bt
import test_plotters_fieldplotter as tpf


class FieldPlotterContourFTest(tpf.FieldPlotterTest):

    plot_type = 'map_contourf'

    @classmethod
    def setUpClass(cls):
        plotter = FieldPlotter()
        rcParams[plotter.plot.default_key] = 'contourf'
        rcParams[plotter.lonlatbox.default_key] = [-180, 180, -90, 90]
        rcParams[plotter.xgrid.default_key] = False
        rcParams[plotter.ygrid.default_key] = False
        super(FieldPlotterContourFTest, cls).setUpClass()

    @unittest.skip('Extend keyword not implemented')
    def test_extend(self):
        pass

    @unittest.skip('miss_color keyword not implemented')
    def test_miss_color(self):
        pass

    @unittest.skip('miss_color keyword not implemented')
    def ref_miss_color(self):
        pass


if __name__ == '__main__':
        unittest.main()
