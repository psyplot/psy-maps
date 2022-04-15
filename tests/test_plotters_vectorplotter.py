"""Test module of the :mod:`psyplot.plotter.maps` module"""

# Disclaimer
# ----------
#
# Copyright (C) 2021 Helmholtz-Zentrum Hereon
# Copyright (C) 2020-2021 Helmholtz-Zentrum Geesthacht
# Copyright (C) 2016-2021 University of Lausanne
#
# This file is part of psy-maps and is released under the GNU LGPL-3.O license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License version 3.0 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU LGPL-3.0 license for more details.
#
# You should have received a copy of the GNU LGPL-3.0 license
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import unittest
import numpy as np
from psy_maps.plotters import VectorPlotter, rcParams
import _base_testing as bt
from psyplot import ArrayList, open_dataset
import psyplot.project as psy
import test_plotters_fieldplotter as tpf


class VectorPlotterTest(tpf.FieldPlotterTest, tpf.MapReferences):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class"""

    plot_type = 'mapvector'

    var = ['u', 'v']

    def plot(self, **kwargs):
        sp = psy.plot.mapvector(self.ncfile, name=[self.var], **kwargs)
        return sp

    @unittest.skip("miss_color formatoption not implemented")
    def ref_miss_color(self, close=True):
        pass

    def ref_arrowsize(self, close=True):
        """Create reference file for arrowsize formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.arrowsize` (and others)
        formatoption"""
        sp = self.plot(arrowsize=100.0)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('arrowsize')))
        if close:
            sp.close(True, True, True)

    def ref_density(self, close=True):
        """Create reference file for density formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.density` (and others)
        formatoption"""
        sp = self.plot()
        # We do not include the density in the initial plot because there the
        # map_extent is not already considered
        sp.update(density=0.5)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('density')))
        if close:
            sp.close(True, True, True)

    @classmethod
    def setUpClass(cls):
        cls.ds = open_dataset(cls.ncfile)
        plotter = VectorPlotter()
        rcParams[plotter.lonlatbox.default_key] = 'Europe'
        # to make sure, we do not have problems with slightly differing
        # axes changes
        rcParams[plotter.map_extent.default_key] = 'data'
        rcParams[plotter.color.default_key] = 'absolute'
        rcParams[plotter.cticklabels.default_key] = '%0.6g'
        rcParams[plotter.xgrid.default_key] = False
        rcParams[plotter.ygrid.default_key] = False
        cls.data = ArrayList.from_dataset(
            cls.ds, t=0, z=0, name=[cls.var], auto_update=True)[0]
        cls.data.attrs['long_name'] = 'absolute wind speed'
        cls.data.name = 'wind'
        cls.plotter = VectorPlotter(cls.data)
        cls.create_dirs()
        cls._color_fmts = cls.plotter.fmt_groups['colors']

    @unittest.skip("Not supported")
    def test_maskless(self):
        pass

    @unittest.skip("Not supported")
    def test_maskgreater(self):
        pass

    @unittest.skip("Not supported")
    def test_maskleq(self):
        pass

    @unittest.skip("Not supported")
    def test_maskgeq(self):
        pass

    @unittest.skip("Not supported")
    def test_maskbetween(self):
        pass

    @unittest.skip("Not supported")
    def test_miss_color(self):
        pass

    def transpose_data(self):
        self.plotter.data = self.plotter.data.transpose(
            'variable', *self.data.dims[1:][::-1])
        self.plotter.data.psy.base = self.data.psy.base

    def test_bounds(self):
        """Test bounds formatoption"""
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(0, 15, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [0.66, 1.74, 2.81, 3.89, 4.96, 6.04, 7.11, 8.19, 9.26, 10.34,
                  11.41]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(1.0, 10.0, 5, endpoint=True).tolist())

    def test_cbarspacing(self, *args):
        """Test cbarspacing formatoption"""
        self.update(
            cbarspacing='proportional', cticks='rounded', color='absolute',
            bounds=np.arange(0, 1.45, 0.1).tolist() + np.linspace(
                    1.5, 13.5, 7, endpoint=True).tolist() + np.arange(
                        13.6, 15.05, 0.1).tolist())
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('cbarspacing')))

    def test_arrowsize(self, *args):
        """Test arrowsize formatoption"""
        self.update(arrowsize=100.0)
        self.compare_figures(next(iter(args), self.get_ref_file('arrowsize')))

    def test_density(self, *args):
        """Test density formatoption"""
        self.update(density=0.5)
        self.compare_figures(next(iter(args), self.get_ref_file('density')))

    @property
    def _minmax_cticks(self):
        speed = (self.plotter.plot_data.values[0]**2 +
                 self.plotter.plot_data.values[1]**2) ** 0.5
        speed = speed[~np.isnan(speed)]
        return np.round(
            np.linspace(speed.min(), speed.max(), 11, endpoint=True),
            decimals=2).tolist()


class VectorPlotterTest2D(bt.TestBase2D, VectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class without time and
    vertical dimension"""

    var = ['u_2d', 'v_2d']


def test_density_with_lonlatbox(regular_test_file):
    """Test that the lonlatbox still works with density.

    see https://github.com/psyplot/psy-maps/issues/43
    """
    sp = psy.plot.mapvector(
        regular_test_file, name=[["u", "v"]], density=0.5, lonlatbox="Europe"
    )
    ax = sp.plotters[0].ax
    xmin, xmax, ymin, ymax = ax.get_extent()
    assert xmax - xmin < 180
    assert ymax - ymin < 90



if __name__ == '__main__':
    unittest.main()
