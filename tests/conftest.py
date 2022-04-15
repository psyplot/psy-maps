"""pytest configuration file for psy-maps."""

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
import pytest
import os.path as osp

try:
    # make sure we import QtWebEngineWidgets at the start
    import psyplot_gui.compat.qtcompat
except ImportError:
    pass

def pytest_addoption(parser):
    group = parser.getgroup("psyplot", "psyplot specific options")
    group.addoption(
        '--ref', help='Create reference figures instead of running the tests',
        action='store_true')


def pytest_configure(config):
    if config.getoption('ref'):
        import unittest
        unittest.TestLoader.testMethodPrefix = 'ref'


@pytest.fixture
def regular_test_file():
    return osp.join(osp.dirname(__file__), "test-t2m-u-v.nc")
